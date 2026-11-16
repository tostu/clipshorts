"""Render: pure filter-graph builder plus the ffmpeg runner (progress, timeout, cancel).

One ffmpeg invocation, one `-filter_complex`: layout crop/scale/stack, captions (`ass`) and
loudness all happen in the same graph, so the video is encoded exactly once.

Layout rects come from layout.json as normalized boxes; they are converted to source pixels
here. Every video crop is computed in pixels with the exact aspect ratio of its target slot
("cover" scaling done by cropping first), so `scale` never distorts.
"""

import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .layout import CONTENT, FACE, FULLCAM, STACKED, Box, fullcam_crop
from .media import Cancelled, MediaError

OUT_W, OUT_H = 1080, 1920
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
LOW_RES_UPSCALE = 2.5  # facecam upscale factor above which `low_res_source` is flagged
GRAPH_VERSION = 2  # bump when the graph shape changes so cached outputs are rebuilt
ENCODERS = ("libx264", "h264_qsv")
STDERR_TAIL_LINES = 12
STDERR_TAIL_CHARS = 1200


class RenderError(MediaError):
    """Render could not be planned or run; message is shown to the user."""


@dataclass(frozen=True)
class RenderOptions:
    fps_cap: int = 30
    split_ratio: float = 0.38  # stacked: share of the output height used by the facecam
    encoder: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20  # libx264 CRF; for h264_qsv it is used as global_quality
    blur_sigma: float = 8.0  # content layout background (applied at 1/4 resolution)

    def as_dict(self) -> dict[str, Any]:
        return {"fps_cap": self.fps_cap, "split_ratio": self.split_ratio,
                "encoder": self.encoder, "preset": self.preset, "crf": self.crf,
                "blur_sigma": self.blur_sigma, "graph_version": GRAPH_VERSION}


# -- escaping ------------------------------------------------------------------------------
def _escape(s: str, special: str) -> str:
    return "".join("\\" + c if c in special else c for c in s)


def escape_filter_value(value: str) -> str:
    """Escape `value` for use as an option value inside a -filter_complex string.

    ffmpeg unquotes twice: once for the filtergraph (specials `\\ ' [ ] , ;`) and once for the
    option list (specials `\\ ' :`). The value is escaped for the option level first, then the
    result for the graph level. Verified against ffmpeg with awkward directory names.
    """
    return _escape(_escape(value, "\\':"), "\\'[],;")


# -- geometry ------------------------------------------------------------------------------
@dataclass(frozen=True)
class PxRect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def _even(v: float) -> int:
    return int(v) // 2 * 2


def to_px(box: Box, sw: int, sh: int) -> PxRect:
    """Normalized box -> even pixel rect inside the source frame."""
    fw, fh = max(2, _even(sw)), max(2, _even(sh))
    x = min(_even(round(box.x * sw)), fw - 2)
    y = min(_even(round(box.y * sh)), fh - 2)
    w = max(2, min(_even(round(box.w * sw)), fw - x))
    h = max(2, min(_even(round(box.h * sh)), fh - y))
    return PxRect(x, y, w, h)


def cover_window(rect: PxRect, tw: int, th: int,
                 focus: tuple[float, float] | None = None) -> PxRect:
    """Largest window of aspect tw:th inside `rect`, centered on `focus` (pixels) if given
    else on the rect center, clamped to the rect. Dimensions and offsets are even."""
    aspect = tw / th
    # 1% tolerance: a rect that is 9:16 up to even-pixel rounding keeps its full height
    if rect.w / rect.h >= aspect * 0.99:
        ch, cw = rect.h, min(float(rect.w), rect.h * aspect)
    else:
        cw, ch = rect.w, rect.w / aspect
    cw, ch = max(2, _even(cw)), max(2, _even(ch))
    fx, fy = focus if focus is not None else (rect.x + rect.w / 2, rect.y + rect.h / 2)
    x = min(max(fx - cw / 2, rect.x), rect.right - cw)
    y = min(max(fy - ch / 2, rect.y), rect.bottom - ch)
    return PxRect(_even(x), _even(y), cw, ch)


def contain_window(rect: PxRect, tw: int, th: int, sw: int, sh: int) -> PxRect:
    """Smallest window of aspect tw:th that contains `rect`, centered on it and shifted
    inside the source frame (shrunk, keeping its aspect, if it is larger than the frame)."""
    aspect = tw / th
    if rect.w / rect.h >= aspect:
        cw, ch = float(rect.w), rect.w / aspect
    else:
        cw, ch = rect.h * aspect, float(rect.h)
    k = min(1.0, sw / cw, sh / ch)
    cw, ch = max(2, _even(cw * k)), max(2, _even(ch * k))
    x = min(max(rect.x + rect.w / 2 - cw / 2, 0), max(0, sw - cw))
    y = min(max(rect.y + rect.h / 2 - ch / 2, 0), max(0, sh - ch))
    return PxRect(_even(x), _even(y), cw, ch)


def face_fallback_rect(face: Box, sw: int, sh: int) -> Box:
    """~4:3 (pixels) rect, about 3x face width, centered on the face, clamped to the frame."""
    w_px = min(3 * face.w * sw, float(sw))
    h_px = w_px * 3 / 4
    if h_px > sh:
        h_px = float(sh)
        w_px = h_px * 4 / 3
    w, h = w_px / sw, h_px / sh
    x = min(max(face.cx - w / 2, 0.0), 1.0 - w)
    y = min(max(face.cy - h / 2, 0.0), 1.0 - h)
    return Box(x, y, w, h)


def _crop(r: PxRect) -> str:
    return f"crop={r.w}:{r.h}:{r.x}:{r.y}"


# -- planning ------------------------------------------------------------------------------
@dataclass
class RenderPlan:
    layout: str  # layout actually rendered (may differ from `requested` on fallback)
    requested: str
    graph: str
    has_audio: bool
    has_captions: bool
    filters: list[str]  # filter names the graph needs (capability pre-check)
    notes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # stored in renders.params_json


def _boxes(layout: dict[str, Any]) -> dict[str, Box]:
    out: dict[str, Box] = {}
    for name, v in (layout.get("rects") or {}).items():
        try:
            out[name] = Box.from_list(v)
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _tail(opts: RenderOptions, ass_path: Path | None, fonts_dir: Path | None) -> str:
    parts = []
    if ass_path is not None:
        ass = f"ass=filename={escape_filter_value(str(ass_path))}"
        if fonts_dir is not None:
            ass += f":fontsdir={escape_filter_value(str(fonts_dir))}"
        parts.append(ass)
    parts.append("format=nv12" if opts.encoder == "h264_qsv" else "format=yuv420p")
    return ",".join(parts)


def _src(pre: str) -> str:
    return f"[0:v]{pre}," if pre else "[0:v]"


def plan_render(layout: dict[str, Any], probe: dict[str, Any], opts: RenderOptions,
                ass_path: Path | None = None, fonts_dir: Path | None = None) -> RenderPlan:
    """Build the filter graph for layout.json `layout` and the probed source."""
    if opts.encoder not in ENCODERS:
        raise RenderError(f"unsupported video encoder {opts.encoder!r} "
                          f"(use one of {', '.join(ENCODERS)})")
    sw, sh = probe.get("width"), probe.get("height")
    if not sw or not sh:
        raise RenderError("source size unknown, cannot plan the render")
    requested = layout.get("type") or CONTENT
    boxes = _boxes(layout)
    notes: list[str] = []
    split = min(0.8, max(0.2, opts.split_ratio))

    # a layout whose rects are missing (e.g. override to fullcam on a clip without a face)
    # degrades to the blurred fit instead of failing the clip
    eff = requested
    face = boxes.get("face")
    if eff == FULLCAM and "crop" not in boxes and face is None:
        eff = CONTENT
    elif eff == STACKED and not ("facecam" in boxes and "content" in boxes):
        eff = CONTENT
    elif eff == FACE and "facecam" not in boxes and face is None:
        eff = CONTENT
    elif eff not in (FULLCAM, STACKED, FACE, CONTENT):
        eff = CONTENT
    if eff != requested:
        notes.append(f"layout {requested!r} lacks rects: rendered as {eff!r}")

    src_fps = probe.get("fps")
    need_fps = not src_fps or src_fps > opts.fps_cap + 0.01
    pre = f"fps={opts.fps_cap}" if need_fps else ""
    tail = _tail(opts, ass_path, fonts_dir)
    scale_full = f"scale={OUT_W}:{OUT_H},setsar=1"
    filters = ["crop", "scale", "format"]
    if need_fps:
        filters.append("fps")
    if ass_path is not None:
        filters.append("ass")

    plan_extra: dict[str, Any] = {}

    def focus_of(box: Box | None) -> tuple[float, float] | None:
        return None if box is None else (box.cx * sw, box.cy * sh)

    if eff == FULLCAM:
        rect = boxes.get("crop") or fullcam_crop(face, (sw, sh))  # type: ignore[arg-type]
        win = cover_window(to_px(rect, sw, sh), OUT_W, OUT_H, focus_of(face))
        graph = f"{_src(pre)}{_crop(win)},{scale_full},{tail}[vout]"
    elif eff == FACE:
        cam_box = boxes.get("facecam")
        if cam_box is None:
            cam_box = face_fallback_rect(boxes["face"], sw, sh)
            notes.append("no facecam rect: built one around the face box")
        cam = to_px(cam_box, sw, sh)
        upscale = min(OUT_W / cam.w, OUT_H / cam.h)
        plan_extra["facecam_upscale"] = round(upscale, 2)
        if upscale > LOW_RES_UPSCALE:
            notes.append("low_res_source")
            plan_extra["low_res_source"] = True
        bw, bh = OUT_W // 4, OUT_H // 4
        graph = (
            f"{_src(pre)}split=2[bg][fg];"
            f"[bg]{_crop(cam)},scale={bw}:{bh}:force_original_aspect_ratio=increase,"
            f"crop={bw}:{bh},gblur=sigma={opts.blur_sigma:g},scale={OUT_W}:{OUT_H},"
            f"eq=brightness=-0.2,setsar=1[bgb];"
            f"[fg]{_crop(cam)},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,setsar=1[fgs];"
            f"[bgb][fgs]overlay=x=(W-w)/2:y=(H-h)/2,{tail}[vout]"
        )
        filters += ["split", "gblur", "overlay", "eq"]
    elif eff == STACKED:
        top_h = _even(OUT_H * split)
        bot_h = OUT_H - top_h
        cam = contain_window(to_px(boxes["facecam"], sw, sh), OUT_W, top_h, sw, sh)
        # content is shown whole (fit to width), centered in the bottom slot over a blurred
        # copy of itself; the facecam fills the top slot
        con = to_px(boxes["content"], sw, sh)
        bw, bh = OUT_W // 4, max(2, _even(bot_h // 4))
        graph = (
            f"{_src(pre)}split=3[cam][con][cbg];"
            f"[cam]{_crop(cam)},scale={OUT_W}:{top_h},setsar=1[top];"
            f"[cbg]{_crop(con)},scale={bw}:{bh}:force_original_aspect_ratio=increase,"
            f"crop={bw}:{bh},gblur=sigma={opts.blur_sigma:g},scale={OUT_W}:{bot_h},"
            f"eq=brightness=-0.2,setsar=1[cbb];"
            f"[con]{_crop(con)},scale={OUT_W}:{bot_h}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,setsar=1[cfg];"
            f"[cbb][cfg]overlay=x=(W-w)/2:y=(H-h)/2[bot];"
            f"[top][bot]vstack=inputs=2,{tail}[vout]"
        )
        filters += ["split", "gblur", "overlay", "eq", "vstack"]
    else:  # CONTENT: whole frame, fitted over a blurred cover of itself
        bw, bh = OUT_W // 4, OUT_H // 4
        graph = (
            f"{_src(pre)}split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"gblur=sigma={opts.blur_sigma:g},scale={OUT_W}:{OUT_H},setsar=1[bgb];"
            f"[fg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,setsar=1[fgs];"
            f"[bgb][fgs]overlay=x=(W-w)/2:y=(H-h)/2,{tail}[vout]"
        )
        filters += ["split", "gblur", "overlay"]

    has_audio = bool(probe.get("has_audio"))
    if has_audio:
        graph += f";[0:a]{LOUDNORM}[aout]"
        filters += ["loudnorm", "aresample"]
    plan_params = {**opts.as_dict(), "layout": eff, "requested": requested,
                   "source": [sw, sh], "captions": ass_path is not None,
                   "audio": has_audio, **plan_extra}
    return RenderPlan(eff, requested, graph, has_audio, ass_path is not None,
                      sorted(set(filters)), notes, plan_params)


def build_command(ffmpeg_bin: str, src: Path, out: Path, plan: RenderPlan,
                  opts: RenderOptions) -> list[str]:
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "error", "-nostats", "-y",
           "-progress", "pipe:1", "-i", str(src), "-filter_complex", plan.graph,
           "-map", "[vout]"]
    if plan.has_audio:
        cmd += ["-map", "[aout]"]
    else:
        cmd += ["-an"]
    if opts.encoder == "h264_qsv":
        cmd += ["-c:v", "h264_qsv", "-preset", opts.preset, "-global_quality", str(opts.crf),
                "-look_ahead", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
                "-pix_fmt", "yuv420p"]
    if plan.has_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]
    cmd += ["-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", str(out)]
    return cmd


# -- progress / timeout --------------------------------------------------------------------
_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")


def parse_progress_seconds(line: str) -> float | None:
    """Seconds of output produced, from one `-progress` line, else None.

    Reads `out_time_us` / `out_time_ms` (both microseconds in ffmpeg) or `out_time`
    (HH:MM:SS.ffffff). `N/A` and negative values (ffmpeg's startup placeholder) are ignored.
    """
    key, sep, val = line.strip().partition("=")
    if not sep:
        return None
    val = val.strip()
    if key in ("out_time_us", "out_time_ms"):
        try:
            us = int(val)
        except ValueError:
            return None
        return us / 1e6 if us >= 0 else None
    if key == "out_time":
        m = _TIME_RE.match(val)
        if not m:
            return None
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    return None


def progress_fraction(seconds: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return max(0.0, min(1.0, seconds / duration))


def render_timeout(duration: float | None, factor: float = 8.0, minimum: float = 120.0) -> float:
    """Wall-clock limit for one render: `factor` x clip duration, never below `minimum`."""
    return max(float(minimum), float(duration or 0.0) * float(factor))


# -- capability probing --------------------------------------------------------------------
_FILTER_RE = re.compile(r"^\s*[TSC.]{2,3}\s+(\S+)\s+\S+->\S+")
_ENCODER_RE = re.compile(r"^\s*[VAS][.FSXBD]{5}\s+(\S+)")


@lru_cache(maxsize=8)
def ffmpeg_capabilities(ffmpeg_bin: str) -> tuple[frozenset[str], frozenset[str]]:
    """(filter names, encoder names) of an ffmpeg binary. Raises RenderError if unusable."""
    def listing(flag: str) -> list[str]:
        try:
            res = subprocess.run([ffmpeg_bin, "-hide_banner", flag], capture_output=True,
                                 text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RenderError(f"cannot run {ffmpeg_bin}: {e}") from e
        return res.stdout.splitlines()

    filters = frozenset(m[1] for ln in listing("-filters") if (m := _FILTER_RE.match(ln)))
    encoders = frozenset(m[1] for ln in listing("-encoders") if (m := _ENCODER_RE.match(ln)))
    return filters, encoders


def missing_features(ffmpeg_bin: str, filters: list[str] | tuple[str, ...] = (),
                     encoders: list[str] | tuple[str, ...] = ()) -> list[str]:
    have_f, have_e = ffmpeg_capabilities(ffmpeg_bin)
    return ([f"filter {f}" for f in filters if f not in have_f]
            + [f"encoder {e}" for e in encoders if e not in have_e])


# -- runner --------------------------------------------------------------------------------
def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass


def run_ffmpeg(
    cmd: list[str],
    out_path: Path,
    *,
    duration: float,
    timeout: float,
    on_progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
    require_filters: list[str] | tuple[str, ...] = (),
    require_encoders: list[str] | tuple[str, ...] = (),
) -> None:
    """Run ffmpeg (started with `-progress pipe:1`) to completion.

    Reports progress as a 0..1 fraction of `duration`. A watchdog thread kills the process on
    timeout or when `cancel` is set. Raises MediaError (stderr tail included) on failure or
    timeout and `Cancelled` on cancel; the partial `out_path` is removed in every failure case.
    """
    if require_filters or require_encoders:
        missing = missing_features(cmd[0], require_filters, require_encoders)
        if missing:
            hint = " (libass is needed for captions)" if "filter ass" in missing else ""
            raise RenderError(f"{cmd[0]} lacks {', '.join(missing)}{hint}")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, errors="replace")
    except FileNotFoundError as e:
        raise RenderError(f"{cmd[0]} not found") from e

    tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)

    def drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                tail.append(line)

    reason: list[str] = []
    finished = threading.Event()

    def watchdog() -> None:
        deadline = time.monotonic() + timeout
        while not finished.wait(0.1):
            if cancel is not None and cancel.is_set():
                reason.append("cancelled")
            elif time.monotonic() > deadline:
                reason.append("timeout")
            else:
                continue
            _kill(proc)
            return

    t_err = threading.Thread(target=drain_stderr, daemon=True)
    t_dog = threading.Thread(target=watchdog, daemon=True)
    t_err.start()
    t_dog.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            secs = parse_progress_seconds(line)
            if secs is not None and on_progress is not None:
                on_progress(min(0.99, progress_fraction(secs, duration)))
        rc = proc.wait()
    finally:
        finished.set()
        if proc.poll() is None:
            _kill(proc)
            proc.wait()
        t_err.join(timeout=2)
        t_dog.join(timeout=2)

    if rc == 0 and not reason:
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RenderError("ffmpeg finished but wrote no output")
        if on_progress is not None:
            on_progress(1.0)
        return
    out_path.unlink(missing_ok=True)
    if reason == ["cancelled"]:
        raise Cancelled("render cancelled")
    text = " | ".join(tail)[-STDERR_TAIL_CHARS:]
    if reason == ["timeout"]:
        raise RenderError(f"ffmpeg timed out after {timeout:.0f}s" + (f": {text}" if text else ""))
    raise RenderError(f"ffmpeg exited with code {rc}" + (f": {text}" if text else ""))


__all__ = [
    "ENCODERS", "GRAPH_VERSION", "LOUDNORM", "OUT_H", "OUT_W", "PxRect",
    "RenderError", "RenderOptions", "RenderPlan", "build_command", "cover_window",
    "escape_filter_value", "ffmpeg_capabilities", "missing_features", "parse_progress_seconds",
    "plan_render", "progress_fraction", "render_timeout", "run_ffmpeg", "to_px",
]
