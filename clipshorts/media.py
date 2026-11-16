"""External-tool calls (yt-dlp, ffprobe) behind small functions so tests can stub them."""

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config


class MediaError(RuntimeError):
    """A download/probe step failed in a way worth showing to the user."""


def ytdlp_download(
    url: str,
    dest: Path,
    fmt: str = "best",
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Download `url` to exactly `dest` (mp4). Returns trimmed yt-dlp info."""
    import yt_dlp  # lazy: keeps import cost and test deps out of the unit-test path

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    opts = {
        "format": fmt,
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            produced = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        raise MediaError(f"yt-dlp: {e}") from e
    if produced != dest:
        if not produced.exists():
            raise MediaError("yt-dlp finished but produced no file")
        produced.replace(dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise MediaError("download produced an empty file")
    cats = info.get("categories") or []
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "creator": info.get("creator"),
        "game": info.get("game") or (cats[0] if cats else None),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "timestamp": info.get("timestamp"),
        "webpage_url": info.get("webpage_url"),
    }


def ffprobe(path: Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Run ffprobe and return a compact summary."""
    try:
        res = subprocess.run(
            [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
             str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError as e:
        raise MediaError(f"{ffprobe_bin} not found") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError("ffprobe timed out") from e
    if res.returncode != 0:
        raise MediaError(f"ffprobe failed: {res.stderr.strip()[:300]}")
    return summarize_probe(json.loads(res.stdout))


def _fps(rate: str | None) -> float | None:
    if not rate or "/" not in rate:
        return None
    n, _, d = rate.partition("/")
    try:
        return round(float(n) / float(d), 3) if float(d) else None
    except ValueError:
        return None


def summarize_probe(raw: dict[str, Any]) -> dict[str, Any]:
    streams = raw.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise MediaError("no video stream in file")
    fmt = raw.get("format", {})
    dur = fmt.get("duration") or v.get("duration")
    return {
        "duration": float(dur) if dur else None,
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": _fps(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name") if a else None,
        "has_audio": a is not None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
    }


@dataclass
class Tools:
    """Injectable external calls. Tests replace the callables."""

    download: Callable[..., dict[str, Any]] = ytdlp_download
    probe: Callable[..., dict[str, Any]] = ffprobe


def default_tools(cfg: Config) -> Tools:
    return Tools()


def apply_nice(level: int) -> None:
    """Lower scheduling priority of the calling thread/process (children inherit)."""
    if level <= 0:
        return
    try:
        os.nice(level)
    except (OSError, AttributeError):
        pass
