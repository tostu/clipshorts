"""Real-ffmpeg end-to-end render: synthetic testsrc + sine tone, stubbed detector and Whisper.

Uses `ffmpeg` from PATH, or the binary named in CLIPSHORTS_TEST_FFMPEG (handy when the system
ffmpeg was built without libass). Skips cleanly when the binary lacks a required filter or
encoder: the caption tests need the `ass` filter (libass), everything else only needs the
filters of the non-caption path.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from clipshorts.layout import Box, Face
from clipshorts.media import Tools
from clipshorts.pipeline import run_pipeline
from clipshorts.render import ffmpeg_capabilities, missing_features
from tests.conftest import FakeTools

CORNER = Box(0.04, 0.62, 0.12, 0.2)
BASE_FILTERS = ["crop", "scale", "format", "fps", "split", "vstack", "overlay", "gblur",
                "loudnorm", "aresample", "setsar"]
BASE_ENCODERS = ["libx264", "aac"]


@pytest.fixture(scope="module")
def ffmpeg_bin() -> str:
    exe = os.environ.get("CLIPSHORTS_TEST_FFMPEG") or "ffmpeg"
    if shutil.which(exe) is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")
    missing = missing_features(exe, BASE_FILTERS, BASE_ENCODERS)
    if missing:
        pytest.skip(f"{exe} lacks {', '.join(missing)}")
    return exe


@pytest.fixture
def ass_bin(ffmpeg_bin) -> str:
    if "ass" not in ffmpeg_capabilities(ffmpeg_bin)[0]:
        pytest.skip(f"{ffmpeg_bin} has no 'ass' filter (built without libass): "
                    "caption burn-in not exercised")
    return ffmpeg_bin


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-800:]
    return res


@pytest.fixture(scope="module")
def source_with_audio(ffmpeg_bin, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src") / "av.mp4"
    _run([ffmpeg_bin, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
          "testsrc2=s=1280x720:r=60:d=3", "-f", "lavfi", "-i",
          "sine=frequency=440:sample_rate=44100:d=3", "-c:v", "libx264", "-preset", "ultrafast",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)])
    return p


@pytest.fixture(scope="module")
def source_silent(ffmpeg_bin, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src") / "v.mp4"
    _run([ffmpeg_bin, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
          "testsrc2=s=1280x720:r=25:d=2", "-c:v", "libx264", "-preset", "ultrafast",
          "-pix_fmt", "yuv420p", str(p)])
    return p


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                "-show_format", str(path)]).stdout
    return json.loads(out)


def luma_frame(ffmpeg_bin: str, path: Path, t: float) -> bytes:
    res = subprocess.run(
        [ffmpeg_bin, "-nostdin", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v",
         "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    assert res.returncode == 0 and len(res.stdout) == 1080 * 1920
    return res.stdout


def make_tools(source: Path, words: list[dict], face: Box = CORNER) -> tuple[Tools, FakeTools]:
    """Real probe, frame extraction and render; fake download, YuNet and Whisper."""
    ft = FakeTools()
    ft.analysis_faces = [Face(face, 0.9)]
    ft.transcript = {**ft.transcript, "words": words}

    def download(url, dest, fmt="best", on_progress=None):
        shutil.copy(source, dest)
        return {"title": "Synthetic", "channel": "test", "game": "Testsrc"}

    base = Tools()
    tools = Tools(download=download, fetch_info=ft.fetch_info, probe=base.probe,
                  extract_frames=base.extract_frames, analyze_frames=ft.analyze_frames,
                  transcribe=ft.transcribe, render=base.render)
    return tools, ft


WORDS = [{"word": w, "start": 0.4 + i * 0.4, "end": 0.7 + i * 0.4}
         for i, w in enumerate(["Das", "ist", "schön", "und", "groß!"])]


def check_output(path: Path, *, audio: bool, max_fps: float = 30.0) -> dict:
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    num, den = v["avg_frame_rate"].split("/")
    assert float(num) / float(den) <= max_fps + 0.01
    audios = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if audio:
        assert len(audios) == 1 and audios[0]["codec_name"] == "aac"
    else:
        assert audios == []
    raw = path.read_bytes()
    assert 0 < raw.find(b"moov") < raw.find(b"mdat")  # +faststart
    return info


def run_clip(cfg, db, bus, tools, ffmpeg_bin: str, layout: str | None = None):
    cfg = cfg.model_copy(update={"ffmpeg_bin": ffmpeg_bin})
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345", layout_override=layout)
    res = run_pipeline(c.id, cfg, db, bus, tools)
    return cfg, c, res


@pytest.mark.parametrize("layout", ["fullcam", "stacked", "face", "content"])
def test_end_to_end_each_layout_without_captions(cfg, db, bus, ffmpeg_bin, source_with_audio,
                                                 layout):
    tools, _ = make_tools(source_with_audio, words=[])
    cfg, c, res = run_clip(cfg, db, bus, tools, ffmpeg_bin, layout)
    assert res.status == "done", res.error
    assert res.ran[-1] == "render"
    out = cfg.clip_dir(c.id) / "output.mp4"
    info = check_output(out, audio=True)
    assert abs(float(info["format"]["duration"]) - 3.0) < 0.4
    # 60 fps source is capped to 30
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v["avg_frame_rate"] == "30/1"
    row = db.get_clip(c.id)
    assert row.status == "done" and row.stage == "render" and row.progress == 1.0
    render = db.latest_render(c.id)
    assert render["layout_type"] == layout and render["params"]["captions"] is False
    assert not (cfg.clip_dir(c.id) / "output.mp4.tmp").exists()


def test_source_without_audio_and_below_fps_cap(cfg, db, bus, ffmpeg_bin, source_silent):
    tools, ft = make_tools(source_silent, words=[])
    cfg, c, res = run_clip(cfg, db, bus, tools, ffmpeg_bin, "content")
    assert res.status == "done", res.error
    info = check_output(cfg.clip_dir(c.id) / "output.mp4", audio=False)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v["avg_frame_rate"] == "25/1"  # below the cap: untouched
    assert ft.transcriptions == []  # no audio stream: whisper skipped


def test_layout_override_rerenders_without_retranscribing(cfg, db, bus, ffmpeg_bin,
                                                          source_with_audio):
    tools, ft = make_tools(source_with_audio, words=[])
    cfg, c, res = run_clip(cfg, db, bus, tools, ffmpeg_bin, "content")
    assert res.status == "done", res.error
    first = (cfg.clip_dir(c.id) / "output.mp4").read_bytes()
    db.update_clip(c.id, layout_override="stacked")
    res = run_pipeline(c.id, cfg, db, bus, tools)
    assert res.status == "done" and res.ran == ["layout", "render"]
    assert len(ft.transcriptions) == 1
    assert (cfg.clip_dir(c.id) / "output.mp4").read_bytes() != first
    assert db.latest_render(c.id)["layout_type"] == "stacked"


@pytest.mark.parametrize("layout", ["stacked", "content"])
def test_end_to_end_with_burned_in_captions(cfg, db, bus, ass_bin, source_with_audio, layout):
    tools, _ = make_tools(source_with_audio, words=WORDS)
    cfg_a, a, res = run_clip(cfg, db, bus, tools, ass_bin, layout)
    assert res.status == "done", res.error
    assert db.latest_render(a.id)["params"]["captions"] is True
    out_a = cfg_a.clip_dir(a.id) / "output.mp4"
    check_output(out_a, audio=True)
    ass = (cfg_a.clip_dir(a.id) / "captions.ass").read_text(encoding="utf-8")
    assert "schön" in ass and "groß" in ass  # umlaut / ß survive into the ASS file

    # same clip without captions: the caption zone (about 60-70% down) must differ
    tools_b, _ = make_tools(source_with_audio, words=[])
    db_slug = db.create_clip("https://clips.twitch.tv/zzz99999", "zzz99999",
                             layout_override=layout)
    res = run_pipeline(db_slug.id, cfg_a, db, bus, tools_b)
    assert res.status == "done", res.error
    out_b = cfg_a.clip_dir(db_slug.id) / "output.mp4"
    fa, fb = luma_frame(ass_bin, out_a, 1.2), luma_frame(ass_bin, out_b, 1.2)
    zone = slice(1080 * 1100, 1080 * 1500)
    diff = sum(1 for x, y in zip(fa[zone], fb[zone], strict=True) if abs(x - y) > 60)
    assert diff > 2000, f"captions not visible in render ({diff} differing pixels)"
    # outside the caption zone both renders agree closely (same layout, same encode)
    top = slice(0, 1080 * 500)
    assert sum(1 for x, y in zip(fa[top], fb[top], strict=True) if abs(x - y) > 60) < 500


def test_missing_ass_filter_fails_clearly(cfg, db, bus, ffmpeg_bin, source_with_audio):
    """Without libass the render must fail with a readable message, not a stack trace."""
    if "ass" in ffmpeg_capabilities(ffmpeg_bin)[0]:
        pytest.skip("this ffmpeg has libass")
    tools, _ = make_tools(source_with_audio, words=WORDS)
    cfg, c, res = run_clip(cfg, db, bus, tools, ffmpeg_bin, "content")
    assert res.status == "failed" and res.stage == "render"
    assert "lacks filter ass" in db.get_clip(c.id).error
    assert not (cfg.clip_dir(c.id) / "output.mp4").exists()
