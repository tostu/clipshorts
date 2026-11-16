"""Render: filter-graph snapshots, escaping, progress, timeout, runner, stage wiring.

Golden files live in tests/golden/graph_*.txt; regenerate with UPDATE_GOLDEN=1.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from clipshorts.layout import (
    CONTENT,
    FACE,
    FULLCAM,
    STACKED,
    Box,
    Face,
    LayoutParams,
    rects_for,
    rects_to_json,
    summarize_faces,
)
from clipshorts.media import Cancelled, MediaError
from clipshorts.pipeline import OUTPUT_FILE, run_pipeline
from clipshorts.render import (
    RenderError,
    RenderOptions,
    build_command,
    cover_window,
    escape_filter_value,
    parse_progress_seconds,
    plan_render,
    progress_fraction,
    render_timeout,
    run_ffmpeg,
    to_px,
)

GOLDEN = Path(__file__).parent / "golden"
BIG = Box(0.35, 0.15, 0.3, 0.5)
CORNER = Box(0.04, 0.62, 0.12, 0.2)
PROBE = {"width": 1920, "height": 1080, "fps": 60.0, "has_audio": True}
ASS = Path("/data/clips/1/captions.ass")
FONTS = Path("/srv/clipshorts/fonts")


def layout_json(layout_type: str, face: Box = BIG) -> dict:
    params = LayoutParams()
    summary = summarize_faces([[Face(face)]] * 8, params)
    rects, _ = rects_for(layout_type, summary, (1920, 1080), params)
    return {"type": layout_type, "rects": rects_to_json(rects)}


def golden(name: str, actual: str) -> None:
    path = GOLDEN / f"graph_{name}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(actual + "\n", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == actual + "\n", f"golden mismatch: {path.name}"


CASES = {
    "fullcam": (layout_json(FULLCAM), PROBE, RenderOptions(), ASS),
    "face": (layout_json(FACE, CORNER), PROBE, RenderOptions(), ASS),
    "stacked": (layout_json(STACKED, CORNER), PROBE, RenderOptions(), ASS),
    "stacked_split45": (layout_json(STACKED, CORNER), PROBE,
                        RenderOptions(split_ratio=0.45), ASS),
    "content": (layout_json(CONTENT), PROBE, RenderOptions(), ASS),
    # 30 fps source needs no fps filter; no captions, no audio, quick sync pixel format
    "content_plain": (layout_json(CONTENT), {**PROBE, "fps": 30.0, "has_audio": False},
                      RenderOptions(), None),
    "fullcam_qsv": (layout_json(FULLCAM), PROBE, RenderOptions(encoder="h264_qsv"), ASS),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_graph_golden(name):
    layout, probe, opts, ass = CASES[name]
    plan = plan_render(layout, probe, opts, ass, FONTS)
    golden(name, plan.graph)


def test_graph_shape_invariants():
    for name, (layout, probe, opts, ass) in CASES.items():
        plan = plan_render(layout, probe, opts, ass, FONTS)
        assert plan.graph.count("[vout]") == 1, name
        assert (";[0:a]loudnorm" in plan.graph) == probe["has_audio"], name
        assert ("ass=" in plan.graph) == (ass is not None), name
        assert "\n" not in plan.graph


def test_stacked_split_ratio_sets_heights():
    layout = layout_json(STACKED, CORNER)
    g = plan_render(layout, PROBE, RenderOptions(split_ratio=0.38), ASS, FONTS).graph
    assert "scale=1080:728" in g and "scale=1080:1192" in g
    g = plan_render(layout, PROBE, RenderOptions(split_ratio=0.5), ASS, FONTS).graph
    assert "scale=1080:960" in g
    # absurd values are clamped, never produce a zero-height slot
    g = plan_render(layout, PROBE, RenderOptions(split_ratio=5), ASS, FONTS).graph
    assert "scale=1080:1536" in g


def test_missing_rects_fall_back_to_content():
    plan = plan_render({"type": STACKED, "rects": {}}, PROBE, RenderOptions(), None, None)
    assert plan.layout == CONTENT and plan.requested == STACKED
    assert "gblur" in plan.graph and "lacks rects" in plan.notes[0]
    plan = plan_render({"type": FULLCAM, "rects": {}}, PROBE, RenderOptions(), None, None)
    assert plan.layout == CONTENT


def test_fullcam_without_crop_rect_derives_it_from_the_face():
    layout = {"type": FULLCAM, "rects": {"face": BIG.to_list()}}
    plan = plan_render(layout, PROBE, RenderOptions(), None, None)
    full = plan_render(layout_json(FULLCAM), PROBE, RenderOptions(), None, None)
    assert plan.layout == FULLCAM and plan.graph == full.graph


def test_fps_filter_only_when_source_exceeds_cap():
    layout = layout_json(FULLCAM)
    assert "fps=30" in plan_render(layout, PROBE, RenderOptions(), None, None).graph
    assert "fps=" not in plan_render(layout, {**PROBE, "fps": 30.0}, RenderOptions(), None,
                                     None).graph
    assert "fps=" not in plan_render(layout, {**PROBE, "fps": 25.0}, RenderOptions(), None,
                                     None).graph
    assert "fps=30" in plan_render(layout, {**PROBE, "fps": None}, RenderOptions(), None,
                                   None).graph
    assert "fps=24" in plan_render(layout, PROBE, RenderOptions(fps_cap=24), None, None).graph


def test_plan_errors():
    with pytest.raises(RenderError, match="encoder"):
        plan_render(layout_json(CONTENT), PROBE, RenderOptions(encoder="nvenc"), None, None)
    with pytest.raises(RenderError, match="size unknown"):
        plan_render(layout_json(CONTENT), {"fps": 30}, RenderOptions(), None, None)


def test_plan_params_and_filters():
    plan = plan_render(layout_json(STACKED, CORNER), PROBE, RenderOptions(), ASS, FONTS)
    assert plan.params["layout"] == STACKED and plan.params["captions"] is True
    assert {"ass", "vstack", "split", "loudnorm", "fps", "crop", "scale"} <= set(plan.filters)
    plain = plan_render(layout_json(FULLCAM), {**PROBE, "has_audio": False, "fps": 30},
                        RenderOptions(), None, None)
    assert "ass" not in plain.filters and "loudnorm" not in plain.filters


def test_build_command_x264():
    opts = RenderOptions()
    plan = plan_render(layout_json(CONTENT), PROBE, opts, ASS, FONTS)
    cmd = build_command("ffmpeg", Path("/in/source.mp4"), Path("/out/o.tmp"), plan, opts)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/out/o.tmp"
    assert cmd.count("-filter_complex") == 1 and cmd[cmd.index("-filter_complex") + 1] == plan.graph
    for flag, val in [("-progress", "pipe:1"), ("-c:v", "libx264"), ("-preset", "veryfast"),
                      ("-crf", "20"), ("-pix_fmt", "yuv420p"), ("-c:a", "aac"),
                      ("-b:a", "160k"), ("-movflags", "+faststart"), ("-f", "mp4")]:
        assert cmd[cmd.index(flag) + 1] == val, flag
    assert cmd.count("-map") == 2 and "[vout]" in cmd and "[aout]" in cmd
    # one encode only
    assert cmd.count("-c:v") == 1 and cmd.count("-i") == 1


def test_build_command_no_audio_and_qsv():
    opts = RenderOptions(encoder="h264_qsv", crf=23)
    plan = plan_render(layout_json(CONTENT), {**PROBE, "has_audio": False}, opts, None, None)
    cmd = build_command("ffmpeg", Path("a.mp4"), Path("b"), plan, opts)
    assert "-an" in cmd and "[aout]" not in cmd and "-c:a" not in cmd
    assert cmd[cmd.index("-c:v") + 1] == "h264_qsv"
    assert cmd[cmd.index("-global_quality") + 1] == "23"
    assert "-crf" not in cmd and "format=nv12" in plan.graph


# -- geometry ------------------------------------------------------------------------------
def test_to_px_is_even_and_inside_frame():
    r = to_px(Box(0.123, 0.456, 0.789, 0.5), 1921, 1081)
    assert all(v % 2 == 0 for v in (r.x, r.y, r.w, r.h))
    assert r.right <= 1920 and r.bottom <= 1080
    r = to_px(Box(0.95, 0.95, 0.5, 0.5), 100, 100)  # box hanging over the edge is clamped
    assert r.right <= 100 and r.bottom <= 100 and r.w >= 2 and r.h >= 2


def test_cover_window_keeps_aspect_and_stays_inside():
    rect = to_px(Box(0.0, 0.0, 1.0, 1.0), 1920, 1080)
    w = cover_window(rect, 1080, 1920)
    assert w.h == 1080 and abs(w.w / w.h - 9 / 16) < 0.01
    # focus far right: window clamped to the frame edge
    w = cover_window(rect, 1080, 1920, focus=(1900, 540))
    assert w.right == 1920 and w.x % 2 == 0
    # rect narrower than the target aspect: full width, shorter height
    tall = to_px(Box(0.0, 0.0, 0.2, 0.9), 1920, 1080)
    w = cover_window(tall, 1080, 730)
    assert w.w == tall.w and w.h < tall.h


# -- escaping ------------------------------------------------------------------------------
def test_escape_filter_value_levels():
    assert escape_filter_value("/data/clips/1/captions.ass") == "/data/clips/1/captions.ass"
    # option level: ':' -> '\:'; graph level then doubles that backslash
    assert escape_filter_value("a:b") == "a\\\\:b"
    assert escape_filter_value("it's") == "it\\\\\\'s"
    assert escape_filter_value("a,b;c[d]") == "a\\,b\\;c\\[d\\]"
    assert escape_filter_value("a\\b") == "a\\\\\\\\b"
    assert escape_filter_value("C:\\x") == "C\\\\:\\\\\\\\x"


def test_ass_paths_are_escaped_in_graph():
    weird = Path("/tmp/we:ird, 'dir'/[x];y/captions.ass")
    plan = plan_render(layout_json(FULLCAM), PROBE, RenderOptions(), weird, Path("/f:o,n"))
    g = plan.graph
    assert "ass=filename=/tmp/we\\\\:ird\\, \\\\\\'dir\\\\\\'/\\[x\\]\\;y/captions.ass" in g
    assert ":fontsdir=/f\\\\:o\\,n," in g


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
def test_escaped_path_survives_real_ffmpeg(tmp_path):
    """The escaping must round-trip through ffmpeg's two unquoting levels. `movie` takes its
    path the same way as `ass`, but exists in every ffmpeg build (no libass needed)."""
    weird = tmp_path / "we:ird, 'dir' [x];y=z" / "in put.mp4"
    weird.parent.mkdir()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=64x64:d=0.2:r=10",
         "-pix_fmt", "yuv420p", str(weird)], check=True, capture_output=True)
    graph = f"movie=filename={escape_filter_value(str(weird))},scale=16:16[o]"
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-filter_complex", graph, "-map", "[o]",
         "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# -- progress / timeout --------------------------------------------------------------------
def test_parse_progress_seconds():
    assert parse_progress_seconds("out_time_us=1500000\n") == 1.5
    assert parse_progress_seconds("out_time_ms=2000000") == 2.0  # ffmpeg: also microseconds
    assert parse_progress_seconds("out_time=00:01:02.500000") == 62.5
    assert parse_progress_seconds("out_time=N/A") is None
    assert parse_progress_seconds("out_time_us=N/A") is None
    assert parse_progress_seconds("out_time_us=-9223372036854775807") is None
    assert parse_progress_seconds("frame=12") is None
    assert parse_progress_seconds("progress=continue") is None
    assert parse_progress_seconds("garbage") is None


def test_progress_fraction_clamps():
    assert progress_fraction(5, 20) == 0.25
    assert progress_fraction(50, 20) == 1.0
    assert progress_fraction(1, 0) == 0.0


def test_render_timeout_scales_with_duration():
    assert render_timeout(30) == 240.0  # 8x
    assert render_timeout(120) == 960.0
    assert render_timeout(5) == 120.0  # floor
    assert render_timeout(None) == 120.0
    assert render_timeout(30, factor=4, minimum=10) == 120.0
    assert render_timeout(1, factor=4, minimum=10) == 10.0


# -- runner (a python child stands in for ffmpeg) ------------------------------------------
def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_run_ffmpeg_reports_progress_and_checks_output(tmp_path):
    out = tmp_path / "o.mp4"
    code = (
        "import sys\n"
        f"open({str(out)!r}, 'wb').write(b'data')\n"
        "print('out_time_us=5000000'); print('progress=continue')\n"
        "print('out_time_us=10000000'); print('progress=end')\n"
    )
    seen: list[float] = []
    run_ffmpeg(_py(code), out, duration=10.0, timeout=20, on_progress=seen.append)
    assert seen == [0.5, 0.99, 1.0]


def test_run_ffmpeg_failure_surfaces_stderr_tail_and_removes_partial(tmp_path):
    out = tmp_path / "o.mp4"
    code = (
        f"import sys; open({str(out)!r}, 'wb').write(b'half')\n"
        "sys.stderr.write('line one\\nNo such filter: ass\\n'); sys.exit(3)"
    )
    with pytest.raises(MediaError) as e:
        run_ffmpeg(_py(code), out, duration=1, timeout=20)
    assert "code 3" in str(e.value) and "No such filter: ass" in str(e.value)
    assert not out.exists()


def test_run_ffmpeg_stderr_tail_is_bounded(tmp_path):
    code = "import sys\nfor i in range(500): sys.stderr.write(f'err {i}\\n')\nsys.exit(1)"
    with pytest.raises(MediaError) as e:
        run_ffmpeg(_py(code), tmp_path / "o", duration=1, timeout=20)
    assert "err 499" in str(e.value) and "err 0 " not in str(e.value)
    assert len(str(e.value)) < 1400


def test_run_ffmpeg_does_not_deadlock_on_chatty_stderr_and_stdout(tmp_path):
    out = tmp_path / "o"
    code = (
        "import sys\n"
        "for i in range(20000):\n"
        "    sys.stderr.write('x' * 100 + '\\n'); print('frame=%d' % i)\n"
        f"open({str(out)!r}, 'wb').write(b'ok')\n"
    )
    run_ffmpeg(_py(code), out, duration=1, timeout=60)


def test_run_ffmpeg_timeout_kills_process(tmp_path):
    out = tmp_path / "o"
    out.write_bytes(b"partial")
    with pytest.raises(MediaError, match="timed out after 1s"):
        run_ffmpeg(_py("import time; time.sleep(60)"), out, duration=1, timeout=1)
    assert not out.exists()


def test_run_ffmpeg_cancel_kills_process(tmp_path):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(Cancelled):
        run_ffmpeg(_py("import time; time.sleep(60)"), tmp_path / "o", duration=1, timeout=60,
                   cancel=cancel)


def test_run_ffmpeg_missing_binary_and_empty_output(tmp_path):
    with pytest.raises(RenderError, match="not found"):
        run_ffmpeg(["/nonexistent/ffmpeg"], tmp_path / "o", duration=1, timeout=5)
    with pytest.raises(RenderError, match="no output"):
        run_ffmpeg(_py("pass"), tmp_path / "o", duration=1, timeout=5)


@needs_ffmpeg
def test_missing_filter_gives_clear_error(tmp_path):
    with pytest.raises(RenderError, match="lacks filter nosuchfilter"):
        run_ffmpeg(["ffmpeg", "-version"], tmp_path / "o", duration=1, timeout=5,
                   require_filters=["nosuchfilter"])


# -- stage wiring (fake render tool) -------------------------------------------------------
def _clip(db, **kw):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    return db.update_clip(c.id, **kw) if kw else c


def test_render_stage_builds_one_pass_command(cfg, db, bus, fake_tools):
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "done" and res.ran[-1] == "render"
    call = fake_tools.renders[0]
    cmd = call["cmd"]
    d = cfg.clip_dir(c.id).resolve()
    assert cmd[cmd.index("-i") + 1] == str(d / "source.mp4")
    assert call["out"].endswith("output.mp4.tmp")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"ass=filename={d / 'captions.ass'}" in graph
    assert f"fontsdir={cfg.fonts_dir.resolve()}" in graph
    assert "ass" in call["filters"] and "loudnorm" in call["filters"]
    # default 30 s probe -> 8x timeout, duration passed through for progress
    assert call["duration"] == 30.0 and call["timeout"] == 240.0
    out = d / OUTPUT_FILE
    assert out.read_bytes() == b"fake-output" and not (d / "output.mp4.tmp").exists()
    row = db.latest_render(c.id)
    assert row["layout_type"] == "fullcam" and row["path"].endswith("output.mp4")
    assert row["params"]["layout"] == "fullcam" and row["params"]["captions"] is True


def test_render_stage_skips_ass_when_no_captions(cfg, db, bus, fake_tools):
    fake_tools.transcript = {**fake_tools.transcript, "words": []}
    c = _clip(db)
    run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    call = fake_tools.renders[0]
    assert "ass=" not in call["cmd"][call["cmd"].index("-filter_complex") + 1]
    assert "ass" not in call["filters"]


def test_layout_override_reruns_render_only_downstream(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    db.update_clip(c.id, layout_override="content")
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == ["layout", "render"]
    assert len(fake_tools.transcriptions) == 1 and len(fake_tools.renders) == 2
    graph = fake_tools.renders[1]["cmd"][fake_tools.renders[1]["cmd"].index("-filter_complex") + 1]
    assert "gblur" in graph
    assert db.latest_render(c.id)["layout_type"] == "content"
    # unchanged -> fully cached, no third render
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == [] and len(fake_tools.renders) == 2


def test_render_config_change_reruns_render_only(cfg, db, bus, fake_tools):
    from clipshorts.config import Config

    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    cfg2 = Config(data_dir=cfg.data_dir, worker_enabled=False, nice_level=0, _env_file=None,
                  x264_crf=24)
    res = run_pipeline(c.id, cfg2, db, bus, t)
    assert res.ran == ["render"]
    assert "-crf" in fake_tools.renders[1]["cmd"]
    cmd = fake_tools.renders[1]["cmd"]
    assert cmd[cmd.index("-crf") + 1] == "24"


def test_render_failure_surfaces_error_and_cleans_tmp(cfg, db, bus, fake_tools):
    fake_tools.fail_render = "ffmpeg exited with code 1: Invalid argument"
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "failed" and res.stage == "render"
    row = db.get_clip(c.id)
    assert row.status == "failed"
    assert row.error == "render: ffmpeg exited with code 1: Invalid argument"
    d = cfg.clip_dir(c.id)
    assert not (d / "output.mp4").exists() and not (d / "output.mp4.tmp").exists()
    fake_tools.fail_render = None  # retry resumes at render
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "done" and res.ran == ["render"]


def test_render_cancel_requeues_clip(cfg, db, bus, fake_tools):
    def cancelled(cmd, out_path, **kw):
        out_path.write_bytes(b"half")
        raise Cancelled("render cancelled")

    fake_tools.render = cancelled
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "cancelled" and res.stage == "render"
    assert db.get_clip(c.id).status == "queued"
    assert not (cfg.clip_dir(c.id) / "output.mp4.tmp").exists()
    assert db.latest_render(c.id) is None


def test_progress_is_persisted_and_published(cfg, db, bus, fake_tools):
    c = _clip(db)
    q = bus.subscribe()
    run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    render_progress = [m["data"]["progress"] for m in msgs
                       if m["data"].get("stage") == "render" and m["data"].get("status")
                       == "running"]
    assert render_progress and max(render_progress) <= 1.0
    assert db.get_clip(c.id).progress == 1.0 and db.get_clip(c.id).status == "done"
