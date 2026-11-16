import json

from clipshorts.config import Config
from clipshorts.layout import Box, Face
from clipshorts.pipeline import LAYOUT_FILE, load_state, run_pipeline

CORNER = Box(0.04, 0.62, 0.12, 0.2)
BIG = Box(0.35, 0.15, 0.3, 0.5)


def _clip(db):
    return db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")


def _layout(cfg, cid):
    return json.loads((cfg.clip_dir(cid) / LAYOUT_FILE).read_text())


def test_layout_stage_writes_json_and_db_row(cfg, db, bus, fake_tools):
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.ran == ["download", "probe", "layout", "transcribe", "captions"]
    assert res.status == "waiting" and res.stage == "render"
    data = _layout(cfg, c.id)
    assert data["type"] == "fullcam" and data["source"] == "auto"
    assert set(data["rects"]) == {"face", "crop"}
    assert data["rects_px"]["crop"][3] == 1080  # source is 1920x1080 in FakeTools.probe
    assert len(data["frames"]) == 8 and data["frames"][0]["file"] == "frames/frame_00.jpg"
    assert data["detected"]["type"] == "fullcam" and data["notes"]
    assert (cfg.clip_dir(c.id) / "frames" / "frame_07.jpg").exists()
    row = db.get_clip(c.id)
    assert row.layout_type == "fullcam" and row.layout_confidence > 0.9
    # sampled between 5% and 95% of the 30 s probe duration
    t = fake_tools.extractions[0]
    assert len(t) == 8 and t[0] == 1.5 and t[-1] == 28.5
    assert "seconds" in load_state(cfg.clip_dir(c.id))["layout"]


def test_layout_stage_classifies_stacked(cfg, db, bus, fake_tools):
    fake_tools.analysis_faces = [Face(CORNER, 0.9)]
    c = _clip(db)
    run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    data = _layout(cfg, c.id)
    # FakeTools gives no motion grid -> stacked, face-focus not evaluated
    assert data["type"] == "stacked" and set(data["rects"]) == {"facecam", "content"}


def test_layout_cached_on_rerun(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert "layout" in res.skipped and len(fake_tools.extractions) == 1


def test_missing_layout_json_reruns_stage(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    (cfg.clip_dir(c.id) / LAYOUT_FILE).unlink()
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == ["layout"]
    # layout.json is gone, so there is nothing to reuse: frames are sampled again
    assert len(fake_tools.extractions) == 2


def test_override_keeps_detected_data_and_reuses_detection(cfg, db, bus, fake_tools):
    fake_tools.analysis_faces = [Face(BIG, 0.9)]
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    assert _layout(cfg, c.id)["type"] == "fullcam"

    db.update_clip(c.id, layout_override="stacked")
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert "layout" in res.ran
    # detection (ffmpeg + YuNet) is not repeated for an override change
    assert len(fake_tools.extractions) == 1 and len(fake_tools.analyses) == 1
    data = _layout(cfg, c.id)
    assert data["source"] == "override" and data["type"] == "stacked"
    assert data["confidence"] == 1.0
    assert set(data["rects"]) == {"facecam", "content"}
    assert data["detected"]["type"] == "fullcam"  # detection result preserved
    assert any("override" in n for n in data["notes"])
    row = db.get_clip(c.id)
    assert row.layout_type == "stacked" and row.layout_confidence == 1.0

    # back to auto
    db.update_clip(c.id, layout_override="auto")
    run_pipeline(c.id, cfg, db, bus, t)
    assert _layout(cfg, c.id)["source"] == "auto"
    assert db.get_clip(c.id).layout_type == "fullcam"


def test_override_content_and_override_without_face(cfg, db, bus, fake_tools):
    fake_tools.analysis_faces = []
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    assert _layout(cfg, c.id)["type"] == "content"
    db.update_clip(c.id, layout_override="stacked")
    run_pipeline(c.id, cfg, db, bus, t)
    data = _layout(cfg, c.id)
    assert data["type"] == "stacked" and data["rects"] == {}
    assert any("no face" in n for n in data["notes"])


def test_threshold_override_reclassifies_without_redetecting(cfg, db, bus, fake_tools, tmp_path):
    fake_tools.analysis_faces = [Face(BIG, 0.9)]
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    cfg2 = Config(data_dir=cfg.data_dir, worker_enabled=False, nice_level=0, _env_file=None,
                  layout_overrides={"fullcam_min_area": 0.5})
    res = run_pipeline(c.id, cfg2, db, bus, t)
    assert "layout" in res.ran and len(fake_tools.extractions) == 1
    assert "fullcam_min_area" in _layout(cfg2, c.id)["thresholds"]
    assert _layout(cfg2, c.id)["thresholds"]["fullcam_min_area"] == 0.5


def test_missing_yunet_model_fails_stage_clearly(cfg, db, bus, tmp_path):
    from clipshorts.media import Tools
    from tests.conftest import FakeTools

    ft = FakeTools()
    tools = Tools(download=ft.download, probe=ft.probe, extract_frames=ft.extract_frames)
    cfg2 = Config(data_dir=cfg.data_dir, worker_enabled=False, nice_level=0, _env_file=None,
                  yunet_model=tmp_path / "missing.onnx")
    c = _clip(db)
    res = run_pipeline(c.id, cfg2, db, bus, tools)  # real default analyzer
    assert res.status == "failed" and res.stage == "layout"
    assert "YuNet model not found" in res.error
    assert db.get_clip(c.id).status == "failed"


def test_analyzer_failure_surfaces(cfg, db, bus, fake_tools):
    fake_tools.fail_analyze = "boom"
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "failed" and res.error == "layout: boom"
