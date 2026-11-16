import json

from clipshorts.pipeline import (
    CAPTIONS_FILE,
    LAYOUT_FILE,
    TRANSCRIPT_FILE,
    load_state,
    run_pipeline,
)


def _clip(db, **kw):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    if kw:
        c = db.update_clip(c.id, **kw)
    return c


def test_transcribe_and_captions_write_artifacts(cfg, db, bus, fake_tools):
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "done" and res.stage == "render"
    d = cfg.clip_dir(c.id)
    t = json.loads((d / TRANSCRIPT_FILE).read_text())
    assert t["language"] == "de" and len(t["words"]) == 4
    ass = (d / CAPTIONS_FILE).read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass and ass.count("Dialogue:") == 4
    call = fake_tools.transcriptions[0]
    assert call["audio"].endswith("source.mp4") and call["language"] is None
    assert call["allowed"] == ["de", "en"]
    # prompt = Twitch title, game, channel (filled in by the download/probe stages)
    assert call["prompt"] == "Big Play. Spiel: Tetris. Kanal: somestreamer"
    state = load_state(d)
    assert "seconds" in state["transcribe"] and "seconds" in state["captions"]


def test_manual_language_is_passed_through(cfg, db, bus, fake_tools):
    c = _clip(db, language="en")
    run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert fake_tools.transcriptions[0]["language"] == "en"
    assert json.loads((cfg.clip_dir(c.id) / TRANSCRIPT_FILE).read_text())["language"] == "en"


def test_auto_language_value_means_detect(cfg, db, bus, fake_tools):
    c = _clip(db, language="auto")
    run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert fake_tools.transcriptions[0]["language"] is None


def test_stages_cached_on_rerun(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert "transcribe" in res.skipped and "captions" in res.skipped
    assert len(fake_tools.transcriptions) == 1


def test_layout_override_reruns_only_layout_and_render(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    ass_before = (cfg.clip_dir(c.id) / CAPTIONS_FILE).stat().st_mtime_ns
    state_before = load_state(cfg.clip_dir(c.id))

    db.update_clip(c.id, layout_override="stacked")
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == ["layout", "render"]
    assert res.skipped == ["download", "probe", "transcribe", "captions"]
    assert res.status == "done" and res.stage == "render"
    assert len(fake_tools.transcriptions) == 1
    assert (cfg.clip_dir(c.id) / CAPTIONS_FILE).stat().st_mtime_ns == ass_before
    state = load_state(cfg.clip_dir(c.id))
    assert state["transcribe"]["hash"] == state_before["transcribe"]["hash"]
    assert state["captions"]["hash"] == state_before["captions"]["hash"]
    assert state["layout"]["hash"] != state_before["layout"]["hash"]
    assert json.loads((cfg.clip_dir(c.id) / LAYOUT_FILE).read_text())["type"] == "stacked"

    # and back to auto: still no transcription
    db.update_clip(c.id, layout_override="auto")
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == ["layout", "render"] and len(fake_tools.transcriptions) == 1


def test_render_hash_depends_on_layout_and_captions():
    from clipshorts.pipeline import STAGES

    render = next(s for s in STAGES if s.name == "render")
    assert set(render.after) == {"captions", "layout"}
    assert next(s for s in STAGES if s.name == "transcribe").after == ("probe",)


def test_language_change_reruns_transcribe_and_captions_not_layout(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    db.update_clip(c.id, language="en")
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == ["transcribe", "captions", "render"] and "layout" in res.skipped


def test_captions_rerun_when_style_changes(cfg, db, bus, fake_tools):
    from clipshorts.config import Config

    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    cfg2 = Config(data_dir=cfg.data_dir, worker_enabled=False, nice_level=0, _env_file=None,
                  caption_font_size=70)
    res = run_pipeline(c.id, cfg2, db, bus, t)
    assert res.ran == ["captions", "render"] and "transcribe" in res.skipped
    assert "Style: Default,Poppins,70," in (cfg2.clip_dir(c.id) / CAPTIONS_FILE).read_text()


def test_empty_transcript_gives_valid_empty_ass(cfg, db, bus, fake_tools):
    fake_tools.transcript = {**fake_tools.transcript, "words": []}
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.stage == "render"
    ass = (cfg.clip_dir(c.id) / CAPTIONS_FILE).read_text()
    assert "[Events]" in ass and "Dialogue:" not in ass


def test_no_audio_skips_whisper(cfg, db, bus, fake_tools):
    orig = fake_tools.probe

    def probe(path, ffprobe_bin="ffprobe"):
        return {**orig(path, ffprobe_bin), "has_audio": False, "audio_codec": None}

    fake_tools.probe = probe
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.stage == "render" and fake_tools.transcriptions == []
    t = json.loads((cfg.clip_dir(c.id) / TRANSCRIPT_FILE).read_text())
    assert t["words"] == [] and t["language_source"] == "no-audio"


def test_transcribe_failure_surfaces_and_keeps_earlier_stages(cfg, db, bus, fake_tools):
    fake_tools.fail_transcribe = "model download failed"
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "failed" and res.error == "transcribe: model download failed"
    fake_tools.fail_transcribe = None
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.ran == ["transcribe", "captions", "render"] and len(fake_tools.downloads) == 1
