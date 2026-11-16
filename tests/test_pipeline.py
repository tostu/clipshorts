import json

from clipshorts.pipeline import (
    STAGE_NAMES,
    STAGES,
    Stage,
    load_state,
    params_hash,
    run_pipeline,
)


def test_stage_order():
    assert [s.name for s in STAGES] == STAGE_NAMES == [
        "download", "probe", "layout", "transcribe", "captions", "render"
    ]


def test_params_hash_deterministic_and_sensitive():
    a = params_hash("s", {"x": 1, "y": 2})
    assert a == params_hash("s", {"y": 2, "x": 1})
    assert a != params_hash("s", {"x": 1, "y": 3})
    assert a != params_hash("t", {"x": 1, "y": 2})
    assert a != params_hash("s", {"x": 1, "y": 2}, upstream="abc")


def _clip(db):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    return c


def test_m1_stages_run_and_write_artifacts(cfg, db, bus, fake_tools):
    c = _clip(db)
    q = bus.subscribe()
    res = run_pipeline(c.id, cfg, db, bus, fake_tools.as_tools())
    assert res.status == "done" and res.stage == "render"
    assert res.ran == ["download", "probe", "layout", "transcribe", "captions", "render"]
    d = cfg.clip_dir(c.id)
    assert (d / "source.mp4").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["twitch"]["title"] == "Big Play"
    assert meta["ffprobe"]["width"] == 1920
    row = db.get_clip(c.id)
    assert row.status == "done" and row.stage == "render" and row.progress == 1.0
    assert (d / "output.mp4").exists()
    assert db.latest_render(c.id)["path"].endswith("output.mp4")
    assert row.title == "Big Play" and row.channel == "somestreamer" and row.game == "Tetris"
    assert row.duration_s == 30.0
    assert not q.empty()  # events were published


def test_rerun_is_cached(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert res.ran == []
    assert res.skipped == ["download", "probe", "layout", "transcribe", "captions", "render"]
    assert len(fake_tools.renders) == 1
    assert len(fake_tools.downloads) == 1 and len(fake_tools.probes) == 1


def test_missing_artifact_reruns_stage(cfg, db, bus, fake_tools):
    c = _clip(db)
    t = fake_tools.as_tools()
    run_pipeline(c.id, cfg, db, bus, t)
    (cfg.clip_dir(c.id) / "meta.json").unlink()
    res = run_pipeline(c.id, cfg, db, bus, t)
    assert "probe" in res.ran and "download" in res.skipped


def test_param_change_invalidates_downstream_only(cfg, db, bus):
    c = _clip(db)
    calls: list[str] = []

    def mk(name, art, param):
        def run(ctx):
            calls.append(name)
            (ctx.dir / art).write_text(name)

        return Stage(name, [art], lambda clip, cfg: {"p": param(clip)}, run)

    stages = [
        mk("a", "a.txt", lambda c: "const"),
        mk("b", "b.txt", lambda c: c.layout_override),
        mk("c", "c.txt", lambda c: "const"),
    ]
    res = run_pipeline(c.id, cfg, db, bus, None, stages)
    assert res.status == "done" and calls == ["a", "b", "c"]
    calls.clear()
    # layout override changes b's params: a stays cached, b and c (chained) re-run
    db.update_clip(c.id, layout_override="stacked")
    res = run_pipeline(c.id, cfg, db, bus, None, stages)
    assert calls == ["b", "c"] and res.skipped == ["a"]
    assert db.get_clip(c.id).status == "done" and db.get_clip(c.id).progress == 1.0
    state = load_state(cfg.clip_dir(c.id))
    assert set(state) == {"a", "b", "c"} and "seconds" in state["a"]


def test_download_error_surfaces(cfg, db, bus):
    from tests.conftest import FakeTools

    ft = FakeTools(fail_download="yt-dlp: ERROR: Clip not found")
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, ft.as_tools())
    assert res.status == "failed" and res.stage == "download"
    row = db.get_clip(c.id)
    assert row.status == "failed"
    assert row.error == "download: yt-dlp: ERROR: Clip not found"


def test_probe_error_keeps_download_cached(cfg, db, bus):
    from tests.conftest import FakeTools

    ft = FakeTools(fail_probe="no video stream in file")
    c = _clip(db)
    run_pipeline(c.id, cfg, db, bus, ft.as_tools())
    assert db.get_clip(c.id).error.startswith("probe:")
    ft.fail_probe = None  # retry: download must not repeat
    res = run_pipeline(c.id, cfg, db, bus, ft.as_tools())
    assert res.skipped == ["download"]
    assert res.ran == ["probe", "layout", "transcribe", "captions", "render"]
    assert len(ft.downloads) == 1
    assert db.get_clip(c.id).error is None


def test_unexpected_exception_is_reported(cfg, db, bus):
    def boom(ctx):
        raise KeyError("oops")

    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, None, [Stage("a", ["a"], lambda c, f: {}, boom)])
    assert res.status == "failed"
    assert db.get_clip(c.id).error.startswith("a: KeyError")


def test_stage_without_artifact_fails(cfg, db, bus):
    c = _clip(db)
    res = run_pipeline(c.id, cfg, db, bus, None,
                       [Stage("a", ["nope.txt"], lambda c, f: {}, lambda ctx: None)])
    assert res.status == "failed" and "missing artifact" in res.error
