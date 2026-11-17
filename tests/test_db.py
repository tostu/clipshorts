import pytest


def test_wal_and_tables(db):
    assert db._one("PRAGMA journal_mode")[0] == "wal"
    tables = {r["name"] for r in db._all("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"clips", "renders", "settings", "jobs"} <= tables


def test_clip_crud(db):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345", channel="x")
    assert c.status == "queued" and c.language == "auto"
    assert db.get_clip(c.id).slug == "abc12345"
    db.update_clip(c.id, title="T", duration_s=12.5)
    assert db.get_clip(c.id).title == "T"
    assert [x.id for x in db.list_clips()] == [c.id]
    assert db.find_clip_by_slug("abc12345").id == c.id
    assert db.get_clip(999) is None
    with pytest.raises(ValueError):
        db.update_clip(c.id, nonsense=1)


def test_backfill_clip_created_at(db, tmp_path):
    has_meta = db.create_clip("u", "withmeta1")
    no_meta = db.create_clip("u", "nometa111")
    (tmp_path / str(has_meta.id)).mkdir()
    (tmp_path / str(has_meta.id) / "meta.json").write_text(
        '{"twitch": {"timestamp": 1700000000}}')
    assert db.backfill_clip_created_at(tmp_path) == 1
    assert db.get_clip(has_meta.id).clip_created_at.startswith("2023-11-14T22:13:20")
    assert db.get_clip(no_meta.id).clip_created_at is None
    assert db.backfill_clip_created_at(tmp_path) == 0  # idempotent


def test_settings_roundtrip(db):
    assert db.get_setting("k", "d") == "d"
    db.set_setting("k", {"a": 1})
    db.set_setting("k", {"a": 2})
    assert db.get_setting("k") == {"a": 2}


def test_render_row(db):
    c = db.create_clip("u", "slug1234")
    rid = db.add_render(c.id, "fullcam", {"x": 1}, "output.mp4")
    assert rid > 0


def test_job_queue_fifo_and_claim(db):
    a = db.create_clip("u", "aaaaaaaa")
    b = db.create_clip("u", "bbbbbbbb")
    j1 = db.enqueue_job("pipeline", a.id)
    db.enqueue_job("pipeline", b.id)
    claimed = db.claim_next_job()
    assert claimed.id == j1.id and claimed.status == "running" and claimed.attempts == 1
    assert db.claim_next_job().ref_id == b.id
    assert db.claim_next_job() is None
    db.finish_job(j1.id, False, "boom")
    j = db.get_job(j1.id)
    assert j.status == "failed" and j.error == "boom"
    db.finish_job(j1.id, True)
    assert db.get_job(j1.id).status == "done"
