"""Review inbox: default status, migration, approve / reject / reset, sync never re-queues."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from clipshorts.db import DB
from clipshorts.main import create_app
from clipshorts.sync import sync_channel


def item(slug: str, views: int = 10) -> dict:
    return {"slug": slug, "url": f"https://clips.twitch.tv/{slug}", "title": slug,
            "views": views, "duration": 30.0, "timestamp": 1}


def finished(db: DB, cfg, slug: str, channel: str | None = "gronkh"):
    """A done clip with a rendered output file."""
    clip = db.create_clip(f"https://clips.twitch.tv/{slug}", slug, channel=channel)
    db.update_clip(clip.id, status="done", title=slug, layout_type="stacked", duration_s=30.0)
    out = cfg.clip_dir(clip.id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "output.mp4").write_bytes(b"fake-output")
    return db.get_clip(clip.id)


@pytest.fixture
def client(cfg, fake_tools):
    with TestClient(create_app(cfg, fake_tools.as_tools())) as c:
        yield c


# -- DB -------------------------------------------------------------------------------------
def test_new_clip_is_pending(db):
    clip = db.create_clip("https://clips.twitch.tv/AbcClip0001", "AbcClip0001")
    assert clip.review_status == "pending" and clip.reviewed_at is None


def test_migration_adds_review_columns_to_old_db(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE clips (id INTEGER PRIMARY KEY AUTOINCREMENT, source_url TEXT NOT NULL, "
        "slug TEXT NOT NULL, channel TEXT, title TEXT, game TEXT, duration_s REAL, "
        "status TEXT NOT NULL DEFAULT 'queued', stage TEXT, progress REAL NOT NULL DEFAULT 0, "
        "error TEXT, language TEXT NOT NULL DEFAULT 'auto', layout_type TEXT, "
        "layout_confidence REAL, layout_override TEXT, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)")
    conn.execute("INSERT INTO clips(source_url, slug, status, created_at, updated_at) "
                 "VALUES ('u', 'OldDoneClip1', 'done', 't', 't')")
    conn.commit()
    conn.close()
    d = DB(path)
    try:
        [clip] = d.list_clips()
        assert clip.review_status == "pending" and clip.reviewed_at is None
        assert [c.slug for c in d.list_inbox()] == ["OldDoneClip1"]
    finally:
        d.close()
    DB(path).close()  # migrating twice is harmless


def test_inbox_only_lists_done_pending_newest_first(db, cfg):
    a = finished(db, cfg, "FirstClip001")
    b = finished(db, cfg, "SecondClip01")
    db.create_clip("https://clips.twitch.tv/QueuedClip1", "QueuedClip1")  # not done
    c = finished(db, cfg, "ThirdClip001")
    db.set_review(a.id, "approved")
    assert [x.id for x in db.list_inbox()] == [c.id, b.id]


# -- API ------------------------------------------------------------------------------------
def test_inbox_endpoint_shape_and_order(client, cfg):
    db = client.app.state.db
    a = finished(db, cfg, "FirstClip001")
    b = finished(db, cfg, "SecondClip01")
    db.create_clip("https://clips.twitch.tv/QueuedClip1", "QueuedClip1")
    rows = client.get("/api/inbox").json()
    assert [r["id"] for r in rows] == [b.id, a.id]
    r = rows[0]
    assert r["title"] == "SecondClip01" and r["channel"] == "gronkh"
    assert r["layout_type"] == "stacked" and r["duration_s"] == 30.0
    assert r["output_url"] == f"/api/clips/{b.id}/output" and "language" in r


def test_approve_leaves_inbox_and_keeps_clip(client, cfg):
    db = client.app.state.db
    clip = finished(db, cfg, "ApproveMe001")
    r = client.post(f"/api/clips/{clip.id}/approve")
    assert r.status_code == 200
    assert r.json()["review_status"] == "approved" and r.json()["reviewed_at"]
    assert client.get("/api/inbox").json() == []
    [listed] = client.get("/api/clips").json()
    assert listed["id"] == clip.id and listed["review_status"] == "approved"
    assert listed["updated_at"] == clip.updated_at  # video cache key must not change
    assert client.get(f"/api/clips/{clip.id}/export").json()["video_url"]


def test_reject_reset_roundtrip(client, cfg):
    db = client.app.state.db
    clip = finished(db, cfg, "RejectMe0001")
    assert client.post(f"/api/clips/{clip.id}/reject").json()["review_status"] == "rejected"
    assert client.get("/api/inbox").json() == []
    r = client.post(f"/api/clips/{clip.id}/review/reset")
    assert r.json()["review_status"] == "pending" and r.json()["reviewed_at"] is None
    assert [x["id"] for x in client.get("/api/inbox").json()] == [clip.id]


def test_review_errors(client, cfg):
    db = client.app.state.db
    for action in ("approve", "reject", "review/reset"):
        assert client.post(f"/api/clips/99/{action}").status_code == 404
    queued = db.create_clip("https://clips.twitch.tv/QueuedClip1", "QueuedClip1")
    assert client.post(f"/api/clips/{queued.id}/approve").status_code == 409
    assert client.post(f"/api/clips/{queued.id}/reject").status_code == 409


def test_review_emits_sse_clip_event(client, cfg):
    bus = client.app.state.bus
    clip = finished(client.app.state.db, cfg, "EventClip001")
    q = bus.subscribe()
    client.post(f"/api/clips/{clip.id}/approve")
    ev = q.get_nowait()
    assert ev["type"] == "clip"
    assert ev["data"]["id"] == clip.id and ev["data"]["review_status"] == "approved"


def test_review_endpoints_need_token(cfg, fake_tools):
    cfg.auth_token = "secret"
    with TestClient(create_app(cfg, fake_tools.as_tools())) as c:
        clip = finished(c.app.state.db, cfg, "AuthClip0001")
        for method, path in [("get", "/api/inbox"),
                             ("post", f"/api/clips/{clip.id}/approve"),
                             ("post", f"/api/clips/{clip.id}/reject"),
                             ("post", f"/api/clips/{clip.id}/review/reset")]:
            assert getattr(c, method)(path).status_code == 401
        h = {"Authorization": "Bearer secret"}
        assert c.post(f"/api/clips/{clip.id}/approve", headers=h).status_code == 200
        assert c.get("/api/inbox", headers=h).json() == []


# -- sync ------------------------------------------------------------------------------------
def test_rejected_clip_is_never_requeued_by_sync(client, cfg, bus, fake_tools):
    """Even if the row is gone later, the slug stays recorded as seen."""
    adb = client.app.state.db
    ch = adb.add_channel("gronkh")
    fake_tools.channel_clips["gronkh"] = []
    tools = fake_tools.as_tools()
    sync_channel(cfg, adb, bus, tools, ch.id)  # baseline
    fake_tools.channel_clips["gronkh"] = [item("RejectedClip1")]
    assert sync_channel(cfg, adb, bus, tools, ch.id).queued == 1
    [clip] = adb.list_clips()
    adb.update_clip(clip.id, status="done")
    assert client.post(f"/api/clips/{clip.id}/reject").status_code == 200
    assert adb.seen_slugs(["RejectedClip1"]) == {"RejectedClip1"}
    assert sync_channel(cfg, adb, bus, tools, ch.id).queued == 0
    # deleting the rejected clip must not bring it back either
    assert client.delete(f"/api/clips/{clip.id}").status_code == 204
    assert sync_channel(cfg, adb, bus, tools, ch.id).queued == 0
    assert adb.list_clips() == []


def test_reject_records_seen_even_for_hand_added_clip(client, cfg):
    db = client.app.state.db
    clip = finished(db, cfg, "ByHandClip01", channel=None)
    client.post(f"/api/clips/{clip.id}/reject")
    assert db.seen_slugs(["ByHandClip01"]) == {"ByHandClip01"}
