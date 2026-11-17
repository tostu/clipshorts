"""Channel auto-sync: baseline, new-clip queueing, filters, API."""

import pytest
from fastapi.testclient import TestClient

from clipshorts.main import create_app
from clipshorts.sync import sync_channel
from clipshorts.twitch import ClipUrlError, parse_channel


def item(slug: str, views: int = 10) -> dict:
    return {"slug": slug, "url": f"https://clips.twitch.tv/{slug}", "title": slug,
            "views": views, "duration": 30.0, "timestamp": 1}


def run(cfg, db, bus, tools, ch):
    return sync_channel(cfg, db, bus, tools.as_tools(), ch.id)


# -- channel name parsing -------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    ("Gronkh", "gronkh"), ("@Gronkh", "gronkh"), ("twitch.tv/Gronkh", "gronkh"),
    ("https://www.twitch.tv/Gronkh/clips?filter=clips", "gronkh"),
    ("https://m.twitch.tv/gronkh", "gronkh"),
])
def test_parse_channel(raw, want):
    assert parse_channel(raw) == want


@pytest.mark.parametrize("raw", ["", "  ", "a b", "https://example.com/gronkh", "x" * 30,
                                 "https://twitch.tv/"])
def test_parse_channel_rejects(raw):
    with pytest.raises(ClipUrlError):
        parse_channel(raw)


# -- sync logic -------------------------------------------------------------------------
def test_first_sync_only_baselines(cfg, db, bus, fake_tools):
    ch = db.add_channel("Gronkh")
    fake_tools.channel_clips["gronkh"] = [item("OldClipSlug1"), item("OldClipSlug2")]
    res = run(cfg, db, bus, fake_tools, ch)
    assert (res.queued, res.baselined, res.error) == (0, 2, None)
    assert db.list_clips() == [] and db.list_jobs() == []
    assert db.channel_baselined(ch.id)
    assert db.get_channel(ch.id).last_sync_at


def test_later_sync_queues_only_new_clips(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh", language="de")
    fake_tools.channel_clips["gronkh"] = [item("OldClipSlug1")]
    run(cfg, db, bus, fake_tools, ch)
    fake_tools.channel_clips["gronkh"] = [item("OldClipSlug1"), item("NewClipSlug1")]
    res = run(cfg, db, bus, fake_tools, ch)
    assert res.queued == 1
    [clip] = db.list_clips()
    assert clip.slug == "NewClipSlug1" and clip.channel == "gronkh" and clip.language == "de"
    assert clip.source_url == "https://clips.twitch.tv/NewClipSlug1"
    assert [j.ref_id for j in db.list_jobs("queued")] == [clip.id]
    # same listing again: nothing new
    assert run(cfg, db, bus, fake_tools, ch).queued == 0
    got = db.get_channel(ch.id)
    assert got.total_queued == 1 and got.last_queued == 0 and got.last_error is None


def test_sync_stores_twitch_clip_creation_time(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh")
    fake_tools.channel_clips["gronkh"] = []
    run(cfg, db, bus, fake_tools, ch)
    fake_tools.channel_clips["gronkh"] = [{**item("NewClipSlug1"), "timestamp": 1700000000}]
    run(cfg, db, bus, fake_tools, ch)
    [clip] = db.list_clips()
    assert clip.clip_created_at.startswith("2023-11-14T22:13:20")
    assert clip.clip_created_at != clip.created_at


def test_deleted_clip_is_not_requeued(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh")
    fake_tools.channel_clips["gronkh"] = []
    run(cfg, db, bus, fake_tools, ch)
    fake_tools.channel_clips["gronkh"] = [item("NewClipSlug1")]
    run(cfg, db, bus, fake_tools, ch)
    db.delete_clip(db.list_clips()[0].id)
    assert run(cfg, db, bus, fake_tools, ch).queued == 0
    assert db.list_clips() == []


def test_min_views_defers_until_reached(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh", min_views=50)
    fake_tools.channel_clips["gronkh"] = []
    run(cfg, db, bus, fake_tools, ch)
    fake_tools.channel_clips["gronkh"] = [item("GrowingClip1", views=5)]
    assert run(cfg, db, bus, fake_tools, ch).queued == 0
    fake_tools.channel_clips["gronkh"] = [item("GrowingClip1", views=80)]
    assert run(cfg, db, bus, fake_tools, ch).queued == 1


def test_max_per_run_takes_best_views_first(cfg, db, bus, fake_tools):
    cfg.sync_max_per_channel = 2
    ch = db.add_channel("gronkh")
    fake_tools.channel_clips["gronkh"] = []
    run(cfg, db, bus, fake_tools, ch)
    fake_tools.channel_clips["gronkh"] = [item("ClipLow0001", 1), item("ClipTop0001", 99),
                                          item("ClipMid0001", 50)]
    assert run(cfg, db, bus, fake_tools, ch).queued == 2
    assert {c.slug for c in db.list_clips()} == {"ClipTop0001", "ClipMid0001"}
    assert run(cfg, db, bus, fake_tools, ch).queued == 1  # the rest follows next run
    assert len(db.list_clips()) == 3


def test_manually_added_clip_is_skipped(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh")
    fake_tools.channel_clips["gronkh"] = []
    run(cfg, db, bus, fake_tools, ch)
    db.create_clip("https://clips.twitch.tv/ByHandClip01", "ByHandClip01")
    fake_tools.channel_clips["gronkh"] = [item("ByHandClip01")]
    assert run(cfg, db, bus, fake_tools, ch).queued == 0
    assert len(db.list_clips()) == 1


def test_listing_failure_is_recorded_and_keeps_baseline_pending(cfg, db, bus, fake_tools):
    ch = db.add_channel("gronkh")
    fake_tools.fail_list = "yt-dlp: channel not found"
    res = run(cfg, db, bus, fake_tools, ch)
    assert res.error and "not found" in res.error
    got = db.get_channel(ch.id)
    assert got.last_error and not db.channel_baselined(ch.id)
    fake_tools.fail_list = None
    fake_tools.channel_clips["gronkh"] = [item("StillOldClip1")]
    assert run(cfg, db, bus, fake_tools, ch).baselined == 1  # baselines, does not queue
    assert db.get_channel(ch.id).last_error is None


# -- API ---------------------------------------------------------------------------------
@pytest.fixture
def client(cfg, fake_tools):
    with TestClient(create_app(cfg, fake_tools.as_tools())) as c:
        yield c


def test_channel_crud(client):
    assert client.get("/api/channels").json() == []
    r = client.post("/api/channels", json={"name": "https://twitch.tv/Gronkh", "min_views": 5})
    assert r.status_code == 201
    ch = r.json()
    assert ch["name"] == "gronkh" and ch["min_views"] == 5 and ch["enabled"] is True
    assert client.post("/api/channels", json={"name": "gronkh"}).status_code == 409
    assert client.post("/api/channels", json={"name": "not a channel"}).status_code == 422
    r = client.patch(f"/api/channels/{ch['id']}", json={"enabled": False, "language": "de"})
    assert r.json()["enabled"] is False and r.json()["language"] == "de"
    assert client.patch("/api/channels/99", json={"enabled": True}).status_code == 404
    assert client.delete(f"/api/channels/{ch['id']}").status_code == 204
    assert client.delete(f"/api/channels/{ch['id']}").status_code == 404


def test_sync_now_endpoint_baselines_then_queues(client, fake_tools):
    cid = client.post("/api/channels", json={"name": "gronkh"}).json()["id"]
    fake_tools.channel_clips["gronkh"] = [item("OldClipSlug1")]
    assert client.post(f"/api/channels/{cid}/sync").json()["total_queued"] == 0
    fake_tools.channel_clips["gronkh"] = [item("OldClipSlug1"), item("NewClipSlug1")]
    ch = client.post(f"/api/channels/{cid}/sync").json()
    assert ch["total_queued"] == 1 and ch["last_queued"] == 1 and ch["last_error"] is None
    assert [c["slug"] for c in client.get("/api/clips").json()] == ["NewClipSlug1"]
    assert client.post("/api/channels/99/sync").status_code == 404
