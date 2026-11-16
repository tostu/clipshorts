import pytest
from fastapi.testclient import TestClient

from clipshorts import metadata as md
from clipshorts.main import create_app
from clipshorts.models import Clip, Metadata

URL = "https://clips.twitch.tv/FamousSecretiveDolphinPoooound-aBc123"


def _clip(**kw) -> Clip:
    base = {"id": 1, "source_url": "u", "slug": "s", "status": "done",
            "created_at": "x", "updated_at": "x"}
    return Clip(**{**base, **kw})


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def _done_clip(client, cfg, url=URL):
    clip = client.post("/api/clips", json={"url": url}).json()
    client.app.state.db.update_clip(
        clip["id"], status="done", title="Big Play", game="Tetris", channel="somestreamer",
        layout_type="stacked")
    d = cfg.clip_dir(clip["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "output.mp4").write_bytes(b"x")
    return clip["id"]


# -- pure logic --------------------------------------------------------------
def test_clean_tag():
    assert md.clean_tag("#Game Over!") == "GameOver"
    assert md.clean_tag("  Äpfel ") == "Äpfel"
    assert md.clean_tag("###") == ""
    assert len(md.clean_tag("a" * 100)) == md.TAG_MAX_LEN


def test_merge_hashtags_dedup_order_and_limit():
    assert md.merge_hashtags(["Twitch", "#Clips", "twitch", "", "Win"], 3) == [
        "Twitch", "Clips", "Win"]
    assert md.merge_hashtags(["a", "b", "c"], 2) == ["a", "b"]


def test_default_metadata_title_creator_game_then_fixed(cfg):
    clip = _clip(title="Big Play", game="Counter-Strike 2", channel="SomeStreamer")
    m = md.default_metadata(cfg, clip)
    assert m.source == "auto" and m.title == "Big Play" and m.caption == ""
    assert m.hashtags == ["SomeStreamer", "SomeStreamerclips", "CounterStrike2", "twitchclips",
                          "twitch", "gaming", "fyp", "deutsch", "twitchde"]


def test_creator_comes_from_twitch_metadata_not_the_clipper(cfg):
    """yt-dlp: creator = streamer, channel/uploader = who made the clip (German4Media)."""
    clip = _clip(id=7, channel="German4Media", game="Just Chatting")
    d = cfg.clip_dir(7)
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        '{"twitch": {"channel": "German4Media", "creator": "Papaplatte"}}')
    tags = md.default_metadata(cfg, clip).hashtags
    assert tags[:3] == ["Papaplatte", "Papaplatteclips", "JustChatting"]
    assert "German4Media" not in tags


def test_default_metadata_without_creator_or_game(cfg):
    m = md.default_metadata(cfg, _clip())
    assert m.hashtags == ["twitchclips", "twitch", "gaming", "fyp", "deutsch", "twitchde"]
    assert m.title == ""


def test_default_metadata_dedups_creator_equal_to_fixed(cfg):
    assert md.default_metadata(cfg, _clip(channel="Twitch")).hashtags == [
        "Twitch", "Twitchclips", "gaming", "fyp", "deutsch", "twitchde"]


def test_default_metadata_caps_and_trims(cfg):
    small = cfg.model_copy(update={"max_hashtags": 3})
    m = md.default_metadata(small, _clip(title="x" * 200, channel="c", game="g"))
    assert m.hashtags == ["c", "cclips", "g"]
    assert len(m.title) == md.TITLE_MAX and m.title.endswith("…")


def test_custom_fixed_hashtags(cfg):
    custom = cfg.model_copy(update={"fixed_hashtags": " #a, b ,,"})
    assert md.default_metadata(custom, _clip()).hashtags == ["a", "b"]


def test_export_text():
    out = md.export_text(_clip(title="T"),
                         Metadata(title="Hook", caption="Cap", hashtags=["a", "b"]))
    assert out["tiktok"]["caption"] == "Hook\n\nCap\n\n#a #b"
    assert out["youtube"]["title"] == "Hook"
    assert out["youtube"]["description"] == "Cap\n\n#a #b\n\n#Shorts"


def test_export_text_falls_back_to_clip_title():
    assert md.export_text(_clip(title="Clip T"), Metadata())["youtube"]["title"] == "Clip T"


# -- API -----------------------------------------------------------------------
def test_defaults(client, cfg):
    cid = _done_clip(client, cfg)
    body = client.get(f"/api/clips/{cid}/metadata").json()
    assert body["source"] == "auto" and body["title"] == "Big Play"
    assert body["hashtags"] == ["somestreamer", "somestreamerclips", "Tetris", "twitchclips",
                                "twitch", "gaming", "fyp", "deutsch", "twitchde"]
    ex = client.get(f"/api/clips/{cid}/export").json()
    assert ex["download_url"] == f"/api/clips/{cid}/output?download=1"
    assert ex["platforms"]["tiktok"]["caption"] == (
        "Big Play\n\n#somestreamer #somestreamerclips #Tetris #twitchclips #twitch #gaming "
        "#fyp #deutsch #twitchde")
    assert ex["platforms"]["youtube"]["title"] == "Big Play"
    assert ex["platforms"]["youtube"]["description"].endswith("#Shorts")


def test_no_llm_endpoint(client, cfg):
    cid = _done_clip(client, cfg)
    assert client.post(f"/api/clips/{cid}/metadata/generate").status_code in (404, 405)


def test_metadata_404(client):
    assert client.get("/api/clips/99/metadata").status_code == 404
    assert client.get("/api/clips/99/export").status_code == 404
    assert client.put("/api/clips/99/metadata", json={}).status_code == 404


def test_manual_edit(client, cfg):
    cid = _done_clip(client, cfg)
    r = client.put(f"/api/clips/{cid}/metadata",
                   json={"title": "Mein Titel", "hashtags": ["#a b", "c", "C"]})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "manual" and body["title"] == "Mein Titel"
    assert body["hashtags"] == ["ab", "c"] and body["caption"] == ""
    # partial update keeps the rest
    r = client.put(f"/api/clips/{cid}/metadata", json={"caption": "Neu"})
    assert r.json()["title"] == "Mein Titel" and r.json()["caption"] == "Neu"
    assert r.json()["hashtags"] == ["ab", "c"]


def test_edit_starts_from_defaults(client, cfg):
    cid = _done_clip(client, cfg)
    r = client.put(f"/api/clips/{cid}/metadata", json={"caption": "Hi"})
    assert r.json()["title"] == "Big Play" and r.json()["hashtags"][0] == "somestreamer"


def test_publications_and_feedback(client, cfg):
    cid = _done_clip(client, cfg)
    r = client.post(f"/api/clips/{cid}/publications",
                    json={"platform": "tiktok", "url": "https://tiktok.com/x"})
    assert r.status_code == 201
    pub = r.json()
    assert pub["platform"] == "tiktok" and pub["rating"] is None and pub["published_at"]
    r = client.patch(f"/api/publications/{pub['id']}",
                     json={"views_24h": 1200, "rating": "good", "note": "hook worked"})
    assert r.json()["views_24h"] == 1200 and r.json()["rating"] == "good"
    # clearing a field with null leaves the others alone
    r = client.patch(f"/api/publications/{pub['id']}", json={"note": None})
    assert r.json()["note"] is None and r.json()["rating"] == "good"
    assert len(client.get(f"/api/clips/{cid}/publications").json()) == 1
    assert client.delete(f"/api/publications/{pub['id']}").status_code == 204
    assert client.delete(f"/api/publications/{pub['id']}").status_code == 404
    assert client.patch(f"/api/publications/{pub['id']}", json={}).status_code == 404


def test_publication_validation(client, cfg):
    cid = _done_clip(client, cfg)
    assert client.post(f"/api/clips/{cid}/publications",
                       json={"platform": "myspace"}).status_code == 422
    assert client.post(f"/api/clips/{cid}/publications",
                       json={"platform": "tiktok", "rating": "meh"}).status_code == 422
    queued = client.post("/api/clips", json={"url": URL.replace("aBc", "xYz")}).json()["id"]
    assert client.post(f"/api/clips/{queued}/publications",
                       json={"platform": "tiktok"}).status_code == 409
    assert client.post("/api/clips/99/publications",
                       json={"platform": "tiktok"}).status_code == 404


def test_rating_report(client, cfg):
    cid = _done_clip(client, cfg)
    for rating in ("good", "good", "flop", None):
        client.post(f"/api/clips/{cid}/publications",
                    json={"platform": "tiktok", "rating": rating})
    assert client.get("/api/report/ratings").json() == [
        {"channel": "somestreamer", "layout": "stacked", "good": 2, "ok": 0, "flop": 1}]


def test_delete_clip_cascades(client, cfg):
    cid = _done_clip(client, cfg)
    client.put(f"/api/clips/{cid}/metadata", json={"caption": "x"})
    client.post(f"/api/clips/{cid}/publications", json={"platform": "youtube"})
    assert client.delete(f"/api/clips/{cid}").status_code == 204
    db = client.app.state.db
    assert db.get_metadata(cid) is None and db.list_publications(cid) == []
