import json

import pytest
from fastapi.testclient import TestClient

from clipshorts import metadata as md
from clipshorts.config import Config
from clipshorts.main import create_app
from clipshorts.models import Clip, Metadata

URL = "https://clips.twitch.tv/FamousSecretiveDolphinPoooound-aBc123"


def _clip(**kw) -> Clip:
    base = {"id": 1, "source_url": "u", "slug": "s", "status": "done",
            "created_at": "x", "updated_at": "x"}
    return Clip(**{**base, **kw})


class FakeLLM:
    def __init__(self):
        self.out = {
            "title": "Unfassbarer Clutch in letzter Sekunde",
            "caption": "Das war knapp. Wer hätte das gedacht?",
            "hashtags": ["#Tetris", "clutch", "Game Over!", "tetris"],
        }
        self.error: str | None = None
        self.calls: list[tuple[str, str]] = []

    def __call__(self, cfg, system, user, schema):
        self.calls.append((system, user))
        if self.error:
            raise md.MetadataError(self.error)
        return self.out


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(cfg, llm):
    llm_cfg = cfg.model_copy(update={"llm_enabled": True, "fixed_hashtags": ""})
    with TestClient(create_app(llm_cfg, llm=llm)) as c:
        yield c


@pytest.fixture
def plain_client(cfg):
    """Defaults: LLM off, built-in fixed hashtags."""
    with TestClient(create_app(cfg)) as c:
        yield c


def _done_clip(client, cfg, transcript=True, url=URL):
    clip = client.post("/api/clips", json={"url": url}).json()
    db = client.app.state.db
    db.update_clip(clip["id"], status="done", title="Big Play", game="Tetris",
                   channel="somestreamer", layout_type="stacked")
    d = cfg.clip_dir(clip["id"])
    d.mkdir(parents=True, exist_ok=True)
    if transcript:
        (d / "transcript.json").write_text(json.dumps({
            "language": "de",
            "words": [{"word": "Das"}, {"word": "ist"}, {"word": "krass."}],
        }))
    (d / "output.mp4").write_bytes(b"x")
    return clip["id"]


# -- pure logic --------------------------------------------------------------
def test_clean_tag():
    assert md.clean_tag("#Game Over!") == "GameOver"
    assert md.clean_tag("  Äpfel ") == "Äpfel"
    assert md.clean_tag("###") == ""
    assert len(md.clean_tag("a" * 100)) == md.TAG_MAX_LEN


def test_merge_hashtags_fixed_first_dedup_and_limit():
    out = md.merge_hashtags(["Twitch", "#Clips"], ["twitch", "Tetris", "", "Win"], 4)
    assert out == ["Twitch", "Clips", "Tetris", "Win"]
    assert md.merge_hashtags([], ["a", "b", "c"], 2) == ["a", "b"]


def test_normalize_trims_and_merges_fixed(tmp_path):
    cfg = Config(data_dir=tmp_path, fixed_hashtags="twitch,#clips", max_hashtags=3,
                 _env_file=None)
    m = md.normalize({"title": "x" * 200, "caption": "hi\n  there", "hashtags": ["a", "b"]}, cfg)
    assert len(m.title) == md.TITLE_MAX and m.title.endswith("…")
    assert m.caption == "hi there"
    assert m.hashtags == ["twitch", "clips", "a"] and m.source == "llm"


def test_normalize_rejects_empty(cfg):
    with pytest.raises(md.MetadataError):
        md.normalize({"title": " ", "caption": "", "hashtags": []}, cfg)


def test_normalize_tolerates_bad_hashtags_type(cfg):
    bare = cfg.model_copy(update={"fixed_hashtags": ""})
    assert md.normalize({"title": "t", "hashtags": "nope"}, bare).hashtags == []


def test_default_metadata_creator_game_then_fixed(cfg):
    clip = _clip(title="Big Play", game="Counter-Strike 2", channel="SomeStreamer")
    m = md.default_metadata(cfg, clip)
    assert m.source == "auto" and m.title == "Big Play" and m.caption == ""
    assert m.hashtags == ["SomeStreamer", "CounterStrike2", "twitchclips", "twitch", "gaming",
                          "fyp"]


def test_default_metadata_without_creator_or_game(cfg):
    m = md.default_metadata(cfg, _clip())
    assert m.hashtags == ["twitchclips", "twitch", "gaming", "fyp"] and m.title == ""


def test_default_metadata_dedups_creator_equal_to_fixed(cfg):
    assert md.default_metadata(cfg, _clip(channel="Twitch")).hashtags[:2] == ["Twitch",
                                                                              "twitchclips"]


def test_transcript_text(tmp_path):
    assert md.transcript_text(tmp_path) == ("", None)
    (tmp_path / "transcript.json").write_text(
        json.dumps({"language": "en", "words": [{"word": "a"}, {"word": " b "}, "junk"]}))
    assert md.transcript_text(tmp_path) == ("a b", "en")


def test_prompts_use_language_and_clip_facts():
    clip = _clip(title="T", game="G", channel="C", duration_s=30.0)
    system, user = md.build_prompts(clip, "hallo welt", "de")
    assert "German" in system
    assert "Game: G" in user and "Transcript: hallo welt" in user and "30 s" in user


def test_export_text():
    out = md.export_text(_clip(title="T"),
                         Metadata(title="Hook", caption="Cap", hashtags=["a", "b"]))
    assert out["tiktok"]["caption"] == "Hook\n\nCap\n\n#a #b"
    assert out["youtube"]["title"] == "Hook"
    assert out["youtube"]["description"] == "Cap\n\n#a #b\n\n#Shorts"


def test_export_text_falls_back_to_clip_title():
    assert md.export_text(_clip(title="Clip T"), Metadata())["youtube"]["title"] == "Clip T"


def test_chat_json_unreachable(cfg):
    bad = Config(data_dir=cfg.data_dir, llm_url="http://127.0.0.1:1", llm_timeout_s=2,
                 _env_file=None)
    with pytest.raises(md.MetadataError, match="not reachable"):
        md.chat_json(bad, "s", "u", md.SCHEMA)


# -- API -----------------------------------------------------------------------
def test_generate_stores_and_returns(client, cfg, llm):
    cid = _done_clip(client, cfg)
    r = client.post(f"/api/clips/{cid}/metadata/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "llm" and body["hashtags"] == ["Tetris", "clutch", "GameOver"]
    assert "krass." in llm.calls[0][1] and "Game: Tetris" in llm.calls[0][1]
    assert client.get(f"/api/clips/{cid}/metadata").json()["title"] == body["title"]


def test_generate_errors(client, cfg, llm):
    assert client.post("/api/clips/99/metadata/generate").status_code == 404
    queued = client.post("/api/clips", json={"url": URL}).json()["id"]
    assert client.post(f"/api/clips/{queued}/metadata/generate").status_code == 409
    client.app.state.db.update_clip(queued, status="done")
    llm.error = "LLM not reachable"
    r = client.post(f"/api/clips/{queued}/metadata/generate")
    assert r.status_code == 502 and "not reachable" in r.json()["detail"]
    assert client.get(f"/api/clips/{queued}/metadata").json()["source"] == "auto"


def test_generate_disabled_by_default(plain_client, cfg):
    cid = _done_clip(plain_client, cfg)
    r = plain_client.post(f"/api/clips/{cid}/metadata/generate")
    assert r.status_code == 409 and "disabled" in r.json()["detail"]
    assert plain_client.get(f"/api/clips/{cid}/metadata").json()["llm_enabled"] is False


def test_defaults_without_llm(plain_client, cfg):
    cid = _done_clip(plain_client, cfg)
    body = plain_client.get(f"/api/clips/{cid}/metadata").json()
    assert body["source"] == "auto" and body["title"] == "Big Play"
    assert body["hashtags"] == ["somestreamer", "Tetris", "twitchclips", "twitch", "gaming",
                                "fyp"]
    ex = plain_client.get(f"/api/clips/{cid}/export").json()
    assert "#somestreamer #Tetris" in ex["platforms"]["tiktok"]["caption"]
    # a manual edit starts from the defaults and then sticks
    r = plain_client.put(f"/api/clips/{cid}/metadata", json={"caption": "Hi"})
    assert r.json()["source"] == "manual" and r.json()["title"] == "Big Play"
    assert r.json()["hashtags"][0] == "somestreamer"


def test_generate_without_transcript_still_works(client, cfg, llm):
    cid = _done_clip(client, cfg, transcript=False)
    assert client.post(f"/api/clips/{cid}/metadata/generate").status_code == 200
    assert "(no speech)" in llm.calls[0][1]


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


def test_export(client, cfg):
    cid = _done_clip(client, cfg)
    client.post(f"/api/clips/{cid}/metadata/generate")
    body = client.get(f"/api/clips/{cid}/export").json()
    assert body["download_url"] == f"/api/clips/{cid}/output?download=1"
    assert "#Tetris" in body["platforms"]["tiktok"]["caption"]
    assert body["platforms"]["youtube"]["description"].endswith("#Shorts")


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
    client.post(f"/api/clips/{cid}/metadata/generate")
    client.post(f"/api/clips/{cid}/publications", json={"platform": "youtube"})
    assert client.delete(f"/api/clips/{cid}").status_code == 204
    db = client.app.state.db
    assert db.get_metadata(cid) is None and db.list_publications(cid) == []
