import pytest
from fastapi.testclient import TestClient

from clipshorts.config import Config
from clipshorts.main import create_app

URL = "https://clips.twitch.tv/FamousSecretiveDolphinPoooound-aBc123"


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_list_empty_and_404(client):
    assert client.get("/api/clips").json() == []
    assert client.get("/api/clips/1").status_code == 404


def test_create_clip_queues_job(client):
    r = client.post("/api/clips", json={"url": URL, "language": "de", "layout": "stacked"})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "queued" and body["language"] == "de"
    assert body["layout_override"] == "stacked"
    assert body["slug"] == "FamousSecretiveDolphinPoooound-aBc123"
    assert client.get(f"/api/clips/{body['id']}").json()["id"] == body["id"]
    assert len(client.get("/api/clips").json()) == 1
    db = client.app.state.db
    jobs = db.list_jobs("queued")
    assert len(jobs) == 1 and jobs[0].ref_id == body["id"]


def test_create_enqueues_fetch_only(client):
    client.post("/api/clips", json={"url": URL})
    assert [j.kind for j in client.app.state.db.list_jobs("queued")] == ["fetch"]


def test_process_requires_fetched_and_enqueues_pipeline(client):
    cid = client.post("/api/clips", json={"url": URL}).json()["id"]
    assert client.post(f"/api/clips/{cid}/process").status_code == 409  # still queued
    assert client.post("/api/clips/99/process").status_code == 404
    db = client.app.state.db
    db.update_clip(cid, status="fetched")
    r = client.post(f"/api/clips/{cid}/process", json={"layout": "face"})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert r.json()["layout_override"] == "face"
    assert [j.kind for j in db.list_jobs("queued")] == ["fetch", "pipeline"]


def test_preview_redirects_to_cdn_without_downloading(cfg, fake_tools):
    with TestClient(create_app(cfg, fake_tools.as_tools())) as c:
        cid = c.post("/api/clips", json={"url": URL}).json()["id"]
        r = c.get(f"/api/clips/{cid}/preview", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "https://cdn.example/clip.mp4?sig=abc"
        assert c.get("/api/clips/99/preview").status_code == 404
    assert fake_tools.downloads == []


def test_raw_serves_source_when_downloaded(client, cfg):
    cid = client.post("/api/clips", json={"url": URL}).json()["id"]
    assert client.get(f"/api/clips/{cid}/raw").status_code == 404
    assert client.get(f"/api/clips/{cid}").json()["raw_url"] is None
    d = cfg.clip_dir(cid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.mp4").write_bytes(b"raw" * 10)
    assert client.get(f"/api/clips/{cid}/raw").content == b"raw" * 10
    assert client.get(f"/api/clips/{cid}").json()["raw_url"] == f"/api/clips/{cid}/raw"


def test_create_layout_auto_means_no_override(client):
    r = client.post("/api/clips", json={"url": URL})
    assert r.json()["layout_override"] is None and r.json()["language"] == "auto"


def test_create_bad_url(client):
    r = client.post("/api/clips", json={"url": "https://example.com/x"})
    assert r.status_code == 422
    assert "invalid clip URL" in r.json()["detail"]
    assert client.get("/api/clips").json() == []


def test_create_bad_enum(client):
    assert client.post("/api/clips", json={"url": URL, "language": "xx"}).status_code == 422


def test_duplicate_returns_existing_and_failed_is_requeued(client):
    first = client.post("/api/clips", json={"url": URL}).json()
    again = client.post("/api/clips", json={"url": URL})
    assert again.status_code == 200 and again.json()["id"] == first["id"]
    assert len(client.app.state.db.list_jobs()) == 1

    db = client.app.state.db
    db.update_clip(first["id"], status="failed", error="download: boom")
    retry = client.post("/api/clips", json={"url": URL})
    assert retry.status_code == 200
    assert retry.json()["status"] == "queued" and retry.json()["error"] is None
    assert len(db.list_jobs("queued")) == 2


def test_status_filter(client):
    a = client.post("/api/clips", json={"url": URL}).json()
    client.app.state.db.update_clip(a["id"], status="failed", error="x")
    assert len(client.get("/api/clips?status=failed").json()) == 1
    assert client.get("/api/clips?status=done").json() == []


def _done_clip(client, cfg, payload: bytes = b"0123456789" * 100):
    clip = client.post("/api/clips", json={"url": URL}).json()
    db = client.app.state.db
    db.claim_next_job()
    db.update_clip(clip["id"], status="done", stage="render", progress=1.0)
    d = cfg.clip_dir(clip["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "output.mp4").write_bytes(payload)
    return clip["id"]


def test_output_404s(client, cfg):
    assert client.get("/api/clips/99/output").status_code == 404
    cid = client.post("/api/clips", json={"url": URL}).json()["id"]
    assert client.get(f"/api/clips/{cid}/output").status_code == 404  # not rendered yet
    assert client.get(f"/api/clips/{cid}").json()["output_url"] is None


def test_output_full_and_clip_exposes_url(client, cfg):
    cid = _done_clip(client, cfg)
    r = client.get(f"/api/clips/{cid}/output")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes" and len(r.content) == 1000
    assert r.headers["content-disposition"].startswith("inline")
    assert client.get(f"/api/clips/{cid}").json()["output_url"] == f"/api/clips/{cid}/output"
    assert client.get("/api/clips").json()[0]["output_url"] == f"/api/clips/{cid}/output"
    dl = client.get(f"/api/clips/{cid}/output?download=1")
    assert dl.headers["content-disposition"].startswith("attachment")
    assert "FamousSecretiveDolphinPoooound-aBc123.mp4" in dl.headers["content-disposition"]


def test_output_range_requests(client, cfg):
    payload = bytes(range(256)) * 4
    cid = _done_clip(client, cfg, payload)
    url = f"/api/clips/{cid}/output"
    r = client.get(url, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == payload[10:20]
    assert r.headers["content-range"] == f"bytes 10-19/{len(payload)}"
    r = client.get(url, headers={"Range": "bytes=1000-"})  # open-ended (browser seeking)
    assert r.status_code == 206 and r.content == payload[1000:]
    r = client.get(url, headers={"Range": "bytes=-24"})  # suffix
    assert r.status_code == 206 and r.content == payload[-24:]
    r = client.get(url, headers={"Range": "bytes=5000-6000"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{len(payload)}"


def test_rerender_requeues_with_new_layout(client, cfg):
    cid = _done_clip(client, cfg)
    db = client.app.state.db
    r = client.post(f"/api/clips/{cid}/rerender", json={"layout": "stacked"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued" and body["layout_override"] == "stacked"
    assert len(db.list_jobs("queued")) == 1 and db.list_jobs("queued")[0].ref_id == cid
    # busy clip: no second job
    assert client.post(f"/api/clips/{cid}/rerender", json={}).status_code == 409
    assert len(db.list_jobs("queued")) == 1


def test_rerender_auto_clears_override_and_empty_body_keeps_it(client, cfg):
    cid = _done_clip(client, cfg)
    db = client.app.state.db
    db.update_clip(cid, layout_override="face")
    r = client.post(f"/api/clips/{cid}/rerender")  # no body at all
    assert r.status_code == 200 and r.json()["layout_override"] == "face"
    db.update_clip(cid, status="done")
    r = client.post(f"/api/clips/{cid}/rerender", json={"layout": "auto"})
    assert r.json()["layout_override"] is None


def test_rerender_validation(client, cfg):
    assert client.post("/api/clips/99/rerender", json={}).status_code == 404
    cid = _done_clip(client, cfg)
    assert client.post(f"/api/clips/{cid}/rerender", json={"layout": "bogus"}).status_code == 422


def test_rerender_end_to_end_with_worker(tmp_path, fake_tools):
    """New clips stop after fetch; /process runs the rest; a re-render re-runs layout + render."""
    import time

    cfg2 = Config(data_dir=tmp_path / "d", worker_enabled=True, nice_level=0, _env_file=None)
    with TestClient(create_app(cfg2, fake_tools.as_tools())) as c:
        cid = c.post("/api/clips", json={"url": URL}).json()["id"]

        def wait_done(states=("done", "failed")):
            for _ in range(200):
                body = c.get(f"/api/clips/{cid}").json()
                if body["status"] in states:
                    return body
                time.sleep(0.05)
            raise AssertionError("worker did not finish")

        fetched = wait_done(("fetched", "failed"))
        assert fetched["status"] == "fetched" and fetched["output_url"] is None
        assert fetched["title"] == "Big Play"
        assert fetched["raw_url"] is None and fake_tools.downloads == []  # embed, no download
        assert fake_tools.transcriptions == [] and fake_tools.renders == []
        assert c.post(f"/api/clips/{cid}/process").status_code == 200
        assert c.post(f"/api/clips/{cid}/process").status_code == 409  # already queued
        first = wait_done()
        assert first["status"] == "done" and first["output_url"]
        assert c.get(first["output_url"]).content == b"fake-output"
        c.post(f"/api/clips/{cid}/rerender", json={"layout": "content"})
        second = wait_done()
        assert second["status"] == "done" and second["layout_type"] == "content"
        assert len(fake_tools.renders) == 2 and len(fake_tools.transcriptions) == 1


def test_auth_token(tmp_path):
    cfg = Config(data_dir=tmp_path / "d", auth_token="tok", worker_enabled=False,
                 nice_level=0, _env_file=None)
    with TestClient(create_app(cfg)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/api/clips").status_code == 401
        assert c.get("/api/clips", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/clips", headers={"Authorization": "Bearer tok"}).status_code == 200
        assert c.get("/api/clips?token=tok").status_code == 200


def _write_transcript(cfg, cid, words):
    import json

    d = cfg.clip_dir(cid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(json.dumps({"language": "de", "words": words}))


def test_captions_edit_roundtrip(client, cfg):
    cid = _done_clip(client, cfg)
    _write_transcript(cfg, cid, [{"word": "ENOCHE", "start": 0.0, "end": 1.0},
                                 {"word": "geil", "start": 1.0, "end": 2.0}])
    got = client.get(f"/api/clips/{cid}/captions").json()
    assert got["edited"] is False and [w["text"] for w in got["words"]] == ["ENOCHE", "geil"]

    words = [{"start": 0.0, "end": 1.0, "text": "Enoch"}, {"start": 1.0, "end": 2.0, "text": ""}]
    r = client.put(f"/api/clips/{cid}/captions", json={"words": words})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    got = client.get(f"/api/clips/{cid}/captions").json()
    assert got["edited"] is True and [w["text"] for w in got["words"]] == ["Enoch"]
    assert client.put(f"/api/clips/{cid}/captions", json={"words": words}).status_code == 409

    client.app.state.db.update_clip(cid, status="done")
    r = client.delete(f"/api/clips/{cid}/captions")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert client.get(f"/api/clips/{cid}/captions").json()["edited"] is False


def test_captions_edit_validation(client, cfg):
    assert client.get("/api/clips/99/captions").status_code == 404
    cid = _done_clip(client, cfg)
    ok = {"words": [{"start": 0, "end": 1, "text": "x"}]}
    assert client.put(f"/api/clips/{cid}/captions", json=ok).status_code == 409  # no transcript
    _write_transcript(cfg, cid, [])
    bad = {"words": [{"start": 2, "end": 1, "text": "x"}]}
    assert client.put(f"/api/clips/{cid}/captions", json=bad).status_code == 422


def test_captions_edit_end_to_end_reruns_captions_and_render_only(tmp_path, fake_tools):
    import time

    cfg2 = Config(data_dir=tmp_path / "d", worker_enabled=True, nice_level=0, _env_file=None)
    with TestClient(create_app(cfg2, fake_tools.as_tools())) as c:
        cid = c.post("/api/clips", json={"url": URL}).json()["id"]

        def wait_done():
            for _ in range(200):
                body = c.get(f"/api/clips/{cid}").json()
                if body["status"] == "fetched":
                    c.post(f"/api/clips/{cid}/process")
                elif body["status"] in ("done", "failed"):
                    return body
                time.sleep(0.05)
            raise AssertionError("worker did not finish")

        assert wait_done()["status"] == "done"
        words = c.get(f"/api/clips/{cid}/captions").json()["words"]
        words[0]["text"] = "Korrigiert"
        c.put(f"/api/clips/{cid}/captions", json={"words": words})
        assert wait_done()["status"] == "done"
        ass = (cfg2.clip_dir(cid) / "captions.ass").read_text(encoding="utf-8")
        assert "Korrigiert" in ass
        assert len(fake_tools.renders) == 2 and len(fake_tools.transcriptions) == 1
        c.delete(f"/api/clips/{cid}/captions")
        assert wait_done()["status"] == "done"
        assert "Korrigiert" not in (cfg2.clip_dir(cid) / "captions.ass").read_text("utf-8")
