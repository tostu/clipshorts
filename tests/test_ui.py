"""M5: static UI, API shape, delete, live SSE over a real socket."""

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from clipshorts.config import Config
from clipshorts.main import create_app

URL = "https://clips.twitch.tv/FamousSecretiveDolphinPoooound-aBc123"


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as c:
        yield c


def test_index_and_assets_served(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>clipshorts</title>" in r.text
    assert 'id="new-form"' in r.text
    assert client.get("/static/app.js").status_code == 200
    assert "text/css" in client.get("/static/style.css").headers["content-type"]
    # self-contained: no external origins referenced
    assert "http://" not in r.text.replace("http://www.w3.org", "")
    # the one allowed origin is Twitch's own clip embed (preview before processing)
    js = client.get("/static/app.js").text.replace("https://clips.twitch.tv/embed", "")
    assert "//cdn" not in r.text and "https://" not in js


def test_filter_controls_shipped(client):
    html = client.get("/").text
    for el_id in ("filter-day", "filter-creator", "filter-clear"):
        assert f'id="{el_id}"' in html
    js = client.get("/static/app.js").text
    assert "function applyFilters" in js and "matchesFilters" in js


def test_ui_public_but_api_needs_token(tmp_path):
    cfg = Config(data_dir=tmp_path / "d", auth_token="tok", worker_enabled=False,
                 nice_level=0, _env_file=None)
    with TestClient(create_app(cfg)) as c:
        assert c.get("/").status_code == 200
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/api/clips").status_code == 401
        assert c.get("/api/events").status_code == 401


def test_clip_shape(client):
    body = client.post("/api/clips", json={"url": URL, "language": "en"}).json()
    got = client.get(f"/api/clips/{body['id']}").json()
    for key in ("id", "status", "stage", "progress", "layout_type", "layout_confidence",
                "language", "detected_language", "error", "output_url", "layout_override"):
        assert key in got
    assert got["status"] == "queued" and got["language"] == "en"
    assert got["output_url"] is None and got["detected_language"] is None


def test_detected_language_from_transcript(client, cfg):
    cid = client.post("/api/clips", json={"url": URL}).json()["id"]
    d = cfg.clip_dir(cid)
    d.mkdir(parents=True)
    (d / "transcript.json").write_text(json.dumps({"language": "de"}))
    assert client.get(f"/api/clips/{cid}").json()["detected_language"] == "de"


def test_delete_clip(client, cfg):
    assert client.delete("/api/clips/99").status_code == 404
    cid = client.post("/api/clips", json={"url": URL}).json()["id"]
    assert client.delete(f"/api/clips/{cid}").status_code == 409  # queued
    db = client.app.state.db
    db.claim_next_job()
    db.update_clip(cid, status="running")
    assert client.delete(f"/api/clips/{cid}").status_code == 409
    db.update_clip(cid, status="done")
    d = cfg.clip_dir(cid)
    d.mkdir(parents=True)
    (d / "output.mp4").write_bytes(b"x")
    assert client.delete(f"/api/clips/{cid}").status_code == 204
    assert not d.exists()
    assert client.get(f"/api/clips/{cid}").status_code == 404
    assert client.get("/api/clips").json() == []
    assert db.list_jobs() == []


class LiveServer:
    def __init__(self, app):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(10)


def test_sse_emits_state_changes(tmp_path, fake_tools):
    cfg = Config(data_dir=tmp_path / "d", worker_enabled=True, nice_level=0, _env_file=None)
    seen: list[dict] = []
    with LiveServer(create_app(cfg, fake_tools.as_tools())) as srv:
        with httpx.Client(base_url=srv.base, timeout=10) as c:
            with c.stream("GET", "/api/events") as stream:
                assert stream.headers["content-type"].startswith("text/event-stream")
                lines = stream.iter_lines()
                assert next(lines) == ": connected"
                created = httpx.post(srv.base + "/api/clips", json={"url": URL}).json()
                event = None
                for line in lines:
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: ") and event == "clip":
                        data = json.loads(line[6:])
                        seen.append(data)
                        if data.get("status") == "fetched":  # processing is manual
                            httpx.post(f"{srv.base}/api/clips/{created['id']}/process")
                        if data.get("status") == "done":
                            break
            final = c.get(f"/api/clips/{created['id']}").json()
    assert {d["id"] for d in seen} == {created["id"]}
    assert "running" in {d["status"] for d in seen}
    assert {"download", "render"} <= {d["stage"] for d in seen}
    assert any(0 < d["progress"] < 1 for d in seen)
    assert final["status"] == "done" and final["output_url"]
    assert final["detected_language"] == "de"
