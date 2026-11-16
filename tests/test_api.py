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


def test_auth_token(tmp_path):
    cfg = Config(data_dir=tmp_path / "d", auth_token="tok", worker_enabled=False,
                 nice_level=0, _env_file=None)
    with TestClient(create_app(cfg)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/api/clips").status_code == 401
        assert c.get("/api/clips", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/clips", headers={"Authorization": "Bearer tok"}).status_code == 200
        assert c.get("/api/clips?token=tok").status_code == 200
