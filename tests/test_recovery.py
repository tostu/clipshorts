import time

from fastapi.testclient import TestClient

from clipshorts.config import Config
from clipshorts.db import DB
from clipshorts.events import EventBus
from clipshorts.main import create_app
from clipshorts.pipeline import STAGES, run_pipeline
from tests.conftest import FakeTools

URL = "https://clips.twitch.tv/FamousSecretiveDolphinPoooound-aBc123"


def test_restart_resumes_without_redownload(tmp_path):
    """Killed mid-pipeline (clip + job 'running', download cached): on restart the worker
    resumes, never re-downloads, and finishes."""
    cfg = Config(data_dir=tmp_path / "d", worker_enabled=True, nice_level=0, _env_file=None)
    cfg.clips_dir.mkdir(parents=True)

    # first life: download stage completed, then the process died
    db = DB(cfg.db_path)
    clip = db.create_clip(URL, "FamousSecretiveDolphinPoooound-aBc123")
    db.enqueue_job("pipeline", clip.id)
    db.claim_next_job()
    first = FakeTools()
    run_pipeline(clip.id, cfg, db, EventBus(), first.as_tools(), stages=STAGES[:2])
    assert len(first.downloads) == 1
    assert (cfg.clip_dir(clip.id) / "stages.json").is_file()
    assert (cfg.clip_dir(clip.id) / "source.mp4").is_file()
    db.update_clip(clip.id, status="running", stage="probe")
    db.close()

    # second life
    second = FakeTools()
    with TestClient(create_app(cfg, second.as_tools())) as c:
        body = {}
        for _ in range(200):
            body = c.get(f"/api/clips/{clip.id}").json()
            if body["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert body["status"] == "done", body
        assert body["output_url"] and c.get(body["output_url"]).content == b"fake-output"
        assert [j.status for j in c.app.state.db.list_jobs()] == ["done"]
    assert second.downloads == []  # cached download reused
    assert len(second.probes) == 1 and len(second.renders) == 1


def test_running_clips_requeued_on_startup(cfg):
    db = DB(cfg.db_path)
    running = db.create_clip("u", "running11")
    done = db.create_clip("u", "donedone1")
    failed = db.create_clip("u", "failed111")
    db.update_clip(running.id, status="running", stage="transcribe")
    db.update_clip(done.id, status="done")
    db.update_clip(failed.id, status="failed", error="x")
    job = db.enqueue_job("pipeline", running.id)
    db.claim_next_job()  # job is now 'running'
    db.close()

    # simulate process restart: new connection on the same file
    db2 = DB(cfg.db_path)
    res = db2.recover()
    assert res == {"clips": 1, "jobs": 1}
    assert db2.get_clip(running.id).status == "queued"
    assert db2.get_clip(running.id).stage == "transcribe"  # stage kept for display
    assert db2.get_clip(done.id).status == "done"
    assert db2.get_clip(failed.id).status == "failed"
    assert db2.get_job(job.id).status == "queued"
    # and the requeued job is claimable again
    assert db2.claim_next_job().id == job.id
    db2.close()
