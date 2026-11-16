from clipshorts.db import DB


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
