import asyncio

from clipshorts.worker import Worker


async def test_process_one_runs_job_and_parks_at_unimplemented_stage(cfg, db, bus, fake_tools):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    job = db.enqueue_job("pipeline", c.id)
    w = Worker(cfg, db, bus, fake_tools.as_tools())
    bus.bind_loop(asyncio.get_running_loop())
    try:
        assert await w.process_one() is True
        assert await w.process_one() is False  # queue empty
    finally:
        await w.shutdown()
    assert db.get_job(job.id).status == "done"
    row = db.get_clip(c.id)
    assert row.status == "waiting" and row.stage == "transcribe"
    assert row.title == "Big Play"


async def test_failed_pipeline_marks_job_failed(cfg, db, bus):
    from tests.conftest import FakeTools

    ft = FakeTools(fail_download="yt-dlp: gone")
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    job = db.enqueue_job("pipeline", c.id)
    w = Worker(cfg, db, bus, ft.as_tools())
    bus.bind_loop(asyncio.get_running_loop())
    try:
        await w.process_one()
    finally:
        await w.shutdown()
    assert db.get_job(job.id).status == "failed"
    assert db.get_clip(c.id).error == "download: yt-dlp: gone"


async def test_unknown_job_kind(cfg, db, bus, fake_tools):
    job = db.enqueue_job("mystery", 1)
    w = Worker(cfg, db, bus, fake_tools.as_tools())
    try:
        await w.process_one()
    finally:
        await w.shutdown()
    assert db.get_job(job.id).status == "failed"


async def test_start_recovers_and_loop_drains(cfg, db, bus, fake_tools):
    c = db.create_clip("https://clips.twitch.tv/abc12345", "abc12345")
    db.enqueue_job("pipeline", c.id)
    db.claim_next_job()  # simulate crash mid-job
    db.update_clip(c.id, status="running")
    w = Worker(cfg, db, bus, fake_tools.as_tools())
    await w.start()
    try:
        for _ in range(100):
            if db.get_clip(c.id).status == "waiting":
                break
            await asyncio.sleep(0.05)
    finally:
        await w.shutdown()
    assert db.get_clip(c.id).status == "waiting"
    assert len(fake_tools.downloads) == 1
