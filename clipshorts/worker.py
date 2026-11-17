"""Single worker loop: one heavy job at a time, at low priority."""

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .config import Config
from .db import DB
from .events import EventBus
from .media import Tools, apply_nice
from .pipeline import FETCH_UNTIL, PipelineResult, run_pipeline

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: Config, db: DB, bus: EventBus, tools: Tools):
        self.cfg, self.db, self.bus, self.tools = cfg, db, bus, tools
        # exactly one thread => exactly one heavy job; niced once at thread start
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pipeline",
            initializer=partial(apply_nice, cfg.nice_level),
        )
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._cancel = threading.Event()  # tells a running stage (ffmpeg) to stop on shutdown

    def poke(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        self.bus.bind_loop(asyncio.get_running_loop())
        rec = self.db.recover()
        if rec["clips"] or rec["jobs"]:
            log.info("recovery: requeued %s clip(s), %s job(s)", rec["clips"], rec["jobs"])
        self._task = asyncio.create_task(self._loop(), name="worker")
        self.poke()

    async def shutdown(self) -> None:
        self._stopping = True
        self._cancel.set()
        self._wake.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                did = await self.process_one()
            except Exception:  # noqa: BLE001 - never let the loop die
                log.exception("worker iteration failed")
                did = False
            if not did:
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), 30)

    async def process_one(self) -> bool:
        """Claim and run the next job. Returns False if the queue was empty."""
        job = self.db.claim_next_job()
        if job is None:
            return False
        if job.kind not in ("pipeline", "fetch"):
            self.db.finish_job(job.id, False, f"unknown job kind {job.kind!r}")
            return True
        # "fetch" = metadata only (preview is Twitch's embed, nothing downloaded);
        # "pipeline" = the full run incl. download, started manually
        until = FETCH_UNTIL if job.kind == "fetch" else None
        loop = asyncio.get_running_loop()
        try:
            res: PipelineResult = await loop.run_in_executor(
                self._pool,
                partial(run_pipeline, job.ref_id, self.cfg, self.db, self.bus, self.tools,
                        cancel=self._cancel, until=until),
            )
        except asyncio.CancelledError:
            raise  # leave job 'running': recovery re-queues it on next start
        except Exception as e:  # noqa: BLE001
            log.exception("job %s crashed", job.id)
            self.db.finish_job(job.id, False, str(e))
            clip = self.db.update_clip(job.ref_id, status="failed", error=f"internal: {e}")
            if clip:
                self.bus.publish("clip", clip.model_dump())
            return True
        if res.status == "cancelled":
            return True  # shutdown mid-render: job stays 'running', recovery re-queues it
        self.db.finish_job(job.id, res.status != "failed", res.error)
        return True
