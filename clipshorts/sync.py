"""Channel auto-sync: poll followed channels, queue new clips like pasted URLs.

First successful sync of a channel only *baselines* (marks what is already there as seen), so
following a channel does not queue its whole last day. After that every unseen clip is queued,
best views first, at most `sync_max_per_channel` per run. A clip below the channel's `min_views`
stays unseen and is re-checked next run (until it drops out of the 24 h listing).
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from .config import Config
from .db import DB
from .events import EventBus
from .media import MediaError, Tools
from .twitch import ClipUrlError, parse_clip_url
from .worker import Worker

log = logging.getLogger(__name__)

LIST_RANGE = "24hr"
LIST_LIMIT = 100


@dataclass
class SyncResult:
    channel: str
    queued: int = 0
    baselined: int = 0
    error: str | None = None


def sync_channel(cfg: Config, db: DB, bus: EventBus, tools: Tools, channel_id: int) -> SyncResult:
    """One blocking sync run (network call inside): safe to run in a thread."""
    ch = db.get_channel(channel_id)
    if ch is None:
        return SyncResult("", error="channel not found")
    res = SyncResult(ch.name)
    try:
        listing = tools.list_clips(ch.name, LIST_RANGE, LIST_LIMIT)
    except (MediaError, OSError) as e:
        res.error = str(e)[:500]
        db.record_sync(ch.id, res.error)
        return res

    if not db.channel_baselined(ch.id):
        for item in listing:
            db.mark_seen(item["slug"], ch.name)
        res.baselined = len(listing)
        db.record_sync(ch.id, None, 0)
        return res

    seen = db.seen_slugs([i["slug"] for i in listing])
    fresh = [i for i in listing if i["slug"] not in seen]
    fresh.sort(key=lambda i: i.get("views") or 0, reverse=True)
    for item in fresh:
        if res.queued >= cfg.sync_max_per_channel:
            break
        if (item.get("views") or 0) < ch.min_views:
            continue  # not popular enough yet; look again next run
        slug = item["slug"]
        db.mark_seen(slug, ch.name)
        if db.find_clip_by_slug(slug) is not None:  # pasted by hand earlier
            continue
        try:
            canonical = parse_clip_url(item["url"]).canonical_url
        except ClipUrlError:
            continue
        clip = db.create_clip(canonical, slug, channel=ch.name, language=ch.language)
        db.enqueue_job("pipeline", clip.id)
        bus.publish("clip", clip.model_dump())
        res.queued += 1
    db.record_sync(ch.id, None, res.queued)
    if res.queued:
        log.info("sync %s: queued %d clip(s)", ch.name, res.queued)
    return res


class Syncer:
    """Background loop (one run at a time) plus an on-demand `sync_one`."""

    def __init__(self, cfg: Config, db: DB, bus: EventBus, tools: Tools, worker: Worker):
        self.cfg, self.db, self.bus, self.tools, self.worker = cfg, db, bus, tools, worker
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="channel-sync")

    async def shutdown(self) -> None:
        for task in [t for t in (self._task, *self._bg) if t]:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def sync_one(self, channel_id: int) -> SyncResult:
        async with self._lock:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None, sync_channel, self.cfg, self.db, self.bus, self.tools, channel_id)
        if res.queued:
            self.worker.poke()
        return res

    def sync_in_background(self, channel_id: int) -> None:
        task = asyncio.create_task(self.sync_one(channel_id))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def sync_all(self) -> None:
        for ch in self.db.list_channels(enabled_only=True):
            try:
                await self.sync_one(ch.id)
            except Exception:  # noqa: BLE001 - one bad channel must not stop the others
                log.exception("sync of %s crashed", ch.name)

    async def _loop(self) -> None:
        await asyncio.sleep(10)  # let startup recovery settle first
        while True:
            try:
                await self.sync_all()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("channel sync iteration failed")
            await asyncio.sleep(max(60.0, self.cfg.sync_interval_minutes * 60))
