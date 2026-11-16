"""REST + SSE routes."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .db import DB
from .events import EventBus
from .models import Clip, ClipCreate
from .twitch import ClipUrlError, parse_clip_url
from .worker import Worker

router = APIRouter(prefix="/api")


def _db(request: Request) -> DB:
    return request.app.state.db


def sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_stream(
    bus: EventBus,
    is_disconnected: Callable[[], Awaitable[bool]],
    heartbeat: float = 15.0,
) -> AsyncIterator[str]:
    q = bus.subscribe()
    try:
        yield ": connected\n\n"
        while True:
            if await is_disconnected():
                return
            try:
                ev = await asyncio.wait_for(q.get(), heartbeat)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield sse_frame(ev["type"], ev["data"])
    finally:
        bus.unsubscribe(q)


@router.get("/events")
async def events(request: Request):
    return StreamingResponse(
        sse_stream(request.app.state.bus, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/clips")
def list_clips(request: Request, status: str | None = None) -> list[Clip]:
    return _db(request).list_clips(status=status)


@router.get("/clips/{clip_id}")
def get_clip(request: Request, clip_id: int) -> Clip:
    clip = _db(request).get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return clip


@router.post("/clips", status_code=201)
def create_clip(request: Request, body: ClipCreate, response: Response) -> Clip:
    db = _db(request)
    try:
        parsed = parse_clip_url(body.url)
    except ClipUrlError as e:
        raise HTTPException(422, f"invalid clip URL: {e}") from e
    override = None if body.layout == "auto" else body.layout

    existing = db.find_clip_by_slug(parsed.slug)
    if existing is not None:
        response.status_code = 200
        if existing.status == "failed":  # retry: cached stages are skipped by params_hash
            clip = db.update_clip(
                existing.id, status="queued", error=None, language=body.language,
                layout_override=override,
            )
            db.enqueue_job("pipeline", existing.id)
            request.app.state.bus.publish("clip", clip.model_dump())
            _poke(request)
            return clip
        return existing

    clip = db.create_clip(
        parsed.canonical_url, parsed.slug, channel=parsed.channel,
        language=body.language, layout_override=override,
    )
    db.enqueue_job("pipeline", clip.id)
    request.app.state.bus.publish("clip", clip.model_dump())
    _poke(request)
    return clip


def _poke(request: Request) -> None:
    worker: Worker | None = getattr(request.app.state, "worker", None)
    if worker:
        worker.poke()
