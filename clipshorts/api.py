"""REST + SSE routes."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from .db import DB
from .events import EventBus
from .models import Clip, ClipCreate, RerenderBody
from .pipeline import OUTPUT_FILE
from .twitch import ClipUrlError, parse_clip_url
from .worker import Worker

router = APIRouter(prefix="/api")


def _db(request: Request) -> DB:
    return request.app.state.db


def _output_path(request: Request, clip_id: int):
    return request.app.state.cfg.clip_dir(clip_id) / OUTPUT_FILE


def _with_output(request: Request, clip: Clip) -> Clip:
    if _output_path(request, clip.id).is_file():
        clip.output_url = f"/api/clips/{clip.id}/output"
    return clip


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
    return [_with_output(request, c) for c in _db(request).list_clips(status=status)]


@router.get("/clips/{clip_id}")
def get_clip(request: Request, clip_id: int) -> Clip:
    clip = _db(request).get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return _with_output(request, clip)


@router.post("/clips/{clip_id}/rerender")
def rerender(request: Request, clip_id: int, body: RerenderBody | None = None) -> Clip:
    """Re-run the pipeline for a clip, optionally with a new layout. Cached stages are
    skipped, so a layout change costs only layout + render (no re-transcription)."""
    db = _db(request)
    clip = db.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if clip.status in ("queued", "running"):
        raise HTTPException(409, f"clip is already {clip.status}")
    fields: dict = {"status": "queued", "error": None}
    if body is not None and body.layout is not None:
        fields["layout_override"] = None if body.layout == "auto" else body.layout
    clip = db.update_clip(clip_id, **fields)
    db.enqueue_job("pipeline", clip_id)
    request.app.state.bus.publish("clip", clip.model_dump())
    _poke(request)
    return _with_output(request, clip)


@router.get("/clips/{clip_id}/output")
def get_output(request: Request, clip_id: int, download: bool = False):
    """The rendered MP4. Range requests are answered (preview seeking, resumable download)."""
    clip = _db(request).get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    path = _output_path(request, clip_id)
    if not path.is_file():
        raise HTTPException(404, "no output yet")
    return FileResponse(
        path, media_type="video/mp4", filename=f"{clip.slug}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


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
