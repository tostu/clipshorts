"""REST + SSE routes."""

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from . import metadata as metadata_mod
from .captions import clean_hook, clean_words, expand_edited
from .db import DB
from .events import EventBus
from .media import browser_playable
from .models import (
    CaptionsOut,
    CaptionsUpdate,
    Channel,
    ChannelCreate,
    ChannelUpdate,
    Clip,
    ClipCreate,
    HookUpdate,
    InboxItem,
    Metadata,
    MetadataUpdate,
    Publication,
    PublicationCreate,
    PublicationUpdate,
    RerenderBody,
)
from .pipeline import (
    CAPTIONS_EDIT_FILE,
    HOOK_FILE,
    OUTPUT_FILE,
    SOURCE_FILE,
    TRANSCRIPT_FILE,
    _edit_words,
    _read_json,
    hook_text,
)
from .sync import Syncer
from .twitch import ClipUrlError, parse_channel, parse_clip_url
from .worker import Worker

router = APIRouter(prefix="/api")


def _db(request: Request) -> DB:
    return request.app.state.db


def _output_path(request: Request, clip_id: int):
    return request.app.state.cfg.clip_dir(clip_id) / OUTPUT_FILE


def _with_output(request: Request, clip: Clip) -> Clip:
    if _output_path(request, clip.id).is_file():
        clip.output_url = f"/api/clips/{clip.id}/output"
    if (request.app.state.cfg.clip_dir(clip.id) / SOURCE_FILE).is_file():
        clip.raw_url = f"/api/clips/{clip.id}/raw"
    try:
        path = request.app.state.cfg.clip_dir(clip.id) / TRANSCRIPT_FILE
        lang = json.loads(path.read_text()).get("language")
        clip.detected_language = lang if isinstance(lang, str) and lang else None
    except (OSError, ValueError, AttributeError):
        pass
    clip.hook = hook_text(request.app.state.cfg.clip_dir(clip.id))
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


@router.delete("/clips/{clip_id}", status_code=204)
def delete_clip(request: Request, clip_id: int) -> Response:
    db = _db(request)
    clip = db.get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if clip.status in ("queued", "running"):
        raise HTTPException(409, f"clip is {clip.status}, cannot delete")
    db.delete_clip(clip_id)
    shutil.rmtree(request.app.state.cfg.clip_dir(clip_id), ignore_errors=True)
    request.app.state.bus.publish("deleted", {"id": clip_id})
    return Response(status_code=204)


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
    fields: dict = {}
    if body is not None and body.layout is not None:
        fields["layout_override"] = None if body.layout == "auto" else body.layout
    # a clip that never got past download/probe only needs the cheap fetch again
    kind = "fetch" if clip.status == "failed" and clip.stage in ("download", "probe") \
        else "pipeline"
    return _with_output(request, _requeue(request, clip_id, kind=kind, **fields))


@router.post("/clips/{clip_id}/process")
def process_clip(request: Request, clip_id: int, body: RerenderBody | None = None) -> Clip:
    """Start the full pipeline (layout, transcribe, captions, render) for a fetched clip.
    Fetched clips only have the raw download and metadata; this is the manual go-ahead."""
    clip = _clip_or_404(request, clip_id)
    if clip.status != "fetched":
        raise HTTPException(409, f"clip is {clip.status}, not waiting to be processed")
    fields: dict = {}
    if body is not None and body.layout is not None:
        fields["layout_override"] = None if body.layout == "auto" else body.layout
    return _with_output(request, _requeue(request, clip_id, **fields))


def _requeue(request: Request, clip_id: int, kind: str = "pipeline", **fields) -> Clip:
    db = _db(request)
    clip = db.update_clip(clip_id, status="queued", error=None, **fields)
    db.enqueue_job(kind, clip_id)
    request.app.state.bus.publish("clip", clip.model_dump())
    _poke(request)
    return clip


# -- caption text edits --------------------------------------------------------------------
def _idle_clip(request: Request, clip_id: int) -> Clip:
    clip = _clip_or_404(request, clip_id)
    if clip.status in ("queued", "running"):
        raise HTTPException(409, f"clip is already {clip.status}")
    return clip


@router.get("/clips/{clip_id}/captions")
def get_captions(request: Request, clip_id: int) -> CaptionsOut:
    """Words shown as captions: the hand-edited list if there is one, else the transcript."""
    _clip_or_404(request, clip_id)
    d = request.app.state.cfg.clip_dir(clip_id)
    edit = _edit_words(d)
    if edit is not None:
        return CaptionsOut(words=edit, edited=True)
    words = clean_words(_read_json(d / TRANSCRIPT_FILE).get("words", []))
    return CaptionsOut(words=[{"start": w.start, "end": w.end, "text": w.text} for w in words],
                       edited=False)


@router.put("/clips/{clip_id}/captions")
def update_captions(request: Request, clip_id: int, body: CaptionsUpdate) -> Clip:
    """Save hand-edited caption words and re-run captions + render (nothing else)."""
    _idle_clip(request, clip_id)
    d = request.app.state.cfg.clip_dir(clip_id)
    if not (d / TRANSCRIPT_FILE).is_file():
        raise HTTPException(409, "no transcript yet")
    items = [w.model_dump() for w in body.words]
    if any(w["end"] < w["start"] for w in items):
        raise HTTPException(422, "word end before start")
    words = expand_edited(items)
    tmp = d / (CAPTIONS_EDIT_FILE + ".tmp")
    tmp.write_text(json.dumps({"words": words}, indent=2, ensure_ascii=False))
    tmp.replace(d / CAPTIONS_EDIT_FILE)
    return _with_output(request, _requeue(request, clip_id))


@router.delete("/clips/{clip_id}/captions")
def reset_captions(request: Request, clip_id: int) -> Clip:
    """Drop hand edits, back to the Whisper text, and re-render."""
    _idle_clip(request, clip_id)
    (request.app.state.cfg.clip_dir(clip_id) / CAPTIONS_EDIT_FILE).unlink(missing_ok=True)
    return _with_output(request, _requeue(request, clip_id))


@router.put("/clips/{clip_id}/hook")
def update_hook(request: Request, clip_id: int, body: HookUpdate) -> Clip:
    """Set the hook text shown over the first seconds (empty removes it); re-runs captions +
    render only."""
    _idle_clip(request, clip_id)
    path = request.app.state.cfg.clip_dir(clip_id) / HOOK_FILE
    text = clean_hook(body.text)
    if text:
        tmp = path.with_name(HOOK_FILE + ".tmp")
        tmp.write_text(json.dumps({"text": text}, ensure_ascii=False))
        tmp.replace(path)
    else:
        path.unlink(missing_ok=True)
    return _with_output(request, _requeue(request, clip_id))


# -- review inbox -----------------------------------------------------------------------
@router.get("/inbox")
def inbox(request: Request) -> list[InboxItem]:
    """Finished shorts still waiting for approve / reject, newest first."""
    items = []
    for c in _db(request).list_inbox():
        c = _with_output(request, c)
        items.append(InboxItem(**c.model_dump(include=set(InboxItem.model_fields))))
    return items


def _review(request: Request, clip_id: int, review_status: str) -> Clip:
    db = _db(request)
    clip = _clip_or_404(request, clip_id)
    if clip.status != "done":
        raise HTTPException(409, "clip has no finished render")
    if review_status == "rejected":
        # keep the slug recorded as seen: sync must never queue a rejected clip again
        db.mark_seen(clip.slug, clip.channel)
    updated = db.set_review(clip_id, review_status)
    assert updated
    request.app.state.bus.publish("clip", updated.model_dump())
    return _with_output(request, updated)


@router.post("/clips/{clip_id}/approve")
def approve_clip(request: Request, clip_id: int) -> Clip:
    return _review(request, clip_id, "approved")


@router.post("/clips/{clip_id}/reject")
def reject_clip(request: Request, clip_id: int) -> Clip:
    return _review(request, clip_id, "rejected")


@router.post("/clips/{clip_id}/review/reset")
def reset_review(request: Request, clip_id: int) -> Clip:
    """Back to pending (undo a decision)."""
    return _review(request, clip_id, "pending")


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


@router.get("/clips/{clip_id}/raw")
def get_raw(request: Request, clip_id: int):
    """The downloaded source clip (preview before processing). Supports range requests."""
    clip = _clip_or_404(request, clip_id)
    cfg = request.app.state.cfg
    path = cfg.clip_dir(clip_id) / SOURCE_FILE
    if not path.is_file():
        raise HTTPException(404, "no download yet")
    path = browser_playable(cfg, path)
    return FileResponse(path, media_type="video/mp4", filename=f"{clip.slug}-raw.mp4",
                        content_disposition_type="inline")


# -- metadata + export (phase 4) ----------------------------------------------------------
def _clip_or_404(request: Request, clip_id: int) -> Clip:
    clip = _db(request).get_clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return clip


def _current_metadata(request: Request, clip: Clip) -> Metadata:
    """Stored (hand-edited) text, else the defaults: clip title + creator, game, fixed hashtags."""
    return _db(request).get_metadata(clip.id) or metadata_mod.default_metadata(
        request.app.state.cfg, clip)


@router.get("/clips/{clip_id}/metadata")
def get_metadata(request: Request, clip_id: int) -> Metadata:
    return _current_metadata(request, _clip_or_404(request, clip_id))


@router.put("/clips/{clip_id}/metadata")
def update_metadata(request: Request, clip_id: int, body: MetadataUpdate) -> Metadata:
    clip = _clip_or_404(request, clip_id)
    db = _db(request)
    cfg = request.app.state.cfg
    cur = _current_metadata(request, clip)
    tags = cur.hashtags
    if body.hashtags is not None:
        tags = metadata_mod.merge_hashtags(body.hashtags, cfg.max_hashtags)
    db.set_metadata(
        clip_id,
        cur.title if body.title is None
        else metadata_mod.clip_text(body.title, metadata_mod.TITLE_MAX),
        cur.caption if body.caption is None
        else metadata_mod.clip_text(body.caption, metadata_mod.CAPTION_MAX),
        tags, "manual",
    )
    return _current_metadata(request, clip)


@router.get("/clips/{clip_id}/export")
def export_clip(request: Request, clip_id: int) -> dict:
    """Copy-paste text per platform plus the MP4 download URL."""
    clip = _with_output(request, _clip_or_404(request, clip_id))
    meta = _current_metadata(request, clip)
    return {
        "video_url": clip.output_url,
        "download_url": f"{clip.output_url}?download=1" if clip.output_url else None,
        "metadata": meta,
        "platforms": metadata_mod.export_text(clip, meta),
    }


@router.get("/clips/{clip_id}/publications")
def list_publications(request: Request, clip_id: int) -> list[Publication]:
    _clip_or_404(request, clip_id)
    return _db(request).list_publications(clip_id)


@router.post("/clips/{clip_id}/publications", status_code=201)
def add_publication(request: Request, clip_id: int, body: PublicationCreate) -> Publication:
    """'Mark as published': platform + URL, optionally feedback right away."""
    clip = _clip_or_404(request, clip_id)
    if clip.status != "done":
        raise HTTPException(409, "clip has no finished render")
    return _db(request).add_publication(
        clip_id, body.platform, body.url, body.published_at, body.views_24h, body.rating,
        body.note,
    )


@router.patch("/publications/{pub_id}")
def update_publication(request: Request, pub_id: int, body: PublicationUpdate) -> Publication:
    """Manual feedback: views at 24 h, rating, note."""
    db = _db(request)
    if not db.get_publication(pub_id):
        raise HTTPException(404, "publication not found")
    pub = db.update_publication(pub_id, **body.model_dump(exclude_unset=True))
    assert pub
    return pub


@router.delete("/publications/{pub_id}", status_code=204)
def delete_publication(request: Request, pub_id: int) -> Response:
    if not _db(request).delete_publication(pub_id):
        raise HTTPException(404, "publication not found")
    return Response(status_code=204)


@router.get("/report/ratings")
def rating_report(request: Request) -> list[dict]:
    """Channel x layout x rating counts from the manual feedback."""
    return _db(request).rating_report()


# -- followed channels (auto-sync) ---------------------------------------------------------
def _channel_or_404(request: Request, channel_id: int) -> Channel:
    ch = _db(request).get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")
    return ch


@router.get("/channels")
def list_channels(request: Request) -> list[Channel]:
    return _db(request).list_channels()


@router.post("/channels", status_code=201)
async def add_channel(request: Request, body: ChannelCreate) -> Channel:
    """Follow a channel. Clips already up are baselined by the first sync, not queued."""
    db = _db(request)
    try:
        name = parse_channel(body.name)
    except ClipUrlError as e:
        raise HTTPException(422, f"invalid channel: {e}") from e
    if db.find_channel(name):
        raise HTTPException(409, "channel already followed")
    ch = db.add_channel(name, body.min_views, body.language)
    syncer: Syncer | None = getattr(request.app.state, "syncer", None)
    if syncer and request.app.state.cfg.sync_enabled and request.app.state.cfg.worker_enabled:
        syncer.sync_in_background(ch.id)  # baseline right away, off the request
    return ch


@router.patch("/channels/{channel_id}")
def update_channel(request: Request, channel_id: int, body: ChannelUpdate) -> Channel:
    _channel_or_404(request, channel_id)
    ch = _db(request).update_channel(channel_id, **body.model_dump(exclude_unset=True,
                                                                    exclude_none=True))
    assert ch
    return ch


@router.delete("/channels/{channel_id}", status_code=204)
def delete_channel(request: Request, channel_id: int) -> Response:
    if not _db(request).delete_channel(channel_id):
        raise HTTPException(404, "channel not found")
    return Response(status_code=204)


@router.post("/channels/{channel_id}/sync")
async def sync_channel_now(request: Request, channel_id: int) -> Channel:
    """Check the channel for new clips right now (also works while auto-sync is off)."""
    _channel_or_404(request, channel_id)
    await request.app.state.syncer.sync_one(channel_id)
    return _channel_or_404(request, channel_id)


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
            db.enqueue_job("fetch", existing.id)
            request.app.state.bus.publish("clip", clip.model_dump())
            _poke(request)
            return clip
        return existing

    clip = db.create_clip(
        parsed.canonical_url, parsed.slug, channel=parsed.channel,
        language=body.language, layout_override=override,
    )
    db.enqueue_job("fetch", clip.id)  # raw download + title only; /process starts the rest
    request.app.state.bus.publish("clip", clip.model_dump())
    _poke(request)
    return clip


def _poke(request: Request) -> None:
    worker: Worker | None = getattr(request.app.state, "worker", None)
    if worker:
        worker.poke()
