"""API / row models."""

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["queued", "running", "done", "failed", "waiting"]
ReviewStatus = Literal["pending", "approved", "rejected"]
Language = Literal["auto", "de", "en"]
LayoutChoice = Literal["auto", "fullcam", "stacked", "face", "content"]


class Clip(BaseModel):
    id: int
    source_url: str
    slug: str
    channel: str | None = None
    title: str | None = None
    game: str | None = None
    duration_s: float | None = None
    status: str
    stage: str | None = None
    progress: float = 0.0
    error: str | None = None
    language: str = "auto"
    layout_type: str | None = None
    layout_confidence: float | None = None
    layout_override: str | None = None
    review_status: ReviewStatus = "pending"  # inbox triage of a finished short
    reviewed_at: str | None = None
    created_at: str
    updated_at: str
    output_url: str | None = None  # filled by the API when output.mp4 exists
    detected_language: str | None = None  # filled by the API from transcript.json


class InboxItem(BaseModel):
    """One pending short in the review inbox."""
    id: int
    slug: str
    title: str | None = None
    channel: str | None = None
    game: str | None = None
    layout_type: str | None = None
    language: str = "auto"
    detected_language: str | None = None
    duration_s: float | None = None
    output_url: str | None = None
    source_url: str
    created_at: str


class ClipCreate(BaseModel):
    url: str
    language: Language = "auto"
    layout: LayoutChoice = "auto"


class RerenderBody(BaseModel):
    layout: LayoutChoice | None = None  # None = keep the current layout choice


class CaptionWord(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(max_length=200)


class CaptionsUpdate(BaseModel):
    words: list[CaptionWord] = Field(max_length=3000)


class CaptionsOut(BaseModel):
    words: list[CaptionWord]
    edited: bool  # True when hand edits replace the Whisper transcript


Platform = Literal["tiktok", "youtube", "instagram", "other"]
Rating = Literal["good", "ok", "flop"]


class Metadata(BaseModel):
    """Post text for one short. `hashtags` has no leading '#'."""
    title: str = ""  # hook line, also the YouTube Shorts title
    caption: str = ""  # TikTok / Reels caption (without hashtags)
    hashtags: list[str] = []
    source: str = "manual"  # auto (defaults, not stored) | manual
    updated_at: str | None = None


class MetadataUpdate(BaseModel):
    title: str | None = None
    caption: str | None = None
    hashtags: list[str] | None = None


class PublicationCreate(BaseModel):
    platform: Platform
    url: str | None = None
    published_at: str | None = None  # ISO; defaults to now
    views_24h: int | None = None
    rating: Rating | None = None
    note: str | None = None


class PublicationUpdate(BaseModel):
    url: str | None = None
    views_24h: int | None = None
    rating: Rating | None = None
    note: str | None = None


class Publication(BaseModel):
    id: int
    clip_id: int
    platform: str
    url: str | None = None
    published_at: str
    views_24h: int | None = None
    rating: str | None = None
    note: str | None = None


class Channel(BaseModel):
    """A followed Twitch channel whose new clips are queued automatically."""
    id: int
    name: str  # lowercase login
    enabled: bool = True
    min_views: int = 0  # clips below this stay pending until they reach it (or age out)
    language: str = "auto"
    created_at: str
    last_sync_at: str | None = None
    last_error: str | None = None
    last_queued: int = 0  # clips queued by the latest sync
    total_queued: int = 0


class ChannelCreate(BaseModel):
    name: str  # login, or a twitch.tv/<channel> URL
    min_views: int = Field(default=0, ge=0)
    language: Language = "auto"


class ChannelUpdate(BaseModel):
    enabled: bool | None = None
    min_views: int | None = Field(default=None, ge=0)
    language: Language | None = None


class Job(BaseModel):
    id: int
    kind: str
    ref_id: int
    status: str
    attempts: int
    error: str | None = None
    created_at: str
    updated_at: str
