"""API / row models."""

from typing import Literal

from pydantic import BaseModel

Status = Literal["queued", "running", "done", "failed", "waiting"]
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
    created_at: str
    updated_at: str
    output_url: str | None = None  # filled by the API when output.mp4 exists


class ClipCreate(BaseModel):
    url: str
    language: Language = "auto"
    layout: LayoutChoice = "auto"


class RerenderBody(BaseModel):
    layout: LayoutChoice | None = None  # None = keep the current layout choice


class Job(BaseModel):
    id: int
    kind: str
    ref_id: int
    status: str
    attempts: int
    error: str | None = None
    created_at: str
    updated_at: str
