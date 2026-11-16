"""Post metadata without an LLM: the Twitch clip title plus a fixed hashtag set.

Hashtags = creator (channel from the clip URL or Twitch metadata), game, then the configured
fixed set, de-duplicated and capped. The text can be edited by hand; export builds the
per-platform copy-paste text.
"""

import re

from .config import Config
from .models import Clip, Metadata

TITLE_MAX = 90  # YouTube Shorts titles are capped at 100 chars
CAPTION_MAX = 300
TAG_MAX_LEN = 30


def clean_tag(tag: str) -> str:
    """'#Game Over!' -> 'GameOver'. Keeps letters and digits (umlauts included)."""
    return re.sub(r"[^\w]", "", tag.strip().lstrip("#"), flags=re.UNICODE).strip("_")[:TAG_MAX_LEN]


def merge_hashtags(tags: list[str], limit: int) -> list[str]:
    """Clean and de-duplicate (case-insensitive), keep order, cap at `limit`."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = clean_tag(str(raw))
        if tag and tag.casefold() not in seen:
            seen.add(tag.casefold())
            out.append(tag)
        if len(out) >= limit:
            break
    return out


def clip_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def default_metadata(cfg: Config, clip: Clip) -> Metadata:
    """Clip title as title, hashtags = creator, game, then the fixed set."""
    tags = merge_hashtags([clip.channel or "", clip.game or "", *cfg.fixed_hashtag_list],
                          cfg.max_hashtags)
    return Metadata(title=clip_text(clip.title or "", TITLE_MAX), hashtags=tags, source="auto")


def hashtag_line(tags: list[str]) -> str:
    return " ".join("#" + t for t in tags)


def export_text(clip: Clip, meta: Metadata) -> dict[str, dict[str, str]]:
    """Per-platform copy-paste text. TikTok/Reels put hashtags in the caption; Shorts take the
    title plus a description, and `#Shorts` marks the video as a Short."""
    tags = hashtag_line(meta.hashtags)
    tiktok = "\n\n".join(p for p in (meta.title, meta.caption, tags) if p)
    yt_title = meta.title or clip.title or clip.slug
    yt_desc = "\n\n".join(p for p in (meta.caption, tags, "#Shorts") if p)
    return {
        "tiktok": {"caption": tiktok},
        "instagram": {"caption": tiktok},
        "youtube": {"title": clip_text(yt_title, 100), "description": yt_desc},
    }
