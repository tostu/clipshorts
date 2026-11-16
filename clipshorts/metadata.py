"""Post metadata (title/hook, caption, hashtags) via a local LLM, plus export text.

The LLM is Ollama's /api/chat with a JSON-schema `format` (structured output) and
`keep_alive` 0 so the model leaves RAM right after the call. Generation runs on demand from the
API, not in the pipeline: a missing or slow LLM must never fail a render.
"""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Config
from .models import Clip, Metadata

TRANSCRIPT_FILE = "transcript.json"
MAX_TRANSCRIPT_CHARS = 3000
TITLE_MAX = 90  # YouTube Shorts titles are capped at 100 chars
CAPTION_MAX = 300
TAG_MAX_LEN = 30

# (cfg, system prompt, user prompt, json schema) -> parsed JSON object
LLM = Callable[[Config, str, str, dict[str, Any]], dict[str, Any]]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "caption", "hashtags"],
}

LANG_NAMES = {"de": "German", "en": "English"}


class MetadataError(RuntimeError):
    """Expected failure (LLM unreachable, bad output); the message is shown to the user."""


def transcript_text(clip_dir: Path) -> tuple[str, str | None]:
    """Spoken text and language from transcript.json ('' when there is none)."""
    try:
        data = json.loads((clip_dir / TRANSCRIPT_FILE).read_text())
    except (OSError, ValueError):
        return "", None
    if not isinstance(data, dict):
        return "", None
    words = [str(w.get("word", "")).strip() for w in data.get("words") or [] if isinstance(w, dict)]
    text = " ".join(w for w in words if w)
    lang = data.get("language")
    return text[:MAX_TRANSCRIPT_CHARS], lang if isinstance(lang, str) and lang else None


def build_prompts(clip: Clip, transcript: str, language: str | None) -> tuple[str, str]:
    lang = LANG_NAMES.get(language or "", "the language of the transcript")
    system = (
        "You write post text for short vertical videos (TikTok, YouTube Shorts, Reels) cut "
        "from Twitch clips. Answer with JSON only. Rules: "
        f"write in {lang}; title = one punchy hook line, at most {TITLE_MAX} characters, "
        "no hashtags and no clickbait lies; caption = one or two short sentences, no hashtags; "
        "hashtags = 3 to 6 specific tags (game, topic, moment) without the # sign, no spaces. "
        "Only use facts from the input. Do not invent names, scores or quotes."
    )
    parts = [f"Clip title: {clip.title or '-'}", f"Game: {clip.game or '-'}",
             f"Channel: {clip.channel or '-'}"]
    if clip.duration_s:
        parts.append(f"Duration: {clip.duration_s:.0f} s")
    parts.append(f"Transcript: {transcript or '(no speech)'}")
    return system, "\n".join(parts)


def chat_json(cfg: Config, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Ollama chat with structured output. Raises MetadataError on any failure."""
    body = {
        "model": cfg.llm_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": schema,
        "stream": False,
        "keep_alive": cfg.llm_keep_alive,
        "options": {"temperature": 0.7},
    }
    req = urllib.request.Request(
        cfg.llm_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout_s) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        raise MetadataError(f"LLM returned HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise MetadataError(f"LLM not reachable at {cfg.llm_url}: {e}") from e
    except ValueError as e:
        raise MetadataError("LLM sent a response that is not JSON") from e
    try:
        out = json.loads(payload["message"]["content"])
    except (KeyError, TypeError, ValueError) as e:
        raise MetadataError("LLM output is not the requested JSON") from e
    if not isinstance(out, dict):
        raise MetadataError("LLM output is not a JSON object")
    return out


def clean_tag(tag: str) -> str:
    """'#Game Over!' -> 'GameOver'. Keeps letters and digits (umlauts included)."""
    return re.sub(r"[^\w]", "", tag.strip().lstrip("#"), flags=re.UNICODE).strip("_")[:TAG_MAX_LEN]


def merge_hashtags(fixed: list[str], specific: list[str], limit: int) -> list[str]:
    """Fixed channel tags first, then clip-specific ones; de-duplicated case-insensitively."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in [*fixed, *specific]:
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


def normalize(raw: dict[str, Any], cfg: Config) -> Metadata:
    """LLM output -> sane Metadata: trimmed lengths, clean tags, fixed tags merged in."""
    title = clip_text(raw.get("title") or "", TITLE_MAX)
    caption = clip_text(raw.get("caption") or "", CAPTION_MAX)
    tags = raw.get("hashtags")
    specific = [str(t) for t in tags] if isinstance(tags, list) else []
    if not title and not caption:
        raise MetadataError("LLM returned neither title nor caption")
    return Metadata(
        title=title, caption=caption, source="llm",
        hashtags=merge_hashtags(cfg.fixed_hashtag_list, specific, cfg.max_hashtags),
    )


def default_metadata(cfg: Config, clip: Clip) -> Metadata:
    """No-LLM text: Twitch clip title as title, hashtags = creator, game, then the fixed set.
    The creator comes from clip.channel (read from the URL, or from Twitch metadata for
    clips.twitch.tv links)."""
    tags = merge_hashtags([clip.channel or "", clip.game or "", *cfg.fixed_hashtag_list], [],
                          cfg.max_hashtags)
    return Metadata(title=clip_text(clip.title or "", TITLE_MAX), hashtags=tags, source="auto")


def generate(cfg: Config, clip: Clip, llm: LLM = chat_json) -> Metadata:
    text, language = transcript_text(cfg.clip_dir(clip.id))
    system, user = build_prompts(clip, text, language)
    return normalize(llm(cfg, system, user, SCHEMA), cfg)


def hashtag_line(tags: list[str]) -> str:
    return " ".join("#" + t for t in tags)


def export_text(clip: Clip, meta: Metadata) -> dict[str, dict[str, str]]:
    """Per-platform copy-paste text. TikTok/Reels put hashtags in the caption; Shorts take the
    title (hook) plus a description, and `#Shorts` marks the video as a Short."""
    tags = hashtag_line(meta.hashtags)
    tiktok = "\n\n".join(p for p in (meta.title, meta.caption, tags) if p)
    yt_title = meta.title or clip.title or clip.slug
    yt_desc = "\n\n".join(p for p in (meta.caption, tags, "#Shorts") if p)
    return {
        "tiktok": {"caption": tiktok},
        "instagram": {"caption": tiktok},
        "youtube": {"title": clip_text(yt_title, 100), "description": yt_desc},
    }
