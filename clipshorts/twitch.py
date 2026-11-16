"""Twitch clip URL parsing (pure, no network)."""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{1,25}$")
_TWITCH_HOSTS = {"twitch.tv", "www.twitch.tv", "m.twitch.tv"}


class ClipUrlError(ValueError):
    """The input is not a recognisable Twitch clip URL."""


@dataclass(frozen=True)
class ParsedClip:
    slug: str
    channel: str | None
    canonical_url: str


def parse_channel(raw: str) -> str:
    """A Twitch login ('Name', '@name') or a twitch.tv/<channel>[/...] URL -> lowercase login."""
    if not isinstance(raw, str) or not raw.strip():
        raise ClipUrlError("empty channel")
    text = raw.strip().lstrip("@")
    if "/" in text or "." in text:
        if "://" not in text:
            text = "https://" + text
        try:
            parts = urlsplit(text)
            host = (parts.hostname or "").lower()
        except ValueError as e:
            raise ClipUrlError(f"invalid URL: {e}") from e
        if host not in _TWITCH_HOSTS:
            raise ClipUrlError("not a Twitch URL")
        segs = [s for s in parts.path.split("/") if s]
        text = segs[0] if segs else ""
    if not _CHANNEL_RE.match(text):
        raise ClipUrlError("invalid channel name")
    return text.lower()


def parse_clip_url(raw: str) -> ParsedClip:
    """Accepts clips.twitch.tv/<slug>, [www.|m.]twitch.tv/<channel>/clip/<slug>
    and [www.|m.]twitch.tv/clip/<slug>. Scheme is optional."""
    if not isinstance(raw, str) or not raw.strip():
        raise ClipUrlError("empty URL")
    text = raw.strip()
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
    except ValueError as e:
        raise ClipUrlError(f"invalid URL: {e}") from e
    if parts.scheme not in ("http", "https"):
        raise ClipUrlError("URL must be http(s)")
    segs = [s for s in parts.path.split("/") if s]

    channel: str | None = None
    slug: str | None = None
    if host == "clips.twitch.tv":
        if len(segs) == 1:
            slug = segs[0]
    elif host in _TWITCH_HOSTS:
        if len(segs) == 2 and segs[0] == "clip":
            slug = segs[1]
        elif len(segs) == 3 and segs[1] == "clip":
            channel, slug = segs[0], segs[2]
            if not _CHANNEL_RE.match(channel):
                raise ClipUrlError("invalid channel name")
    else:
        raise ClipUrlError("not a Twitch URL")

    if not slug or not _SLUG_RE.match(slug):
        raise ClipUrlError("no valid clip slug in URL")
    return ParsedClip(
        slug=slug,
        channel=channel.lower() if channel else None,
        canonical_url=f"https://clips.twitch.tv/{slug}",
    )
