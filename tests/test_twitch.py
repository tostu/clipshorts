import pytest

from clipshorts.twitch import ClipUrlError, parse_clip_url

SLUG = "FamousSecretiveDolphinPoooound-aBc123"


@pytest.mark.parametrize(
    "url, channel",
    [
        (f"https://clips.twitch.tv/{SLUG}", None),
        (f"clips.twitch.tv/{SLUG}", None),
        (f"  http://clips.twitch.tv/{SLUG}/?foo=bar  ", None),
        (f"https://www.twitch.tv/SomeStreamer/clip/{SLUG}", "somestreamer"),
        (f"https://twitch.tv/some_streamer/clip/{SLUG}?filter=clips&range=7d", "some_streamer"),
        (f"https://m.twitch.tv/SomeStreamer/clip/{SLUG}", "somestreamer"),
        (f"https://m.twitch.tv/clip/{SLUG}", None),
        (f"HTTPS://CLIPS.TWITCH.TV/{SLUG}#t=3", None),
    ],
)
def test_valid(url, channel):
    p = parse_clip_url(url)
    assert p.slug == SLUG
    assert p.channel == channel
    assert p.canonical_url == f"https://clips.twitch.tv/{SLUG}"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not a url",
        "https://example.com/abc12345678",
        "https://twitch.tv/somestreamer",
        "https://twitch.tv/somestreamer/videos/123456789",
        "https://twitch.tv/somestreamer/clip",
        "https://clips.twitch.tv/",
        "https://clips.twitch.tv/short",
        "https://clips.twitch.tv/has space inside1",
        f"https://clips.twitch.tv/{SLUG}/extra",
        f"https://evil.com/clips.twitch.tv/{SLUG}",
        f"https://clips.twitch.tv.evil.com/{SLUG}",
        f"ftp://clips.twitch.tv/{SLUG}",
        f"https://twitch.tv/bad-chan!/clip/{SLUG}",
        "https://[broken/clip/abc",
    ],
)
def test_invalid(bad):
    with pytest.raises(ClipUrlError):
        parse_clip_url(bad)


def test_non_string():
    with pytest.raises(ClipUrlError):
        parse_clip_url(None)  # type: ignore[arg-type]
