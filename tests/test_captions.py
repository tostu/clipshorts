"""Chunker + ASS writer. Golden files live in tests/golden; regenerate with UPDATE_GOLDEN=1."""

import os
import re
from pathlib import Path

import pytest

from clipshorts.captions import (
    FONT_FILE,
    FONTS_DIR,
    CaptionStyle,
    captions_from_transcript,
    chunk_words,
    clean_words,
    is_sentence_end,
    text_width,
)

GOLDEN = Path(__file__).parent / "golden"
STYLE = CaptionStyle()


def seq(items, t0=0.0, dur=0.3):
    """Words from tokens; a float in the list inserts a pause of that many seconds."""
    out, t = [], t0
    for it in items:
        if isinstance(it, float):
            t += it
            continue
        out.append({"word": it, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur
    return out


def transcript(items, language, **kw):
    return {"language": language, "words": seq(items, **kw)}


def golden(name: str, actual: str) -> None:
    path = GOLDEN / f"{name}.ass"
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == actual, f"golden mismatch: {path.name}"


CASES = {
    "de_umlauts": ("de", "Das ist ja unglaublich schön, aber wir müssen größere Äpfel für "
                         "Übermut kaufen. Ölfässer sind groß!".split()),
    "de_abbreviations": ("de", ("Das kostet ca. 20 Euro z. B. beim Händler. Dr. Müller "
                                "kommt bzw. geht später. Fertig.").split()),
    "de_ordinals": ("de", "Am 3. Oktober gewinnt der 1. FC Köln den 2. Platz. Danke.".split()),
    "de_compound": ("de", ["Das", "ist", "die",
                           "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
                           "Donaudampfschifffahrtsgesellschaftskapitän", "ok"]),
    "en_basic": ("en", "Okay chat, I can't believe this actually worked. Mr. Smith said "
                       "it's over, e.g. done!".split()),
    "pauses": ("de", ["Moment", "mal", 0.8, "jetzt", "geht", "es", "los", 0.45, "wirklich",
                      "wahr", 0.51, "ende"]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_ass(name):
    lang, items = CASES[name]
    ass, _ = captions_from_transcript(transcript(items, lang), STYLE)
    golden(name, ass)


def test_empty_transcript_golden():
    ass, chunks = captions_from_transcript({"language": "de", "words": []}, STYLE)
    assert chunks == []
    assert "[Events]" in ass and "Dialogue:" not in ass
    golden("empty", ass)


def test_empty_transcript_missing_keys():
    ass, chunks = captions_from_transcript({}, STYLE)
    assert chunks == [] and "PlayResX: 1080" in ass


def test_header_style():
    ass, _ = captions_from_transcript({"words": []}, STYLE)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "WrapStyle: 2" in ass
    style = re.search(r"^Style: (.*)$", ass, re.M).group(1).split(",")
    assert style[1] == "Poppins" and style[7] == "-1"  # bold
    margins = style[-4:-1]
    assert int(margins[1]) > int(margins[0])  # extra right margin
    # caption baseline at 65% down: bottom margin is 35% of the height
    assert int(margins[2]) == 672


def test_chunks_max_three_words_and_all_words_kept():
    words = clean_words(seq("eins zwei drei vier fünf sechs sieben".split()))
    chunks = chunk_words(words, STYLE, "de")
    assert [len(c.words) for c in chunks] == [3, 3, 1]
    assert [w.text for c in chunks for w in c.words] == [w.text for w in words]


def test_break_at_sentence_end_and_pause():
    words = clean_words(seq(["Ja.", "Nein", "doch"]))
    assert [len(c.words) for c in chunk_words(words, STYLE, "de")] == [1, 2]
    words = clean_words(seq(["a", "b", 0.6, "c"]))
    assert [len(c.words) for c in chunk_words(words, STYLE, "de")] == [2, 1]
    words = clean_words(seq(["a", "b", 0.5, "c"]))  # exactly 0.5 s is not a pause
    assert [len(c.words) for c in chunk_words(words, STYLE, "de")] == [3]


def test_ordinal_and_abbreviation_do_not_break():
    words = clean_words(seq(["am", "3.", "Mai"]))
    assert [len(c.words) for c in chunk_words(words, STYLE, "de")] == [3]
    words = clean_words(seq(["ca.", "20", "Euro"]))
    assert [len(c.words) for c in chunk_words(words, STYLE, "de")] == [3]
    # English has no ordinal dot: "3." ends the sentence
    assert is_sentence_end("3.", "en") and not is_sentence_end("3.", "de")


@pytest.mark.parametrize("tok,lang,expected", [
    ("Ende.", "de", True), ("Was?", "de", True), ("Wow!", None, True), ("Hm…", "de", True),
    ("nein,", "de", False), ("bzw.", "de", False), ("z.B.", "de", False),
    ("Dr.", "de", False), ("Dr.", "en", False), ("B.", "de", False),
    ("\"Ende.\"", "de", True), ("(ca.)", "de", False), ("", "de", False),
    ("e.g.", "en", False), ("21.", "de", False), ("1.5", "de", False),
])
def test_is_sentence_end(tok, lang, expected):
    assert is_sentence_end(tok, lang) is expected


def test_every_chunk_fits_width_or_is_flagged():
    long_words = ["Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz", "ok",
                  "Donaudampfschifffahrtsgesellschaftskapitän", "und", "so"]
    chunks = chunk_words(clean_words(seq(long_words)), STYLE, "de")
    avail = STYLE.max_width - 2 * STYLE.outline
    for c in chunks:
        text = " ".join(w.text for w in c.words)
        assert c.font_size >= STYLE.min_font_size
        assert text_width(text, c.font_size) == pytest.approx(c.width)
        assert c.width <= avail or (c.overflow and c.font_size == STYLE.min_font_size)
    # compounds shrink instead of being split
    assert chunks[0].font_size < STYLE.font_size and len(chunks[0].words) == 1
    assert [w.text for c in chunks for w in c.words] == long_words


def test_width_limits_words_per_chunk():
    words = clean_words(seq(["Donaudampfschifffahrt", "Kapitänsmütze", "Wochenendausflug"]))
    chunks = chunk_words(words, STYLE, "de")
    assert len(chunks) > 1  # does not fit on one line at the base size
    assert STYLE.max_width == pytest.approx(864.0)


def test_normal_chunk_keeps_base_font():
    chunks = chunk_words(clean_words(seq(["Das", "ist", "gut"])), STYLE, "de")
    assert chunks[0].font_size == STYLE.font_size and not chunks[0].overflow


def test_clean_words_fixes_timing_and_drops_empty():
    raw = [{"word": " Hi ", "start": 1.0, "end": 1.0}, {"word": "  ", "start": 1.0, "end": 2.0},
           {"word": "du", "start": 0.5, "end": 0.6}]
    w = clean_words(raw)
    assert [x.text for x in w] == ["Hi", "du"]
    assert w[0].end > w[0].start and w[1].start >= w[0].end


def test_events_are_per_word_highlighted_and_continuous():
    ass, chunks = captions_from_transcript(transcript(["Das", "ist", "gut", "ok"], "de"), STYLE)
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(events) == 4
    first = events[0].split(",", 9)
    assert first[1] == "0:00:00.00" and first[2] == "0:00:00.30"
    assert "{\\1c&H0000D7FF&}Das{\\1c&H00FFFFFF&} ist gut" in events[0]
    # each non-final word ends exactly where the next starts (no flicker)
    assert events[0].split(",")[2] == events[1].split(",")[1]


def test_text_is_escaped():
    ass, _ = captions_from_transcript(
        {"words": [{"word": "{\\an8}hack", "start": 0, "end": 1}]}, STYLE)
    body = ass.split("Dialogue:")[1]
    assert "\\an8" not in body and "(an8)hack" in body


def test_font_is_bundled_with_umlauts():
    from PIL import ImageFont

    path = FONTS_DIR / FONT_FILE
    assert path.read_bytes()[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO")
    assert (FONTS_DIR / "OFL.txt").is_file()
    font = ImageFont.truetype(str(path), 80)
    for ch in "äöüÄÖÜß":
        assert font.getmask(ch).getbbox() is not None, ch
