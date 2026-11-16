"""Caption chunker + ASS writer. Pure logic: no media, no Whisper.

Words (with timestamps) -> chunks of at most N words that fit the frame width -> one ASS
Dialogue event per word, showing the whole chunk with the active word highlighted.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

FONTS_DIR = Path(__file__).parent / "fonts"
FONT_FILE = "Poppins-Bold.ttf"
FONT_FAMILY = "Poppins"

# -- sentence ends (ported from shorts-flow, extended for English) -------------------------
DE_ABBREVIATIONS = {
    "z.b.", "d.h.", "u.a.", "z.t.", "u.v.m.", "o.ä.", "i.d.r.", "s.o.", "s.u.",
    "usw.", "bzw.", "ca.", "vgl.", "evtl.", "ggf.", "inkl.", "max.", "min.",
    "dr.", "prof.", "hr.", "fr.", "nr.", "str.", "std.", "tel.", "etc.", "sog.",
    # Whisper often writes "z. B." as separate tokens
    "z.", "d.", "u.", "i.", "o.", "s.", "v.",
}
EN_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
    "approx.", "no.", "inc.", "ltd.",
}
CLOSING_CHARS = "\"'“”„‘’«»)]}"
OPENING_CHARS = "\"'“”„‘’«»([{"


def is_sentence_end(token: str, language: str | None = None) -> bool:
    """True if `token` ends a sentence. Handles quotes, abbreviations and German ordinals."""
    word = token.strip().rstrip(CLOSING_CHARS).lstrip(OPENING_CHARS)
    if not word or word[-1] not in ".!?…":
        return False
    if word[-1] != ".":
        return True
    low = word.lower()
    lang = (language or "").lower()
    if lang == "de":
        abbr = DE_ABBREVIATIONS
    elif lang == "en":
        abbr = EN_ABBREVIATIONS
    else:
        abbr = DE_ABBREVIATIONS | EN_ABBREVIATIONS
    if low in abbr:
        return False
    stem = word[:-1]
    if lang != "en" and stem.isdigit():  # "3." = German ordinal
        return False
    # single-letter initial or part of "z. B.": "B."
    return not (len(stem) == 1 and stem.isalpha() and stem.isupper())


# -- style ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptionStyle:
    play_w: int = 1080
    play_h: int = 1920
    font_size: int = 84
    min_font_size: int = 30  # only extreme compounds get this small (63-char word ~ 31)
    max_words: int = 3
    max_width_frac: float = 0.8  # of the frame width
    pause_s: float = 0.5  # gap that forces a new caption
    outline: int = 7
    margin_l: int = 60
    margin_r: int = 140  # extra room for the TikTok button column
    bottom_frac: float = 0.65  # caption baseline position, fraction of frame height
    primary: str = "&H00FFFFFF"  # white
    highlight: str = "&H0000D7FF"  # gold (BGR)
    outline_colour: str = "&H00000000"
    hold_s: float = 0.2  # keep the last word visible this long (never into the next caption)

    @property
    def margin_v(self) -> int:
        return round(self.play_h * (1 - self.bottom_frac))

    @property
    def max_width(self) -> float:
        usable = self.play_w - self.margin_l - self.margin_r
        return min(self.play_w * self.max_width_frac, usable)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    words: list[Word]
    font_size: int
    width: float
    overflow: bool = False  # a single word is still too wide at the minimum font size

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


# -- text measurement ------------------------------------------------------------------------
_REF = 1000


@lru_cache(maxsize=4)
def _ref_font(path: str):
    from PIL import ImageFont

    font = ImageFont.truetype(path, _REF)
    asc, desc = font.getmetrics()
    return font, _REF / (asc + desc)


def text_width(text: str, font_size: int, font_path: Path | None = None) -> float:
    """Rendered width in pixels of `text` at ASS Fontsize `font_size`.

    ASS (libass/VSFilter) Fontsize is the line cell height (ascent + descent), not the em
    size, so the em is scaled down by cell/em before measuring with Pillow.
    """
    font, ratio = _ref_font(str(font_path or FONTS_DIR / FONT_FILE))
    return font.getlength(text) * font_size * ratio / _REF


# -- chunking --------------------------------------------------------------------------------
def clean_words(raw: list[dict[str, Any]]) -> list[Word]:
    """Normalise transcript words: strip text, drop empties, enforce monotonic timing."""
    out: list[Word] = []
    for w in raw:
        text = str(w.get("word", w.get("text", ""))).strip().replace("\n", " ")
        if not text:
            continue
        start = max(float(w["start"]), out[-1].end if out else 0.0)
        end = max(float(w["end"]), start + 0.05)
        out.append(Word(round(start, 3), round(end, 3), text))
    return out


def _fit(text: str, style: CaptionStyle, font_path: Path | None) -> tuple[int, float, bool]:
    """Largest font size <= style.font_size where text fits; (size, width, overflow)."""
    avail = style.max_width - 2 * style.outline
    size = style.font_size
    while True:
        width = text_width(text, size, font_path)
        if width <= avail:
            return size, width, False
        if size <= style.min_font_size:
            return size, width, True
        size = max(style.min_font_size, size - 2)


def chunk_words(words: list[Word], style: CaptionStyle, language: str | None = None,
                font_path: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            size, width, over = _fit(" ".join(w.text for w in cur), style, font_path)
            chunks.append(Chunk(list(cur), size, width, over))
            cur.clear()

    for w in words:
        if cur:
            gap = round(w.start - cur[-1].end, 3)
            full = len(cur) >= style.max_words
            ended = is_sentence_end(cur[-1].text, language)
            # width is judged at the base size: a longer chunk would need a smaller font
            too_wide = text_width(" ".join(x.text for x in [*cur, w]), style.font_size,
                                  font_path) > style.max_width - 2 * style.outline
            if full or ended or gap > style.pause_s or too_wide:
                flush()
        cur.append(w)
    flush()
    return chunks


# -- ASS writer ------------------------------------------------------------------------------
def _ts(cs: int) -> str:
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _header(style: CaptionStyle) -> str:
    s = style
    return (
        "[Script Info]\n"
        "; generated by clipshorts\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {s.play_w}\n"
        f"PlayResY: {s.play_h}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: None\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_FAMILY},{s.font_size},{s.primary},{s.primary},"
        f"{s.outline_colour},&H64000000,-1,0,0,0,100,100,0,0,1,{s.outline},0,2,"
        f"{s.margin_l},{s.margin_r},{s.margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def build_ass(chunks: list[Chunk], style: CaptionStyle) -> str:
    """One event per word: the chunk text with the active word in the highlight colour."""
    lines: list[str] = []
    for ci, ch in enumerate(chunks):
        next_start = chunks[ci + 1].start if ci + 1 < len(chunks) else None
        for wi, w in enumerate(ch.words):
            start = round(w.start * 100)
            if wi + 1 < len(ch.words):
                end = round(ch.words[wi + 1].start * 100)
            else:
                end = round((w.end + style.hold_s) * 100)
                if next_start is not None:
                    end = min(end, round(next_start * 100))
            end = max(end, start + 1)
            parts = []
            for j, x in enumerate(ch.words):
                t = _esc(x.text)
                parts.append(f"{{\\1c{style.highlight}&}}{t}{{\\1c{style.primary}&}}"
                             if j == wi else t)
            fs = f"{{\\fs{ch.font_size}}}" if ch.font_size != style.font_size else ""
            lines.append(
                f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{fs}{' '.join(parts)}"
            )
    return _header(style) + "\n".join(lines) + ("\n" if lines else "")


def captions_from_transcript(transcript: dict[str, Any], style: CaptionStyle,
                             font_path: Path | None = None) -> tuple[str, list[Chunk]]:
    words = clean_words(transcript.get("words", []))
    chunks = chunk_words(words, style, transcript.get("language"), font_path)
    return build_ass(chunks, style), chunks
