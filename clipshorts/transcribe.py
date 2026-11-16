"""faster-whisper wrapper: lazy model load, idle unload, language clamping.

The heavy import and the model only exist once `transcribe()` is called, so the unit-test run
never needs Whisper. Tests inject `model_factory` or replace `Tools.transcribe` entirely.
"""

import logging
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

TRANSCRIPT_VERSION = 1
# Whisper settings that change the output; part of the stage hash.
WHISPER_SETTINGS = {"vad_filter": True, "word_timestamps": True,
                    "condition_on_previous_text": False, "beam_size": 5}


# -- pure helpers --------------------------------------------------------------------------
def clamp_language(probs: Iterable[tuple[str, float]], allowed: list[str],
                   detected: str | None = None) -> str:
    """Pick the most probable language among `allowed`. Falls back to allowed[0].

    `probs` is Whisper's all_language_probs. If no allowed language appears in it (or it is
    empty) the first allowed language wins, since auto-detect may only choose from the set.
    """
    allowed = [a.lower() for a in allowed]
    if not allowed:
        return (detected or "en").lower()
    best: tuple[str, float] | None = None
    for lang, p in probs:
        if lang.lower() in allowed and (best is None or p > best[1]):
            best = (lang.lower(), float(p))
    if best:
        return best[0]
    if detected and detected.lower() in allowed:
        return detected.lower()
    return allowed[0]


def build_prompt(title: str | None, game: str | None, channel: str | None) -> str | None:
    """initial_prompt from free Twitch context. None when there is nothing to say."""
    parts = []
    if title:
        parts.append(title.strip())
    if game:
        parts.append(f"Spiel: {game.strip()}")
    if channel:
        parts.append(f"Kanal: {channel.strip()}")
    return ". ".join(p for p in parts if p) or None


def empty_transcript(language: str | None, source: str = "none") -> dict[str, Any]:
    return {"version": TRANSCRIPT_VERSION, "language": language or "",
            "language_source": source, "language_probability": None, "duration": 0.0,
            "segments": [], "words": [], "settings": WHISPER_SETTINGS}


def build_transcript(segments: Iterable[Any], *, language: str, source: str,
                     probability: float | None, duration: float,
                     on_progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Convert faster-whisper segments (objects with start/end/text/words/...) to our JSON."""
    out_segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for seg in segments:
        idx = len(out_segments)
        out_segments.append({
            "id": idx, "start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
            "text": str(seg.text).strip(),
            "avg_logprob": _num(getattr(seg, "avg_logprob", None)),
            "no_speech_prob": _num(getattr(seg, "no_speech_prob", None)),
        })
        for w in getattr(seg, "words", None) or []:
            words.append({"word": str(w.word).strip(), "start": round(float(w.start), 3),
                          "end": round(float(w.end), 3),
                          "probability": _num(getattr(w, "probability", None)), "seg": idx})
        if on_progress and duration > 0:
            on_progress(min(1.0, float(seg.end) / duration))
    return {"version": TRANSCRIPT_VERSION, "language": language, "language_source": source,
            "language_probability": _num(probability), "duration": round(duration, 3),
            "segments": out_segments, "words": [w for w in words if w["word"]],
            "settings": WHISPER_SETTINGS}


def _num(v: Any) -> float | None:
    return None if v is None else round(float(v), 4)


# -- model wrapper ---------------------------------------------------------------------------
class WhisperTranscriber:
    """Lazy faster-whisper model with idle unload.

    `model_factory(name, compute_type, download_root, cpu_threads)` returns an object with
    faster-whisper's `detect_language` / `transcribe`; `audio_loader(path)` returns the
    decoded 16 kHz mono signal. Both are injectable for tests.
    """

    def __init__(self, cfg: Config, model_factory: Callable[..., Any] | None = None,
                 audio_loader: Callable[[Path], Any] | None = None):
        self.cfg = cfg
        self._factory = model_factory or _default_factory
        self._load_audio = audio_loader or _default_audio_loader
        self._model: Any = None
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self.loads = 0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> Any:
        if self._model is None:
            log.info("loading whisper model %s (%s)", self.cfg.whisper_model,
                     self.cfg.whisper_compute_type)
            self._model = self._factory(self.cfg.whisper_model, self.cfg.whisper_compute_type,
                                        self.cfg.whisper_download_root,
                                        self.cfg.whisper_threads)
            self.loads += 1
        return self._model

    def unload(self) -> bool:
        """Drop the model unless a transcription is running. True if it was unloaded."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            was = self._model is not None
            self._model = None
            if was:
                log.info("whisper model unloaded (idle)")
            return was
        finally:
            self._lock.release()

    def _schedule_unload(self) -> None:
        minutes = self.cfg.whisper_idle_minutes
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if minutes <= 0:
            self._model = None
            return
        t = threading.Timer(minutes * 60, self.unload)
        t.daemon = True
        t.start()
        self._timer = t

    def __call__(self, audio: Path, *, language: str | None, allowed: list[str],
                 prompt: str | None, on_progress: Callable[[float], None] | None = None
                 ) -> dict[str, Any]:
        with self._lock:
            try:
                return self._run(audio, language, allowed, prompt, on_progress)
            finally:
                self._schedule_unload()

    def _run(self, audio_path: Path, language: str | None, allowed: list[str],
             prompt: str | None, on_progress: Callable[[float], None] | None) -> dict[str, Any]:
        model = self._ensure()
        signal = self._load_audio(audio_path)
        duration = len(signal) / 16000.0
        prob: float | None = None
        if language:
            lang, source = language.lower(), "manual"
        elif len(allowed) == 1:
            lang, source = allowed[0].lower(), "configured"
        else:
            source = "auto"
            try:
                detected, prob, all_probs = model.detect_language(
                    signal, vad_filter=True, language_detection_segments=2)
                lang = clamp_language(all_probs, allowed, detected)
            except Exception:  # noqa: BLE001 - e.g. VAD found no speech at all
                log.warning("language detection failed, using %s", allowed[0], exc_info=True)
                lang = allowed[0].lower()
        segments, info = model.transcribe(
            signal, language=lang, initial_prompt=prompt, vad_filter=True,
            word_timestamps=True, condition_on_previous_text=False, beam_size=5,
        )
        if source == "auto" and prob is None:
            prob = getattr(info, "language_probability", None)
        return build_transcript(segments, language=lang, source=source, probability=prob,
                                duration=duration or float(getattr(info, "duration", 0) or 0),
                                on_progress=on_progress)


def _default_factory(name: str, compute_type: str, download_root: Path | None,
                     threads: int) -> Any:
    from faster_whisper import WhisperModel  # lazy: heavy import

    return WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=threads,
                        download_root=str(download_root) if download_root else None)


def _default_audio_loader(path: Path) -> Any:
    from faster_whisper import decode_audio

    return decode_audio(str(path), sampling_rate=16000)


_shared: WhisperTranscriber | None = None
_shared_lock = threading.Lock()


def shared_transcriber(cfg: Config) -> WhisperTranscriber:
    """Process-wide transcriber so the model survives between clips (until idle unload)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = WhisperTranscriber(cfg)
        _shared.cfg = cfg
        return _shared


def default_transcribe(audio: Path, *, cfg: Config, language: str | None, allowed: list[str],
                       prompt: str | None, on_progress: Callable[[float], None] | None = None
                       ) -> dict[str, Any]:
    return shared_transcriber(cfg)(audio, language=language, allowed=allowed, prompt=prompt,
                                   on_progress=on_progress)
