import wave
from types import SimpleNamespace as NS

import pytest

from clipshorts.config import Config
from clipshorts.transcribe import (
    WhisperTranscriber,
    build_transcript,
    clamp_language,
    empty_transcript,
)


# -- language clamp ----------------------------------------------------------------------
@pytest.mark.parametrize("probs,allowed,detected,expected", [
    ([("de", 0.6), ("en", 0.3)], ["de", "en"], "de", "de"),
    ([("nl", 0.7), ("de", 0.2), ("en", 0.05)], ["de", "en"], "nl", "de"),  # Dutch -> German
    ([("sv", 0.8), ("en", 0.15), ("de", 0.01)], ["de", "en"], "sv", "en"),
    ([("fr", 0.9)], ["de", "en"], "fr", "de"),  # none of the allowed: first allowed
    ([], ["en", "de"], None, "en"),
    ([], ["en", "de"], "DE", "de"),
    ([("de", 0.5), ("en", 0.4)], ["en"], "de", "en"),
    ([("DE", 0.5)], ["de"], None, "de"),
    ([("fr", 0.5)], [], "fr", "fr"),  # nothing configured: trust the detector
])
def test_clamp_language(probs, allowed, detected, expected):
    assert clamp_language(probs, allowed, detected) == expected


# -- transcript JSON -----------------------------------------------------------------------
def _seg(start, end, text, words, lp=-0.2, ns=0.01):
    return NS(start=start, end=end, text=f" {text}", avg_logprob=lp, no_speech_prob=ns,
              words=[NS(start=a, end=b, word=f" {w}", probability=p) for a, b, w, p in words])


def test_build_transcript_shape():
    segs = [_seg(0.0, 1.0, "Hallo Welt", [(0.0, 0.5, "Hallo", 0.98), (0.5, 1.0, "Welt", 0.9)]),
            _seg(1.2, 2.0, "ok", [(1.2, 2.0, "ok", 0.5)], lp=-0.9, ns=0.4)]
    seen = []
    t = build_transcript(segs, language="de", source="auto", probability=0.93, duration=2.0,
                         on_progress=seen.append)
    assert t["language"] == "de" and t["language_source"] == "auto"
    assert t["language_probability"] == 0.93 and t["duration"] == 2.0
    assert [s["text"] for s in t["segments"]] == ["Hallo Welt", "ok"]
    assert t["segments"][1]["avg_logprob"] == -0.9 and t["segments"][1]["no_speech_prob"] == 0.4
    assert [w["word"] for w in t["words"]] == ["Hallo", "Welt", "ok"]
    assert t["words"][0] == {"word": "Hallo", "start": 0.0, "end": 0.5, "probability": 0.98,
                             "seg": 0}
    assert t["words"][2]["seg"] == 1
    assert t["settings"]["vad_filter"] is True and t["settings"]["word_timestamps"] is True
    assert t["settings"]["condition_on_previous_text"] is False
    assert seen == [0.5, 1.0]


def test_build_transcript_handles_generator_and_no_words():
    t = build_transcript((s for s in [NS(start=0, end=1, text="x", words=None)]),
                         language="en", source="manual", probability=None, duration=1.0)
    assert t["words"] == [] and t["segments"][0]["avg_logprob"] is None


def test_empty_transcript():
    t = empty_transcript("de", "no-audio")
    assert t["words"] == [] and t["language"] == "de" and t["language_source"] == "no-audio"


# -- stubbed model ---------------------------------------------------------------------------
class FakeModel:
    def __init__(self, probs=None):
        self.probs = probs or [("nl", 0.6), ("de", 0.3), ("en", 0.1)]
        self.calls = []
        self.detect_calls = 0

    def detect_language(self, audio, **kw):
        self.detect_calls += 1
        return self.probs[0][0], self.probs[0][1], self.probs

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        segs = [_seg(0.0, 1.0, "Hallo", [(0.0, 1.0, "Hallo", 0.9)])]
        return iter(segs), NS(language=kw["language"], language_probability=0.8, duration=2.0)


def _cfg(tmp_path, **kw):
    return Config(data_dir=tmp_path, worker_enabled=False, nice_level=0, _env_file=None, **kw)


def _tr(tmp_path, model=None, **cfg):
    model = model or FakeModel()
    made = []

    def factory(name, compute, root, threads):
        made.append((name, compute, root, threads))
        return model

    tr = WhisperTranscriber(_cfg(tmp_path, **cfg), model_factory=factory,
                            audio_loader=lambda p: [0.0] * 32000)
    return tr, model, made


def test_lazy_load_and_whisper_arguments(tmp_path):
    tr, model, made = _tr(tmp_path)
    assert not tr.loaded and made == []  # nothing loaded at construction
    out = tr(tmp_path / "a.mp4", language=None, allowed=["de", "en"], prompt="Titel. Spiel: X")
    assert made == [("large-v3-turbo", "int8", None, 0)] and tr.loaded
    assert out["language"] == "de" and out["language_source"] == "auto"  # nl clamped -> de
    assert out["duration"] == 2.0
    kw = model.calls[0]
    assert kw["language"] == "de" and kw["initial_prompt"] == "Titel. Spiel: X"
    assert kw["vad_filter"] is True and kw["word_timestamps"] is True
    assert kw["condition_on_previous_text"] is False
    tr(tmp_path / "a.mp4", language=None, allowed=["de", "en"], prompt=None)
    assert tr.loads == 1  # model reused
    tr.unload()


def test_manual_language_skips_detection(tmp_path):
    tr, model, _ = _tr(tmp_path)
    out = tr(tmp_path / "a.mp4", language="en", allowed=["de", "en"], prompt=None)
    assert model.detect_calls == 0 and out["language"] == "en"
    assert out["language_source"] == "manual" and model.calls[0]["language"] == "en"
    tr.unload()


def test_single_allowed_language_skips_detection(tmp_path):
    tr, model, _ = _tr(tmp_path)
    out = tr(tmp_path / "a.mp4", language=None, allowed=["de"], prompt=None)
    assert model.detect_calls == 0 and out["language"] == "de"
    tr.unload()


def test_detection_failure_falls_back_to_first_allowed(tmp_path):
    model = FakeModel()

    def boom(*a, **k):
        raise RuntimeError("no speech")

    model.detect_language = boom
    tr, _, _ = _tr(tmp_path, model=model)
    out = tr(tmp_path / "a.mp4", language=None, allowed=["en", "de"], prompt=None)
    assert out["language"] == "en"
    tr.unload()


def test_idle_unload(tmp_path):
    tr, _, made = _tr(tmp_path, whisper_idle_minutes=10)
    tr(tmp_path / "a.mp4", language="de", allowed=["de"], prompt=None)
    assert tr.loaded and tr._timer is not None  # armed, fires after 10 idle minutes
    assert tr.unload() is True and not tr.loaded and tr._timer is None
    assert tr.unload() is False
    tr(tmp_path / "a.mp4", language="de", allowed=["de"], prompt=None)  # reload on demand
    assert len(made) == 2
    tr.unload()


def test_zero_idle_minutes_unloads_immediately(tmp_path):
    tr, _, made = _tr(tmp_path, whisper_idle_minutes=0)
    tr(tmp_path / "a.mp4", language="de", allowed=["de"], prompt=None)
    assert not tr.loaded and tr._timer is None
    tr(tmp_path / "a.mp4", language="de", allowed=["de"], prompt=None)
    assert len(made) == 2


def test_no_unload_while_transcribing(tmp_path):
    import threading

    tr, model, _ = _tr(tmp_path)
    inside, release = threading.Event(), threading.Event()

    def slow(audio, **kw):
        inside.set()
        release.wait(5)
        return iter([]), NS(language="de", language_probability=1.0, duration=1.0)

    model.transcribe = slow
    t = threading.Thread(target=lambda: tr(tmp_path / "a.mp4", language="de", allowed=["de"],
                                           prompt=None))
    t.start()
    assert inside.wait(5)
    assert tr.unload() is False and tr.loaded  # idle timer firing mid-job must not unload
    release.set()
    t.join()
    assert tr.loaded
    tr.unload()


# -- optional real model ----------------------------------------------------------------------
def test_real_whisper_on_silence(tmp_path):
    try:
        from faster_whisper import WhisperModel

        WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"whisper model not available offline: {type(e).__name__}")
    wav = tmp_path / "silence.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    cfg = _cfg(tmp_path, whisper_model="tiny", whisper_idle_minutes=0)
    out = WhisperTranscriber(cfg)(wav, language=None, allowed=["de", "en"], prompt=None)
    assert out["language"] in ("de", "en") and out["words"] == []
