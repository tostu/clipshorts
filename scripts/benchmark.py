#!/usr/bin/env python3
"""M0 benchmark: YuNet face detection, Whisper small int8, one ffmpeg 9:16 render.

Every section skips gracefully (prints SKIP) when its model/binary is missing.
Nothing is downloaded unless --download is passed.

    python scripts/benchmark.py [--yunet PATH] [--audio PATH] [--model small]
                                [--seconds 30] [--download]
"""

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def skip(name: str, why: str) -> None:
    print(f"[SKIP] {name}: {why}")


def bench_yunet(model: Path | None, frames: int = 20) -> None:
    name = "YuNet"
    if model is None or not model.is_file():
        return skip(name, "model file not found (pass --yunet face_detection_yunet_*.onnx)")
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        return skip(name, f"opencv/numpy missing ({e})")
    if not hasattr(cv2, "FaceDetectorYN"):
        return skip(name, "cv2.FaceDetectorYN unavailable")
    w, h = 640, 360
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.7)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    det.detect(img)  # warm-up
    times = []
    for _ in range(frames):
        t = time.perf_counter()
        det.detect(img)
        times.append((time.perf_counter() - t) * 1000)
    print(f"[ OK ] {name}: {statistics.median(times):.1f} ms/frame (median of {frames}, {w}x{h})")


def bench_whisper(model_name: str, audio: Path | None, seconds: int, download: bool) -> None:
    name = f"Whisper {model_name} int8"
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        return skip(name, f"faster-whisper missing ({e})")
    t0 = time.perf_counter()
    try:
        model = WhisperModel(
            model_name, device="cpu", compute_type="int8", local_files_only=not download
        )
    except Exception as e:  # noqa: BLE001 - model not cached / no network
        return skip(name, f"model not available locally ({type(e).__name__}); use --download")
    load_s = time.perf_counter() - t0
    tmp = None
    if audio is None:
        if not shutil.which("ffmpeg"):
            return skip(name, "no --audio given and ffmpeg missing to synthesize one")
        tmp = Path(tempfile.mkdtemp()) / "tone.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"sine=frequency=220:duration={seconds}", "-ar", "16000", "-ac", "1", str(tmp)],
            check=True,
        )
        audio = tmp
        print("       (no --audio: using a tone, speed is indicative only; pass real speech)")
    t = time.perf_counter()
    try:
        segs, info = model.transcribe(
            str(audio), vad_filter=True, word_timestamps=True, condition_on_previous_text=False
        )
        n = sum(1 for _ in segs)
    except Exception as e:  # noqa: BLE001 - e.g. incompatible av/ctranslate2 build
        return skip(name, f"transcribe failed ({type(e).__name__}: {e})")
    dt = time.perf_counter() - t
    print(
        f"[ OK ] {name}: load {load_s:.1f}s, transcribe {dt:.1f}s for "
        f"{info.duration:.0f}s audio (rtf {dt / max(info.duration, 1e-9):.2f}, {n} segments)"
    )


def bench_render(seconds: int) -> None:
    name = "ffmpeg render"
    if not shutil.which("ffmpeg"):
        return skip(name, "ffmpeg not found")
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "src.mp4", Path(d) / "out.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={seconds}",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
             str(src)],
            check=True,
        )
        vf = ("crop=ih*9/16:ih,scale=1080:1920,fps=30")
        t = time.perf_counter()
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
             "-vf", vf, "-af", "loudnorm", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
             "-movflags", "+faststart", str(out)],
        )
        dt = time.perf_counter() - t
        if r.returncode != 0:
            return skip(name, f"ffmpeg failed (rc={r.returncode})")
        print(f"[ OK ] {name}: {dt:.1f}s for {seconds}s clip (x{dt / seconds:.2f} realtime, "
              "libx264 veryfast crf20, loudnorm)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yunet", type=Path, default=None)
    ap.add_argument("--audio", type=Path, default=None)
    ap.add_argument("--model", default="small")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--download", action="store_true", help="allow fetching the Whisper model")
    a = ap.parse_args()
    bench_yunet(a.yunet)
    bench_whisper(a.model, a.audio, a.seconds, a.download)
    bench_render(a.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
