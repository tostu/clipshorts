import pytest

from clipshorts.config import Config
from clipshorts.db import DB
from clipshorts.events import EventBus
from clipshorts.layout import Box, Face, FrameAnalysis
from clipshorts.media import MediaError, SampledFrame, Tools


@pytest.fixture
def cfg(tmp_path) -> Config:
    data = tmp_path / "data"
    data.mkdir()
    return Config(data_dir=data, worker_enabled=False, nice_level=0, _env_file=None)


@pytest.fixture
def db(cfg):
    d = DB(cfg.db_path)
    yield d
    d.close()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


class FakeTools:
    """Stubs for yt-dlp / ffprobe; records calls."""

    def __init__(self, fail_download: str | None = None, fail_probe: str | None = None):
        self.downloads: list[str] = []
        self.probes: list[str] = []
        self.extractions: list[list[float]] = []
        self.analyses: list[int] = []
        self.fail_download = fail_download
        self.fail_probe = fail_probe
        self.fail_analyze: str | None = None
        self.analysis_faces: list[Face] | None = None
        self.transcriptions: list[dict] = []
        self.fail_transcribe: str | None = None
        self.renders: list[dict] = []
        self.fail_render: str | None = None
        self.listings: list[str] = []
        self.fail_list: str | None = None
        self.channel_clips: dict[str, list[dict]] = {}  # channel -> listing items
        self.transcript: dict = {
            "version": 1, "language": "de", "language_source": "auto",
            "segments": [],
            "words": [
                {"word": "Das", "start": 0.0, "end": 0.3},
                {"word": "ist", "start": 0.3, "end": 0.5},
                {"word": "krass.", "start": 0.5, "end": 1.0},
                {"word": "Wirklich", "start": 1.2, "end": 1.6},
            ],
        }

    def download(self, url, dest, fmt="best", on_progress=None):
        self.downloads.append(url)
        if self.fail_download:
            raise MediaError(self.fail_download)
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        dest.write_bytes(b"fake-mp4")
        return {"title": "Big Play", "channel": "somestreamer", "game": "Tetris",
                "duration": 31.5, "id": "x"}

    def probe(self, path, ffprobe_bin="ffprobe"):
        self.probes.append(str(path))
        if self.fail_probe:
            raise MediaError(self.fail_probe)
        return {"duration": 30.0, "width": 1920, "height": 1080, "fps": 60.0,
                "video_codec": "h264", "audio_codec": "aac", "has_audio": True}

    def extract_frames(self, src, times, out_dir, ffmpeg_bin="ffmpeg", det_width=640,
                       on_progress=None):
        self.extractions.append(list(times))
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, t in enumerate(times):
            orig, det = out_dir / f"frame_{i:02d}.jpg", out_dir / f"det_{i:02d}.jpg"
            orig.write_bytes(b"jpg")
            det.write_bytes(b"jpg")
            frames.append(SampledFrame(i, t, orig, det))
        if on_progress:
            on_progress(1.0)
        return frames

    def analyze_frames(self, det_paths, model_path, score_threshold, on_progress=None):
        self.analyses.append(len(det_paths))
        if self.fail_analyze:
            raise MediaError(self.fail_analyze)
        n = len(det_paths)
        if self.analysis_faces is None:  # default: big centered face in every frame
            dets = [[Face(Box(0.35, 0.15, 0.3, 0.5), 0.9)] for _ in range(n)]
        else:
            dets = [list(self.analysis_faces) for _ in range(n)]
        return FrameAnalysis((640, 360), dets, None)

    def transcribe(self, audio, *, cfg=None, language=None, allowed=None, prompt=None,
                   on_progress=None):
        self.transcriptions.append({"audio": str(audio), "language": language,
                                    "allowed": allowed, "prompt": prompt})
        if on_progress:
            on_progress(1.0)
        if self.fail_transcribe:
            raise MediaError(self.fail_transcribe)
        return {**self.transcript, "language": language or self.transcript["language"]}

    def render(self, cmd, out_path, *, duration, timeout, on_progress=None, cancel=None,
               require_filters=(), require_encoders=()):
        self.renders.append({"cmd": list(cmd), "out": str(out_path), "duration": duration,
                             "timeout": timeout, "filters": list(require_filters)})
        if self.fail_render:
            raise MediaError(self.fail_render)
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        out_path.write_bytes(b"fake-output")

    def list_clips(self, channel, range_="24hr", limit=50):
        self.listings.append(channel)
        if self.fail_list:
            raise MediaError(self.fail_list)
        return [dict(c) for c in self.channel_clips.get(channel, [])]

    def as_tools(self) -> Tools:
        return Tools(download=self.download, probe=self.probe,
                     extract_frames=self.extract_frames, analyze_frames=self.analyze_frames,
                     transcribe=self.transcribe, render=self.render,
                     list_clips=self.list_clips)


@pytest.fixture
def fake_tools() -> FakeTools:
    return FakeTools()
