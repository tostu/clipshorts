import pytest

from clipshorts.config import Config
from clipshorts.db import DB
from clipshorts.events import EventBus
from clipshorts.media import MediaError, Tools


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
        self.fail_download = fail_download
        self.fail_probe = fail_probe

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

    def as_tools(self) -> Tools:
        return Tools(download=self.download, probe=self.probe)


@pytest.fixture
def fake_tools() -> FakeTools:
    return FakeTools()
