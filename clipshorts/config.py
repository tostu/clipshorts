"""Environment-driven configuration (env + .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

YUNET_FILE = "face_detection_yunet_2023mar.onnx"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")

    # Optional bearer token (primary protection = VPN/Tailscale)
    auth_token: str = ""

    # Comma-separated caption languages; auto-detect is clamped to these
    languages: str = "de,en"

    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    # Model is unloaded after this many idle minutes (<= 0: right after each transcription)
    whisper_idle_minutes: float = 5.0
    whisper_download_root: Path | None = None
    whisper_threads: int = 0  # 0 = faster-whisper default

    # Captions (PlayRes is fixed at 1080x1920)
    caption_font_size: int = 84
    caption_max_words: int = 3
    fps_cap: int = 30

    # Render. video_encoder: libx264 (default) or h264_qsv (needs /dev/dri mounted; untested)
    video_encoder: str = "libx264"
    x264_preset: str = "veryfast"
    x264_crf: int = 20
    render_split_ratio: float = 0.38  # stacked layout: facecam share of the output height
    render_timeout_factor: float = 8.0  # wall-clock limit = factor x clip duration ...
    render_timeout_min_s: float = 120.0  # ... but never below this

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ytdlp_format: str = "best"

    # Layout detection. Empty yunet_model = the model vendored under clipshorts/models/.
    yunet_model: Path | None = None
    layout_frames: int = 8
    layout_det_width: int = 640
    layout_score_threshold: float = 0.7
    # JSON in env, e.g. LAYOUT_OVERRIDES='{"fullcam_min_area": 0.1}'; keys: LayoutParams fields
    layout_overrides: dict[str, float] = {}

    # Worker runs at low priority so other workloads are not starved
    nice_level: int = 10
    worker_enabled: bool = True

    log_level: str = "INFO"

    @property
    def language_list(self) -> list[str]:
        return [x.strip().lower() for x in self.languages.split(",") if x.strip()]

    @property
    def yunet_model_path(self) -> Path:
        return self.yunet_model or Path(__file__).parent / "models" / YUNET_FILE

    @property
    def fonts_dir(self) -> Path:
        """Bundled OFL font directory; pass to ffmpeg's ass filter as fontsdir."""
        return Path(__file__).parent / "fonts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipshorts.db"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    def clip_dir(self, clip_id: int) -> Path:
        return self.clips_dir / str(clip_id)


@lru_cache
def get_config() -> Config:
    return Config()
