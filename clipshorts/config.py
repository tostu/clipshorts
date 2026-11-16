"""Environment-driven configuration (env + .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")

    # Optional bearer token (primary protection = VPN/Tailscale)
    auth_token: str = ""

    # Comma-separated caption languages; auto-detect is clamped to these
    languages: str = "de,en"

    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    fps_cap: int = 30

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ytdlp_format: str = "best"

    # Worker runs at low priority so other workloads are not starved
    nice_level: int = 10
    worker_enabled: bool = True

    log_level: str = "INFO"

    @property
    def language_list(self) -> list[str]:
        return [x.strip().lower() for x in self.languages.split(",") if x.strip()]

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
