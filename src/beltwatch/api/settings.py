"""Runtime settings, read from ``BELTWATCH_*`` environment variables."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: Path = Path("var")
    releases_dir: Path = Path("model_releases")
    max_files_per_audit: int = Field(default=20, ge=1)
    max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=40_000_000, ge=1)
    max_pending_audits: int = Field(default=50, ge=1)
    lease_seconds: float = Field(default=120.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    job_deadline_seconds: float = Field(default=600.0, gt=0)
    retention_hours: float = Field(default=24.0, gt=0)
    worker_stale_seconds: float = Field(default=30.0, gt=0)
    require_worker: bool = True
    metrics_token: str | None = None
    device: str = "cpu"
    torch_threads: int | None = None
    frontend_dir: Path | None = None
    """Static review UI; defaults to the repository's ``frontend/`` directory."""

    @property
    def database_path(self) -> Path:
        return self.data_dir / "beltwatch.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        values: dict[str, object] = {}
        for name in cls.model_fields:
            raw = os.environ.get(f"BELTWATCH_{name.upper()}")
            if raw is not None:
                values[name] = raw
        return cls.model_validate(values)
