"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    samples_dir: Path
    cors_origins: tuple[str, ...]
    cors_origin_regex: str | None
    max_upload_bytes: int
    rate_limit_per_minute: int


def load_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    max_upload_mb = float(os.getenv("MAX_UPLOAD_MB", "4"))
    rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    if max_upload_mb <= 0 or rate_limit <= 0:
        raise ValueError("MAX_UPLOAD_MB and RATE_LIMIT_PER_MINUTE must be positive.")
    return Settings(
        models_dir=Path(os.getenv("MODELS_DIR", BACKEND_DIR / "models")),
        samples_dir=Path(os.getenv("SAMPLES_DIR", BACKEND_DIR / "samples")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        cors_origin_regex=os.getenv("CORS_ORIGIN_REGEX") or None,
        max_upload_bytes=int(max_upload_mb * 1024 * 1024),
        rate_limit_per_minute=rate_limit,
    )
