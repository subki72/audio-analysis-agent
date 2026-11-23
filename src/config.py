"""Centralized, validated application configuration and input path guards."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3"
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_API_TIMEOUT_SECONDS = 60.0
DEFAULT_RATE_LIMIT_PER_MINUTE = 30
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".wma"}
)


def normalize_groq_base_url(base_url: str | None) -> str | None:
    """Strip trailing ``/openai/v1`` suffix because the Groq SDK appends it automatically.

    Args:
        base_url: Optional custom Groq API base URL.

    Returns:
        Normalized base URL string, or ``None`` if unset/blank.
    """
    if not base_url:
        return None
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return None
    if cleaned.endswith("/openai/v1"):
        cleaned = cleaned[:-10]
    return cleaned or None


def configure_logging(level: str = "INFO") -> None:
    """Configure root structured leveled logging for CLI and MCP runtimes.

    Args:
        level: Standard Python logging level name (e.g. ``INFO``, ``DEBUG``).
    """
    normalized = (level or "INFO").strip().upper()
    resolved_level = getattr(logging, normalized, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)


def validate_audio_file_path(
    file_path: str,
    *,
    must_exist: bool = True,
    allowed_extensions: frozenset[str] | None = None,
) -> Path:
    """Validate an incoming audio file path against empty, null-byte, and existence issues.

    Args:
        file_path: Candidate file path supplied by CLI, MCP client, or LLM tool call.
        must_exist: When True, verify the target exists and is a regular file.
        allowed_extensions: Optional set of permitted lowercase extensions.

    Returns:
        Validated ``Path`` instance.

    Raises:
        ValueError: If the path is empty, contains null bytes, or has a disallowed extension.
        FileNotFoundError: If ``must_exist`` is True and the path is missing or a directory.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Audio file path must be a non-empty string.")
    if "\x00" in file_path:
        raise ValueError("Audio file path contains invalid null byte characters.")

    candidate = Path(file_path.strip())
    if allowed_extensions is not None and candidate.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported audio extension '{candidate.suffix}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    return candidate


class AppConfig(BaseModel):
    """Validated runtime configuration for Audio Analysis Agent and MCP server."""

    model_config = ConfigDict(extra="ignore")

    groq_api_key: str | None = Field(default=None, description="Groq API authentication key.")
    groq_model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1)
    transcription_model: str = Field(default=DEFAULT_TRANSCRIPTION_MODEL, min_length=1)
    groq_base_url: str | None = Field(default=None)
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=50)
    api_timeout_seconds: float = Field(default=DEFAULT_API_TIMEOUT_SECONDS, ge=5.0, le=300.0)
    rate_limit_calls_per_minute: int = Field(default=DEFAULT_RATE_LIMIT_PER_MINUTE, ge=1, le=600)
    ffmpeg_path: str | None = Field(default=None)
    ffprobe_path: str | None = Field(default=None)
    log_level: str = Field(default="INFO")
    noise_threshold: str = Field(default="-50dB", min_length=2)
    min_silence_duration: float = Field(default=2.0, gt=0.0, le=300.0)
    clipping_threshold_db: float = Field(default=-1.0, le=0.0, ge=-60.0)
    low_volume_threshold_db: float = Field(default=-35.0, le=0.0, ge=-100.0)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and verify logging level against standard Python levels."""
        normalized = value.strip().upper()
        if normalized not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL '{value}'. Must be one of {sorted(VALID_LOG_LEVELS)}."
            )
        return normalized

    @field_validator("groq_base_url", mode="before")
    @classmethod
    def _clean_base_url(cls, value: str | None) -> str | None:
        """Normalize Groq base URL to avoid duplicated API path segments."""
        return normalize_groq_base_url(value)

    @classmethod
    def from_env(cls, settings: dict[str, Any] | None = None, *, load_env_file: bool = True) -> AppConfig:
        """Build a validated ``AppConfig`` from environment variables and optional YAML settings.

        Args:
            settings: Optional dictionary loaded from ``config/settings.yaml``.
            load_env_file: Whether to invoke ``load_dotenv()`` before reading ``os.environ``.

        Returns:
            Validated ``AppConfig`` instance.
        """
        if load_env_file:
            load_dotenv()

        cfg = settings or {}
        api_cfg = cfg.get("api", {})
        analysis_cfg = cfg.get("analysis", {})

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL") or analysis_cfg.get("llm_model", DEFAULT_LLM_MODEL),
            transcription_model=api_cfg.get(
                "model", os.getenv("GROQ_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
            ),
            groq_base_url=os.getenv("GROQ_BASE_URL") or None,
            max_iterations=int(os.getenv("MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
            api_timeout_seconds=float(
                os.getenv("API_TIMEOUT_SECONDS", str(api_cfg.get("timeout", DEFAULT_API_TIMEOUT_SECONDS)))
            ),
            rate_limit_calls_per_minute=int(
                os.getenv("RATE_LIMIT_CALLS_PER_MINUTE", str(DEFAULT_RATE_LIMIT_PER_MINUTE))
            ),
            ffmpeg_path=os.getenv("FFMPEG_PATH") or None,
            ffprobe_path=os.getenv("FFPROBE_PATH") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            noise_threshold=str(analysis_cfg.get("noise_threshold", "-50dB")),
            min_silence_duration=float(analysis_cfg.get("min_silence_duration", 2.0)),
            clipping_threshold_db=float(analysis_cfg.get("clipping_threshold_db", -1.0)),
            low_volume_threshold_db=float(analysis_cfg.get("low_volume_threshold_db", -35.0)),
        )

    def validate(self, *, require_api_key: bool = True) -> None:
        """Perform runtime readiness validation without logging raw secret material.

        Args:
            require_api_key: Raise ``ValueError`` if ``groq_api_key`` is missing when True.

        Raises:
            ValueError: If required runtime settings are missing or paths are invalid.
        """
        if require_api_key and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")
        if self.ffmpeg_path and not Path(self.ffmpeg_path).is_file():
            raise ValueError(f"Configured FFMPEG_PATH does not exist: {self.ffmpeg_path}")
        if self.ffprobe_path and not Path(self.ffprobe_path).is_file():
            raise ValueError(f"Configured FFPROBE_PATH does not exist: {self.ffprobe_path}")
        logger.debug(
            "AppConfig validated (model=%s, max_iterations=%d, timeout=%.1fs, log_level=%s)",
            self.groq_model,
            self.max_iterations,
            self.api_timeout_seconds,
            self.log_level,
        )
