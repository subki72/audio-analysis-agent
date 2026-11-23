"""System readiness health probes, centralized error handling, and rate limiting."""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from src.config import AppConfig
from src.tools.ffmpeg_finder import find_ffmpeg, find_ffprobe

logger = logging.getLogger(__name__)

HEALTH_ENDPOINT = "/health"

T = TypeVar("T")


def _probe_binary(finder: Callable[[], str], binary_label: str) -> dict[str, Any]:
    """Verify availability of a required media binary without raising."""
    try:
        resolved_path = finder()
        return {"available": True, "path": resolved_path}
    except FileNotFoundError as exc:
        logger.warning("Health probe could not locate %s: %s", binary_label, exc)
        return {"available": False, "path": None, "detail": str(exc)}


def _probe_configuration(config: AppConfig | None) -> dict[str, Any]:
    """Inspect runtime configuration readiness without exposing credentials."""
    try:
        resolved = config if config is not None else AppConfig.from_env(load_env_file=False)
        resolved.validate(require_api_key=False)
        return {
            "valid": True,
            "api_key_configured": bool(resolved.groq_api_key),
            "model": resolved.groq_model,
            "max_iterations": resolved.max_iterations,
            "log_level": resolved.log_level,
        }
    except ValueError as exc:
        logger.warning("Configuration health probe failed: %s", exc)
        return {
            "valid": False,
            "api_key_configured": False,
            "detail": str(exc),
        }


def check_system_health(config: AppConfig | None = None) -> dict[str, Any]:
    """Return a structured readiness payload matching the ``/health`` service contract.

    Checks FFmpeg/FFprobe binary discovery and validated configuration readiness.

    Args:
        config: Optional pre-loaded ``AppConfig`` instance for deterministic testing.

    Returns:
        Dictionary with ``endpoint``, ``status``, ``timestamp``, ``python_version``,
        and detailed per-component ``checks``.
    """
    ffmpeg_check = _probe_binary(find_ffmpeg, "ffmpeg")
    ffprobe_check = _probe_binary(find_ffprobe, "ffprobe")
    config_check = _probe_configuration(config)

    is_healthy = (
        ffmpeg_check["available"]
        and ffprobe_check["available"]
        and config_check["valid"]
    )
    status = "healthy" if is_healthy else "degraded"
    logger.info("System health check completed with status=%s", status)

    return {
        "endpoint": HEALTH_ENDPOINT,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "checks": {
            "ffmpeg": ffmpeg_check,
            "ffprobe": ffprobe_check,
            "configuration": config_check,
        },
    }


class AudioAnalysisErrorHandler:
    """Centralized exception boundary converting failures into safe error payloads."""

    def handle_exception(self, operation: str, exc: Exception) -> dict[str, Any]:
        """Log an exception with context and produce a safe structured error response.

        Args:
            operation: Logical tool or workflow name where the error occurred.
            exc: Raised exception instance.

        Returns:
            Dictionary containing ``error``, ``error_type``, and ``operation``.
        """
        if isinstance(exc, FileNotFoundError):
            logger.warning("Operation '%s' file not found: %s", operation, exc)
            return {
                "error": str(exc),
                "error_type": "FileNotFoundError",
                "operation": operation,
            }

        if isinstance(exc, ValueError):
            logger.warning("Operation '%s' validation failed: %s", operation, exc)
            return {
                "error": str(exc),
                "error_type": "ValidationError",
                "operation": operation,
            }

        if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
            logger.error("Operation '%s' timed out: %s", operation, exc)
            return {
                "error": f"Operation '{operation}' timed out while processing audio.",
                "error_type": "TimeoutError",
                "operation": operation,
            }

        logger.exception("Unexpected error during operation '%s'", operation)
        return {
            "error": f"Internal error while executing '{operation}'. Check service logs for details.",
            "error_type": "InternalError",
            "operation": operation,
        }

    def execute(
        self,
        operation: str,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T | dict[str, Any]:
        """Run ``func`` inside the centralized error boundary.

        Args:
            operation: Descriptive operation identifier for logging.
            func: Callable to execute.
            *args: Positional arguments forwarded to ``func``.
            **kwargs: Keyword arguments forwarded to ``func``.

        Returns:
            Callable return value on success, or structured error dictionary on failure.
        """
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return self.handle_exception(operation, exc)


class SimpleRateLimiter:
    """Sliding-window rate limiter guarding outbound LLM API invocations."""

    def __init__(
        self,
        max_calls: int = 30,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._timestamps: deque[float] = deque()

    def acquire(self) -> float:
        """Record an outbound call, sleeping briefly if the sliding window is saturated.

        Returns:
            Total seconds slept (``0.0`` when quota was immediately available).
        """
        now = self._clock()
        self._evict_expired(now)

        slept = 0.0
        if len(self._timestamps) >= self.max_calls:
            oldest = self._timestamps[0]
            wait_time = max(0.0, self.window_seconds - (now - oldest))
            if wait_time > 0:
                logger.info("Rate limiter throttling outbound API call for %.2fs", wait_time)
                self._sleeper(wait_time)
                slept = wait_time
            now = self._clock()
            self._evict_expired(now)

        self._timestamps.append(now)
        return slept

    def _evict_expired(self, now: float) -> None:
        """Drop call timestamps older than the active sliding window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()
