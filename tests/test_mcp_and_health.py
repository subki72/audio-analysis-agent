"""Offline tests for AppConfig, system health probes, error handler, rate limiter, and MCP tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import mcp_server
from src.config import (
    AppConfig,
    SUPPORTED_AUDIO_EXTENSIONS,
    normalize_groq_base_url,
    validate_audio_file_path,
)
from src.health import (
    HEALTH_ENDPOINT,
    AudioAnalysisErrorHandler,
    SimpleRateLimiter,
    check_system_health,
)


def test_app_config_validates_env_and_defaults() -> None:
    """AppConfig reads environment variables, normalizes base URL, and passes validation."""
    env = {
        "GROQ_API_KEY": "test_token_placeholder",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "GROQ_BASE_URL": "https://api.groq.com/openai/v1/",
        "MAX_ITERATIONS": "12",
        "API_TIMEOUT_SECONDS": "45",
        "LOG_LEVEL": "debug",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = AppConfig.from_env(load_env_file=False)
        cfg.validate(require_api_key=True)

    assert cfg.groq_base_url == "https://api.groq.com"
    assert cfg.max_iterations == 12
    assert cfg.api_timeout_seconds == 45.0
    assert cfg.log_level == "DEBUG"


def test_app_config_rejects_invalid_values() -> None:
    """AppConfig raises explicit validation errors on invalid log levels or missing binary paths."""
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        AppConfig(log_level="VERBOSE")

    bad_ffmpeg = AppConfig(groq_api_key="test_token_placeholder", ffmpeg_path="/nonexistent/ffmpeg")
    with pytest.raises(ValueError, match="FFMPEG_PATH"):
        bad_ffmpeg.validate()

    assert normalize_groq_base_url("   ") is None
    assert normalize_groq_base_url("https://api.groq.com/openai/v1") == "https://api.groq.com"


def test_validate_audio_file_path_guards(tmp_path: Path) -> None:
    """Path validator rejects empty strings, null bytes, unsupported extensions, and missing files."""
    with pytest.raises(ValueError, match="non-empty"):
        validate_audio_file_path("   ")
    with pytest.raises(ValueError, match="null byte"):
        validate_audio_file_path("bad\x00.mp3")

    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        validate_audio_file_path(str(txt_file), allowed_extensions=SUPPORTED_AUDIO_EXTENSIONS)

    with pytest.raises(FileNotFoundError):
        validate_audio_file_path(str(tmp_path / "missing.mp3"), must_exist=True)

    mp3_file = tmp_path / "valid.mp3"
    mp3_file.write_bytes(b"mp3")
    assert validate_audio_file_path(str(mp3_file), allowed_extensions=SUPPORTED_AUDIO_EXTENSIONS) == mp3_file


def test_check_system_health_healthy_and_degraded() -> None:
    """Health check reports healthy when binaries & config are valid and degraded otherwise."""
    cfg = AppConfig(groq_api_key="test_token_placeholder")
    with (
        patch("src.health.find_ffmpeg", return_value="/usr/bin/ffmpeg"),
        patch("src.health.find_ffprobe", return_value="/usr/bin/ffprobe"),
    ):
        healthy = check_system_health(config=cfg)

    assert healthy["endpoint"] == HEALTH_ENDPOINT
    assert healthy["status"] == "healthy"
    assert healthy["checks"]["ffmpeg"]["available"] is True
    assert healthy["checks"]["configuration"]["api_key_configured"] is True

    with (
        patch("src.health.find_ffmpeg", side_effect=FileNotFoundError("ffmpeg missing")),
        patch("src.health.find_ffprobe", return_value="/usr/bin/ffprobe"),
    ):
        degraded = check_system_health(config=cfg)

    assert degraded["status"] == "degraded"
    assert degraded["checks"]["ffmpeg"]["available"] is False


def test_error_handler_maps_exceptions_safely() -> None:
    """AudioAnalysisErrorHandler maps known exceptions and redacts unexpected internal errors."""
    handler = AudioAnalysisErrorHandler()

    missing = handler.execute("meta", lambda: (_ for _ in ()).throw(FileNotFoundError("not found")))
    assert missing["error_type"] == "FileNotFoundError"

    invalid = handler.execute("validate", lambda: (_ for _ in ()).throw(ValueError("bad input")))
    assert invalid["error_type"] == "ValidationError"

    timed_out = handler.execute("ffmpeg", lambda: (_ for _ in ()).throw(TimeoutError("too slow")))
    assert timed_out["error_type"] == "TimeoutError"

    internal = handler.execute("secret_op", lambda: (_ for _ in ()).throw(RuntimeError("secret stack trace")))
    assert internal["error_type"] == "InternalError"
    assert "secret stack trace" not in internal["error"]


def test_simple_rate_limiter_throttles_when_saturated() -> None:
    """Rate limiter enforces sliding window limits using injected clock and sleeper."""
    current_time = [100.0]
    slept_calls: list[float] = []

    def fake_clock() -> float:
        return current_time[0]

    def fake_sleep(seconds: float) -> None:
        slept_calls.append(seconds)
        current_time[0] += seconds

    limiter = SimpleRateLimiter(max_calls=2, window_seconds=10.0, clock=fake_clock, sleeper=fake_sleep)
    assert limiter.acquire() == 0.0
    current_time[0] = 102.0
    assert limiter.acquire() == 0.0
    current_time[0] = 103.0
    waited = limiter.acquire()

    assert waited == pytest.approx(7.0)
    assert slept_calls == [pytest.approx(7.0)]


def test_mcp_tools_delegate_and_validate(tmp_path: Path) -> None:
    """All 5 FastMCP tools validate inputs, delegate cleanly, and expose health status."""
    sample = tmp_path / "audio.mp3"
    sample.write_bytes(b"data")

    with (
        patch("mcp_server.get_audio_metadata", return_value={"filename": "audio.mp3"}),
        patch("mcp_server.detect_silence", return_value={"silence_detected": False}),
        patch("mcp_server.detect_clipping", return_value={"volume_clipping": False}),
        patch("mcp_server.detect_noise", return_value={"noise_assessment": "clean"}),
        patch("mcp_server.check_system_health", return_value={"endpoint": "/health", "status": "healthy"}),
    ):
        assert mcp_server.get_audio_metadata_tool(str(sample))["filename"] == "audio.mp3"
        assert mcp_server.detect_silence_tool(str(sample))["silence_detected"] is False
        assert mcp_server.detect_clipping_tool(str(sample))["volume_clipping"] is False
        assert mcp_server.detect_noise_tool(str(sample))["noise_assessment"] == "clean"
        assert mcp_server.health_check_tool()["status"] == "healthy"

    missing_res = mcp_server.get_audio_metadata_tool(str(tmp_path / "missing.mp3"))
    assert missing_res["error_type"] == "FileNotFoundError"
