"""Integration tests for FFmpeg-based audio analysis tools.

Runs against the real sample fixture ``data/bad_audio.mp3`` when FFmpeg is
installed, and automatically skips FFmpeg-dependent cases otherwise.
"""

from __future__ import annotations

import os

import pytest

from src.tools.ffmpeg_finder import find_ffmpeg
from src.tools.metadata import get_audio_metadata
from src.tools.noise import detect_noise
from src.tools.silence import detect_silence
from src.tools.volume import detect_clipping


def _ffmpeg_available() -> bool:
    """Check whether FFmpeg is discoverable through the project binary finder."""
    try:
        find_ffmpeg()
        return True
    except FileNotFoundError:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg not installed",
)

AUDIO_FILE = os.path.join("data", "bad_audio.mp3")


@requires_ffmpeg
def test_metadata_returns_valid_data() -> None:
    """Verify ffprobe metadata extraction returns positive audio properties."""
    result = get_audio_metadata(AUDIO_FILE)
    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert result["duration_seconds"] > 0
    assert result["channels"] >= 1
    assert result["sample_rate"] > 0


def test_metadata_file_not_found() -> None:
    """Verify missing files produce a structured error dictionary."""
    result = get_audio_metadata("file_yang_tidak_ada.mp3")
    assert "error" in result


@requires_ffmpeg
def test_silence_detection_schema() -> None:
    """Verify silencedetect returns valid schema bounds and segment list."""
    result = detect_silence(AUDIO_FILE)
    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert isinstance(result["silence_detected"], bool)
    assert 0.0 <= result["silence_ratio"] <= 1.0
    assert isinstance(result["silence_segments"], list)


def test_silence_file_not_found() -> None:
    """Verify silence detection returns an error for missing files."""
    result = detect_silence("file_yang_tidak_ada.mp3")
    assert "error" in result


@requires_ffmpeg
def test_clipping_detection_schema() -> None:
    """Verify volumedetect returns valid clipping flags and severity enum."""
    result = detect_clipping(AUDIO_FILE)
    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert isinstance(result["volume_clipping"], bool)
    assert result["clipping_severity"] in ("none", "warning", "moderate", "severe")
    assert "mean_volume_db" in result


@requires_ffmpeg
def test_noise_detection_schema() -> None:
    """Verify astats noise analysis returns a valid assessment and dynamic range."""
    result = detect_noise(AUDIO_FILE)
    assert "error" not in result, f"Tool returned error: {result.get('error')}"
    assert result["noise_assessment"] in ("clean", "moderate_noise", "high_noise", "unknown")
    assert "dynamic_range_db" in result
    if result["dynamic_range_db"] is not None:
        assert result["dynamic_range_db"] > 0
