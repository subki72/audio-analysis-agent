"""Pytest session configuration for VoiceScript test suite."""

from __future__ import annotations

from typing import Any
import warnings


def pytest_configure(config: Any) -> None:
    """Suppress noisy pydub RuntimeWarnings when FFmpeg is absent from PATH."""
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffprobe or avprobe",
        category=RuntimeWarning,
    )
