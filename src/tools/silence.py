"""Tool: detect_silence.

Detects silent segments in an audio file using FFmpeg's ``silencedetect`` filter.
Command: ``ffmpeg -i <file> -af silencedetect=noise=-50dB:d=2 -f null -``
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

from src.tools.ffmpeg_finder import find_ffmpeg
from src.tools.registry import register_tool

logger = logging.getLogger(__name__)

FFMPEG_SILENCE_TIMEOUT_SECONDS = 120

_CONFIG: dict[str, Any] = {
    "noise_threshold": "-50dB",
    "min_silence_duration": 2.0,
}


def configure(analysis_settings: dict[str, Any]) -> None:
    """Override default silence thresholds from the ``analysis`` block in settings.yaml.

    Args:
        analysis_settings: Configuration dictionary with optional ``noise_threshold``
            and ``min_silence_duration`` keys.
    """
    if "noise_threshold" in analysis_settings:
        _CONFIG["noise_threshold"] = str(analysis_settings["noise_threshold"])
    if "min_silence_duration" in analysis_settings:
        _CONFIG["min_silence_duration"] = float(analysis_settings["min_silence_duration"])


def _parse_audio_duration(stderr: str) -> float:
    """Extract total media duration in seconds from FFmpeg stderr header."""
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
    if not match:
        return 0.0
    hours, minutes, seconds, centiseconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centiseconds) / 100


def _parse_silence_segments(stderr: str) -> tuple[list[dict[str, float]], float]:
    """Parse ``silence_start`` / ``silence_end`` pairs from FFmpeg silencedetect output."""
    starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", stderr)

    segments: list[dict[str, float]] = []
    total_silence = 0.0
    for idx, (end_str, duration_str) in enumerate(ends):
        start = float(starts[idx]) if idx < len(starts) else 0.0
        duration = float(duration_str)
        segments.append(
            {
                "start": round(start, 3),
                "end": round(float(end_str), 3),
                "duration": round(duration, 3),
            }
        )
        total_silence += duration
    return segments, total_silence


def _run_silence_filter(file_path: str, threshold: str, min_duration: float) -> str:
    """Invoke FFmpeg silencedetect and return captured stderr output."""
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-i", file_path,
        "-af", f"silencedetect=noise={threshold}:d={min_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_SILENCE_TIMEOUT_SECONDS)
    return result.stderr


@register_tool
def detect_silence(
    file_path: str,
    noise_threshold: str | None = None,
    min_duration: float | None = None,
) -> dict[str, Any]:
    """Mendeteksi segmen keheningan dalam file audio menggunakan ffmpeg silencedetect.

    Args:
        file_path: Path ke file audio.
        noise_threshold: Threshold noise level (default: -50dB).
        min_duration: Durasi minimum keheningan yang dianggap signifikan (detik).

    Returns:
        Dictionary metrik silence_detected, silence_segments, total_silence_duration,
        silence_ratio, dan audio_duration.
    """
    if not os.path.exists(file_path):
        logger.warning("Silence detection requested for missing file: %s", file_path)
        return {"error": f"File tidak ditemukan: {file_path}"}

    threshold = noise_threshold if noise_threshold is not None else _CONFIG["noise_threshold"]
    duration = min_duration if min_duration is not None else _CONFIG["min_silence_duration"]

    try:
        stderr = _run_silence_filter(file_path, threshold, duration)
        audio_duration = _parse_audio_duration(stderr)
        segments, total_silence = _parse_silence_segments(stderr)
        silence_ratio = round(total_silence / audio_duration, 4) if audio_duration > 0 else 0.0
        logger.info("Detected %d silence segments in %s (ratio=%.4f)", len(segments), file_path, silence_ratio)
        return {
            "silence_detected": len(segments) > 0,
            "silence_segments": segments,
            "total_silence_duration": round(total_silence, 3),
            "silence_ratio": silence_ratio,
            "audio_duration": round(audio_duration, 3),
            "threshold_used": threshold,
            "min_duration_used": duration,
        }
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg silencedetect timed out after %ds for %s", FFMPEG_SILENCE_TIMEOUT_SECONDS, file_path)
        return {"error": "ffmpeg timeout setelah 120 detik."}
    except FileNotFoundError:
        logger.error("ffmpeg binary missing during silence detection for %s", file_path)
        return {"error": "ffmpeg tidak ditemukan. Pastikan ffmpeg terinstal."}
