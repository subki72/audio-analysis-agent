"""Tool: detect_clipping.

Measures volume levels and detects audio clipping via FFmpeg's ``volumedetect`` filter.
Command: ``ffmpeg -i <file> -af volumedetect -f null -``
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

FFMPEG_VOLUME_TIMEOUT_SECONDS = 120
CLIPPING_WARNING_MARGIN_DB = 2.0

_CONFIG: dict[str, float] = {
    "clipping_threshold_db": -1.0,
    "low_volume_threshold_db": -35.0,
}


def configure(analysis_settings: dict[str, Any]) -> None:
    """Override default volume thresholds from the ``analysis`` block in settings.yaml.

    Args:
        analysis_settings: Configuration dictionary with optional
            ``clipping_threshold_db`` and ``low_volume_threshold_db`` keys.
    """
    if "clipping_threshold_db" in analysis_settings:
        _CONFIG["clipping_threshold_db"] = float(analysis_settings["clipping_threshold_db"])
    if "low_volume_threshold_db" in analysis_settings:
        _CONFIG["low_volume_threshold_db"] = float(analysis_settings["low_volume_threshold_db"])


def _classify_clipping(max_volume: float | None, threshold_db: float) -> tuple[bool, str]:
    """Classify clipping status and severity from peak volume in dBFS."""
    if max_volume is None:
        return False, "none"
    if max_volume >= 0.0:
        return True, "severe"
    if max_volume >= threshold_db:
        return True, "moderate"
    if max_volume >= threshold_db - CLIPPING_WARNING_MARGIN_DB:
        return False, "warning"
    return False, "none"


def _parse_volume_metrics(stderr: str) -> tuple[float | None, float | None, dict[str, int]]:
    """Extract mean volume, max volume, and dB histogram from FFmpeg volumedetect output."""
    mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
    max_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)

    mean_volume = float(mean_match.group(1)) if mean_match else None
    max_volume = float(max_match.group(1)) if max_match else None

    histogram: dict[str, int] = {}
    for db_val, count in re.findall(r"histogram_(\d+)db:\s*(\d+)", stderr):
        histogram[f"-{db_val}dB"] = int(count)

    return mean_volume, max_volume, histogram


def _run_volume_filter(file_path: str) -> str:
    """Invoke FFmpeg volumedetect and return captured stderr output."""
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-i", file_path, "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_VOLUME_TIMEOUT_SECONDS)
    return result.stderr


@register_tool
def detect_clipping(
    file_path: str,
    clipping_threshold_db: float | None = None,
    low_volume_threshold_db: float | None = None,
) -> dict[str, Any]:
    """Mengukur level volume dan mendeteksi clipping pada file audio.

    Args:
        file_path: Path ke file audio.
        clipping_threshold_db: Max volume >= ini dianggap clipping (default: -1.0 dB).
        low_volume_threshold_db: Mean volume < ini dianggap terlalu rendah (default: -35.0 dB).

    Returns:
        Dictionary metrik volume, status clipping, severity, dan histogram.
    """
    if not os.path.exists(file_path):
        logger.warning("Clipping detection requested for missing file: %s", file_path)
        return {"error": f"File tidak ditemukan: {file_path}"}

    clip_threshold = clipping_threshold_db if clipping_threshold_db is not None else _CONFIG["clipping_threshold_db"]
    low_threshold = (
        low_volume_threshold_db if low_volume_threshold_db is not None else _CONFIG["low_volume_threshold_db"]
    )

    try:
        stderr = _run_volume_filter(file_path)
        mean_volume, max_volume, histogram = _parse_volume_metrics(stderr)
        clipping, severity = _classify_clipping(max_volume, clip_threshold)
        low_volume = mean_volume is not None and mean_volume < low_threshold
        logger.info("Analyzed volume for %s (severity=%s, low_volume=%s)", file_path, severity, low_volume)
        return {
            "mean_volume_db": round(mean_volume, 2) if mean_volume is not None else None,
            "max_volume_db": round(max_volume, 2) if max_volume is not None else None,
            "volume_clipping": clipping,
            "clipping_severity": severity,
            "low_volume_detected": low_volume,
            "histogram": histogram,
        }
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg volumedetect timed out after %ds for %s", FFMPEG_VOLUME_TIMEOUT_SECONDS, file_path)
        return {"error": "ffmpeg timeout setelah 120 detik."}
    except FileNotFoundError:
        logger.error("ffmpeg binary missing during volume analysis for %s", file_path)
        return {"error": "ffmpeg tidak ditemukan. Pastikan ffmpeg terinstal."}
