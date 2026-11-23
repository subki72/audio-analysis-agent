"""Tool: detect_noise.

Analyzes background noise floor and dynamic range using FFmpeg's ``astats`` filter.
Command: ``ffmpeg -i <file> -af astats=metadata=1:reset=1 -f null -``
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

FFMPEG_NOISE_TIMEOUT_SECONDS = 120
CLEAN_DYNAMIC_RANGE_DB = 20.0
MODERATE_DYNAMIC_RANGE_DB = 10.0


def _parse_astats_metrics(stderr: str) -> tuple[float | None, float | None, float | None]:
    """Parse RMS level, peak level, and crest factor from FFmpeg astats stderr output."""
    rms_match = re.search(r"RMS level dB:\s*([-\d.]+)", stderr)
    peak_match = re.search(r"Peak level dB:\s*([-\d.]+)", stderr)
    crest_match = re.search(r"Crest factor:\s*([-\d.]+)", stderr)

    rms_level = float(rms_match.group(1)) if rms_match else None
    peak_level = float(peak_match.group(1)) if peak_match else None
    crest_factor = float(crest_match.group(1)) if crest_match else None
    return rms_level, peak_level, crest_factor


def _assess_noise(dynamic_range: float | None) -> tuple[str, str]:
    """Classify background noise level and build a human-readable explanation."""
    if dynamic_range is None:
        return "unknown", "Tidak dapat menentukan — data metrik tidak lengkap."
    if dynamic_range > CLEAN_DYNAMIC_RANGE_DB:
        return (
            "clean",
            f"Dynamic range {dynamic_range} dB menunjukkan audio memiliki "
            "dinamika yang baik dengan noise floor yang rendah.",
        )
    if dynamic_range >= MODERATE_DYNAMIC_RANGE_DB:
        return (
            "moderate_noise",
            f"Dynamic range {dynamic_range} dB mengindikasikan adanya noise "
            "latar belakang, namun masih dalam batas yang dapat ditoleransi.",
        )
    return (
        "high_noise",
        f"Dynamic range {dynamic_range} dB sangat rendah — audio terdengar "
        "flat, kemungkinan besar terdapat noise latar belakang yang signifikan.",
    )


@register_tool
def detect_noise(file_path: str) -> dict[str, Any]:
    """Menganalisis noise level dan dynamic range file audio.

    Args:
        file_path: Path ke file audio.

    Returns:
        Dictionary dengan kunci: rms_level_db, peak_level_db, dynamic_range_db,
        crest_factor_db, noise_assessment, assessment_reason.
    """
    if not os.path.exists(file_path):
        logger.warning("Noise detection requested for missing file: %s", file_path)
        return {"error": f"File tidak ditemukan: {file_path}"}

    try:
        ffmpeg = find_ffmpeg()
        cmd = [ffmpeg, "-i", file_path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_NOISE_TIMEOUT_SECONDS,
        )
        rms_level, peak_level, crest_factor = _parse_astats_metrics(result.stderr)
        dynamic_range = (
            round(peak_level - rms_level, 2)
            if peak_level is not None and rms_level is not None
            else None
        )
        assessment, reason = _assess_noise(dynamic_range)

        logger.info("Assessed noise for %s (assessment=%s, dynamic_range=%s)", file_path, assessment, dynamic_range)
        return {
            "rms_level_db": round(rms_level, 2) if rms_level is not None else None,
            "peak_level_db": round(peak_level, 2) if peak_level is not None else None,
            "dynamic_range_db": dynamic_range,
            "crest_factor_db": round(crest_factor, 2) if crest_factor is not None else None,
            "noise_assessment": assessment,
            "assessment_reason": reason,
        }
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg astats timed out after %ds for %s", FFMPEG_NOISE_TIMEOUT_SECONDS, file_path)
        return {"error": "ffmpeg timeout setelah 120 detik."}
    except FileNotFoundError:
        logger.error("ffmpeg binary missing during noise analysis for %s", file_path)
        return {"error": "ffmpeg tidak ditemukan. Pastikan ffmpeg terinstal."}
