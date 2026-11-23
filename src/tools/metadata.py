"""Tool: get_audio_metadata.

Extracts technical audio metadata using ``ffprobe``.
Command: ``ffprobe -v quiet -print_format json -show_format -show_streams <file>``
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from src.tools.ffmpeg_finder import find_ffprobe
from src.tools.registry import register_tool

logger = logging.getLogger(__name__)

FFPROBE_TIMEOUT_SECONDS = 30


def _find_audio_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first audio stream dictionary from ffprobe output, if present."""
    for stream in streams:
        if stream.get("codec_type") == "audio":
            return stream
    return None


def _build_metadata_payload(file_path: str, probe_data: dict[str, Any]) -> dict[str, Any]:
    """Transform raw ffprobe JSON into the standardized metadata dictionary."""
    fmt = probe_data.get("format", {})
    audio_stream = _find_audio_stream(probe_data.get("streams", []))

    metadata: dict[str, Any] = {
        "filename": os.path.basename(file_path),
        "format": fmt.get("format_name", "unknown"),
        "format_long_name": fmt.get("format_long_name", "unknown"),
        "size_bytes": int(fmt.get("size", 0)),
        "duration_seconds": float(fmt.get("duration", 0.0)),
        "bitrate": int(fmt.get("bit_rate", 0)),
    }

    if audio_stream:
        metadata.update(
            {
                "codec": audio_stream.get("codec_name", "unknown"),
                "codec_long_name": audio_stream.get("codec_long_name", "unknown"),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "channel_layout": audio_stream.get("channel_layout", "unknown"),
            }
        )
    else:
        metadata.update({"codec": "unknown", "sample_rate": 0, "channels": 0})

    return metadata


@register_tool
def get_audio_metadata(file_path: str) -> dict[str, Any]:
    """Mengambil metadata dari file audio menggunakan ffprobe.

    Args:
        file_path: Path ke file audio.

    Returns:
        Dictionary dengan kunci: filename, format, size_bytes, duration_seconds,
        sample_rate, channels, codec, bitrate.
    """
    if not os.path.exists(file_path):
        logger.warning("Audio metadata requested for missing file: %s", file_path)
        return {"error": f"File tidak ditemukan: {file_path}"}

    try:
        ffprobe = find_ffprobe()
        cmd = [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            logger.error("ffprobe failed for %s: %s", file_path, result.stderr.strip())
            return {"error": f"ffprobe gagal: {result.stderr.strip()}"}

        probe_data = json.loads(result.stdout)
        metadata = _build_metadata_payload(file_path, probe_data)
        logger.info("Extracted metadata for %s (duration=%.2fs)", file_path, metadata["duration_seconds"])
        return metadata
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out after %ds for %s", FFPROBE_TIMEOUT_SECONDS, file_path)
        return {"error": "ffprobe timeout setelah 30 detik."}
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse ffprobe JSON for %s: %s", file_path, exc)
        return {"error": f"Gagal parsing output ffprobe: {exc}"}
    except FileNotFoundError:
        logger.error("ffprobe executable not found while inspecting %s", file_path)
        return {"error": "ffprobe tidak ditemukan. Pastikan ffmpeg terinstal."}
