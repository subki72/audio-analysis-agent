"""Configuration loading and subtitle timestamp formatting helpers."""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Read a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path: Path to ``settings.yaml``.

    Returns:
        Parsed configuration dictionary, or an empty dict if the file is missing.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
            return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        logger.warning("Configuration file not found at %s; using defaults.", config_path)
        return {}


def format_timestamp(seconds: float, is_vtt: bool = False) -> str:
    """Convert elapsed seconds to SRT (``HH:MM:SS,mmm``) or VTT (``HH:MM:SS.mmm``) format.

    Args:
        seconds: Non-negative timestamp in seconds.
        is_vtt: Use dot millisecond separator for WebVTT when True, comma for SRT otherwise.

    Returns:
        Zero-padded timestamp string.
    """
    safe_seconds = max(0.0, float(seconds))
    hours = int(safe_seconds // 3600)
    minutes = int((safe_seconds % 3600) // 60)
    secs = int(safe_seconds % 60)
    milliseconds = min(999, int(round((safe_seconds - int(safe_seconds)) * 1000)))

    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def _format_subtitle_blocks(segments: list[dict[str, Any]], *, is_vtt: bool) -> list[str]:
    """Render subtitle cue lines shared by SRT and WebVTT exporters."""
    lines: list[str] = ["WEBVTT", ""] if is_vtt else []
    for index, segment in enumerate(segments, 1):
        start_str = format_timestamp(segment.get("start", 0.0), is_vtt=is_vtt)
        end_str = format_timestamp(segment.get("end", 0.0), is_vtt=is_vtt)
        text = str(segment.get("text", "")).strip()
        lines.extend([str(index), f"{start_str} --> {end_str}", text, ""])
    return lines


def json_to_srt(segments: list[dict[str, Any]]) -> str:
    """Convert Whisper ``verbose_json`` segments into an SRT subtitle string."""
    return "\n".join(_format_subtitle_blocks(segments, is_vtt=False))


def json_to_vtt(segments: list[dict[str, Any]]) -> str:
    """Convert Whisper ``verbose_json`` segments into a WebVTT subtitle string."""
    return "\n".join(_format_subtitle_blocks(segments, is_vtt=True))
