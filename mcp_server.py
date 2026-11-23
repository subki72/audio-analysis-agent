"""MCP Server for Audio Analysis Agent.

Exposes 4 FFmpeg audio analysis tools and a system health probe via FastMCP
(Model Context Protocol).

Run:
    python mcp_server.py
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import AppConfig, configure_logging, validate_audio_file_path
from src.health import AudioAnalysisErrorHandler, check_system_health
from src.tools.metadata import get_audio_metadata
from src.tools.noise import detect_noise
from src.tools.silence import detect_silence
from src.tools.volume import detect_clipping

logger = logging.getLogger(__name__)
error_handler = AudioAnalysisErrorHandler()

mcp = FastMCP(
    "Audio Analysis Agent",
    instructions=(
        "MCP server yang menyediakan tools analisis kualitas audio "
        "menggunakan ffmpeg/ffprobe serta readiness health check."
    ),
)


def _run_audio_tool(operation: str, tool_fn: Any, file_path: str) -> dict[str, Any]:
    """Validate audio path and invoke an FFmpeg tool inside the centralized error boundary."""
    logger.info("MCP tool invoked: %s (file=%s)", operation, file_path)
    try:
        validated = validate_audio_file_path(file_path, must_exist=True)
    except (ValueError, FileNotFoundError) as exc:
        return error_handler.handle_exception(operation, exc)
    return error_handler.execute(operation, tool_fn, str(validated))


@mcp.tool()
def get_audio_metadata_tool(file_path: str) -> dict[str, Any]:
    """Mengambil metadata dari file audio menggunakan ffprobe.

    Mengembalikan informasi format, durasi, sample rate, channels, codec, dan bitrate.

    Args:
        file_path: Path absolut atau relatif ke file audio yang akan dianalisis.

    Returns:
        Dictionary metadata audio atau payload error terstruktur.
    """
    return _run_audio_tool("get_audio_metadata", get_audio_metadata, file_path)


@mcp.tool()
def detect_silence_tool(file_path: str) -> dict[str, Any]:
    """Mendeteksi segmen keheningan (silence) dalam file audio menggunakan ffmpeg silencedetect.

    Mengembalikan daftar segmen hening beserta durasi totalnya dan silence ratio.

    Args:
        file_path: Path absolut atau relatif ke file audio yang akan dianalisis.

    Returns:
        Dictionary hasil deteksi silence atau payload error terstruktur.
    """
    return _run_audio_tool("detect_silence", detect_silence, file_path)


@mcp.tool()
def detect_clipping_tool(file_path: str) -> dict[str, Any]:
    """Mengukur level volume dan mendeteksi clipping pada file audio menggunakan ffmpeg volumedetect.

    Mengembalikan mean volume, max volume, status clipping, dan severity level.

    Args:
        file_path: Path absolut atau relatif ke file audio yang akan dianalisis.

    Returns:
        Dictionary metrik volume/clipping atau payload error terstruktur.
    """
    return _run_audio_tool("detect_clipping", detect_clipping, file_path)


@mcp.tool()
def detect_noise_tool(file_path: str) -> dict[str, Any]:
    """Menganalisis noise level dan dynamic range audio menggunakan ffmpeg astats.

    Mengembalikan RMS level, peak level, dynamic range, crest factor,
    dan penilaian noise secara keseluruhan.

    Args:
        file_path: Path absolut atau relatif ke file audio yang akan dianalisis.

    Returns:
        Dictionary penilaian noise atau payload error terstruktur.
    """
    return _run_audio_tool("detect_noise", detect_noise, file_path)


@mcp.tool()
def health_check_tool() -> dict[str, Any]:
    """Memeriksa kesiapan layanan Audio Analysis Agent (FFmpeg, FFprobe, dan konfigurasi).

    Returns:
        Dictionary kontrak ``/health`` berisi status kesiapan tiap komponen.
    """
    logger.info("MCP tool invoked: health_check")
    return error_handler.execute("health_check", check_system_health)


def main() -> None:
    """Configure structured logging and start the FastMCP server."""
    config = AppConfig.from_env()
    configure_logging(config.log_level)
    logger.info("Starting Audio Analysis Agent MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
