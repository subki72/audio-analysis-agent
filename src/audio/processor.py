"""Local audio processing utilities (metadata inspection, format conversion, splitting)."""

from __future__ import annotations

import logging
import os
from typing import Any

from pydub import AudioSegment

logger = logging.getLogger(__name__)

BYTES_PER_MEGABYTE = 1024 * 1024
DEFAULT_CHUNK_DURATION_SECONDS = 600


class AudioProcessor:
    """Handle local audio metadata inspection, conversion, and chunk splitting."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = settings or {}

    def get_metadata(self, file_path: str) -> dict[str, Any]:
        """Return basic local audio metadata including size, format, and duration.

        Args:
            file_path: Path to an existing audio file.

        Returns:
            Dictionary with keys ``path``, ``size_bytes``, ``duration_seconds``, and ``format``.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {file_path}")

        try:
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000.0
        except Exception as exc:
            logger.warning("Could not decode duration via pydub for %s: %s", file_path, exc)
            duration_seconds = 0.0

        return {
            "path": file_path,
            "size_bytes": os.path.getsize(file_path),
            "duration_seconds": duration_seconds,
            "format": os.path.splitext(file_path)[1][1:],
        }

    def convert_format(self, file_path: str, target_format: str) -> str:
        """Convert an audio file into ``target_format`` (e.g. ``mp3`` or ``wav``).

        Args:
            file_path: Source audio file path.
            target_format: Target audio format identifier without leading dot.

        Returns:
            Output path of the converted audio file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {file_path}")
        audio = AudioSegment.from_file(file_path)
        base, _ = os.path.splitext(file_path)
        output_path = f"{base}.{target_format}"
        audio.export(output_path, format=target_format)
        logger.info("Converted %s to %s", file_path, output_path)
        return output_path

    def split_audio(self, file_path: str, max_size_mb: int = 25) -> list[str]:
        """Split an audio file into fixed-duration chunks when it exceeds ``max_size_mb``.

        Args:
            file_path: Path to the source audio file.
            max_size_mb: Maximum allowed single-file upload size in megabytes.

        Returns:
            List of audio chunk file paths (or ``[file_path]`` if already within limit).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {file_path}")

        file_size_mb = os.path.getsize(file_path) / BYTES_PER_MEGABYTE
        if file_size_mb <= max_size_mb:
            return [file_path]

        audio_cfg = self.settings.get("audio", {})
        chunk_duration_sec = audio_cfg.get("chunk_duration_seconds", DEFAULT_CHUNK_DURATION_SECONDS)
        chunk_duration_ms = int(chunk_duration_sec * 1000)
        target_format = str(audio_cfg.get("target_format", "mp3"))

        audio = AudioSegment.from_file(file_path)
        base, _ = os.path.splitext(file_path)
        chunks: list[str] = []

        for offset in range(0, len(audio), chunk_duration_ms):
            part_num = (offset // chunk_duration_ms) + 1
            chunk_path = f"{base}_part{part_num}.{target_format}"
            audio[offset:offset + chunk_duration_ms].export(chunk_path, format=target_format)
            chunks.append(chunk_path)

        logger.info("Split %s into %d chunks (max_size_mb=%d)", file_path, len(chunks), max_size_mb)
        return chunks
