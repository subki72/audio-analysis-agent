"""Groq Whisper speech-to-text transcription client wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from src.config import DEFAULT_API_TIMEOUT_SECONDS, normalize_groq_base_url
from src.utils.helpers import json_to_srt, json_to_vtt

load_dotenv()

logger = logging.getLogger(__name__)


def _resolve_api_format(response_format: str) -> str:
    """Map requested output subtitle/text format to the corresponding Groq API format."""
    if response_format == "json":
        return "json"
    if response_format in ("srt", "vtt"):
        return "verbose_json"
    return "text"


def _format_transcription_response(transcription: Any, response_format: str) -> str:
    """Convert raw Groq transcription payload into requested text/subtitle representation."""
    if response_format in ("srt", "vtt"):
        segments = getattr(transcription, "segments", [])
        if not segments and isinstance(transcription, dict):
            segments = transcription.get("segments", [])
        return json_to_srt(segments) if response_format == "srt" else json_to_vtt(segments)

    if isinstance(transcription, str):
        return transcription
    return str(getattr(transcription, "text", str(transcription)))


class Transcriber:
    """Manage audio transcription requests against the Groq Whisper API."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        api_settings = self.settings.get("api", {})

        self.api_key: str | None = os.getenv("GROQ_API_KEY")
        self.timeout: float = float(api_settings.get("timeout", DEFAULT_API_TIMEOUT_SECONDS))
        self.model: str = str(api_settings.get("model", os.getenv("GROQ_MODEL", "whisper-large-v3")))
        base_url = normalize_groq_base_url(os.getenv("GROQ_BASE_URL"))

        self.client: Groq | None = None
        if self.api_key:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = Groq(**client_kwargs)

    def transcribe_file(
        self,
        file_path: str,
        language: str = "id",
        response_format: str = "txt",
    ) -> str:
        """Send an audio file to the Groq Whisper API and return formatted text.

        Args:
            file_path: Path to the local audio file.
            language: ISO-639-1 language code (e.g. ``id``, ``en``).
            response_format: Output format (``txt``, ``srt``, ``vtt``, or ``json``).

        Returns:
            Formatted transcript string.

        Raises:
            ValueError: If no Groq API key is configured.
            FileNotFoundError: If ``file_path`` does not exist.
        """
        if not self.client:
            raise ValueError(
                "API Key untuk GROQ tidak ditemukan. "
                "Pastikan Anda telah mengonfigurasinya di file .env."
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {file_path}")

        api_format = _resolve_api_format(response_format)
        logger.info(
            "Submitting %s to Groq Whisper (model=%s, lang=%s, format=%s)",
            file_path,
            self.model,
            language,
            response_format,
        )
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model=self.model,
                language=language,
                response_format=api_format,
            )
        return _format_transcription_response(transcription, response_format)

    def save_output(self, content: str, output_path: str) -> None:
        """Persist transcript text to disk, creating parent directories if needed."""
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(content)
        logger.info("Saved transcript output to %s", output_path)
