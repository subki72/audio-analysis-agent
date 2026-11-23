"""Unit tests for Transcriber initialization and model configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from src.transcription.transcriber import Transcriber


def test_transcriber_no_api_key() -> None:
    """Transcriber initializes with client=None when GROQ_API_KEY is absent."""
    with patch.dict("os.environ", {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        transcriber = Transcriber(settings={})
        assert transcriber.api_key is None
        assert transcriber.client is None


def test_transcriber_with_api_key() -> None:
    """Transcriber builds a Groq client when GROQ_API_KEY is set."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_token_placeholder"}):
        transcriber = Transcriber(settings={})
        assert transcriber.api_key == "test_token_placeholder"
        assert transcriber.client is not None


def test_transcriber_default_model() -> None:
    """Transcriber defaults to whisper-large-v3 when no model override is set."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_token_placeholder"}, clear=True):
        transcriber = Transcriber(settings={})
        assert transcriber.model == "whisper-large-v3"


def test_transcriber_model_from_settings() -> None:
    """Transcriber prioritizes model configured in settings['api']['model']."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_token_placeholder"}):
        settings = {"api": {"model": "whisper-large-v3-turbo"}}
        transcriber = Transcriber(settings=settings)
        assert transcriber.model == "whisper-large-v3-turbo"
