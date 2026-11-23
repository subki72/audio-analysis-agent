"""Tests for local AudioProcessor metadata extraction."""

from __future__ import annotations

from src.audio.processor import AudioProcessor


def test_audio_metadata() -> None:
    """Validate basic local audio metadata fields on sample MP3 fixture."""
    processor = AudioProcessor()
    meta = processor.get_metadata("data/bad_audio.mp3")
    assert meta["path"] == "data/bad_audio.mp3"
    assert "size_bytes" in meta
    assert meta["size_bytes"] > 0
    assert meta["format"] == "mp3"
