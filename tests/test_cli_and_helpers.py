"""Offline unit & CLI integration tests for helpers, AudioProcessor, registry, Transcriber, and Click CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.audio.processor import AudioProcessor
from src.main import cli
from src.tools.registry import _resolve_json_type
from src.transcription.transcriber import Transcriber
from src.utils.helpers import format_timestamp, json_to_srt, json_to_vtt, load_config


def test_helpers_config_and_subtitles(tmp_path: Path) -> None:
    """Helpers load YAML safely, fallback on missing files, and export SRT/VTT accurately."""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("api:\n  model: whisper-large-v3\n", encoding="utf-8")
    assert load_config(str(cfg_file))["api"]["model"] == "whisper-large-v3"
    assert load_config(str(tmp_path / "missing.yaml")) == {}

    assert format_timestamp(3661.25, is_vtt=False) == "01:01:01,250"
    assert format_timestamp(3661.25, is_vtt=True) == "01:01:01.250"

    segments = [{"start": 0.0, "end": 2.5, "text": "Halo dunia"}]
    srt = json_to_srt(segments)
    vtt = json_to_vtt(segments)
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in vtt


def test_audio_processor_split_and_errors(tmp_path: Path) -> None:
    """AudioProcessor returns original path when under limit and raises on missing file."""
    small_file = tmp_path / "small.mp3"
    small_file.write_bytes(b"0" * 1024)
    processor = AudioProcessor()

    assert processor.split_audio(str(small_file), max_size_mb=25) == [str(small_file)]
    with pytest.raises(FileNotFoundError):
        processor.get_metadata(str(tmp_path / "missing.mp3"))


def test_registry_resolves_optional_union_types() -> None:
    """Registry maps Python primitives and Optional unions to JSON Schema types."""
    assert _resolve_json_type(str) == "string"
    assert _resolve_json_type(float | None) == "number"
    assert _resolve_json_type(int | None) == "integer"
    assert _resolve_json_type(bool) == "boolean"


def test_transcriber_transcribe_formats_and_save(tmp_path: Path) -> None:
    """Transcriber sends audio to mocked Groq client and converts txt/srt/vtt outputs."""
    audio_file = tmp_path / "voice.mp3"
    audio_file.write_bytes(b"fake-mp3")

    with patch.dict("os.environ", {"GROQ_API_KEY": "test_token_placeholder"}):
        transcriber = Transcriber(settings={})

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="Transkrip contoh",
        segments=[{"start": 0.0, "end": 1.5, "text": "Transkrip contoh"}],
    )
    transcriber.client = mock_client

    assert transcriber.transcribe_file(str(audio_file), response_format="txt") == "Transkrip contoh"
    assert "-->" in transcriber.transcribe_file(str(audio_file), response_format="srt")
    assert transcriber.transcribe_file(str(audio_file), response_format="vtt").startswith("WEBVTT")

    out_path = tmp_path / "out" / "transcript.txt"
    transcriber.save_output("Transkrip contoh", str(out_path))
    assert out_path.read_text(encoding="utf-8") == "Transkrip contoh"


def test_cli_health_and_info_commands(tmp_path: Path) -> None:
    """Click CLI health and info commands succeed with valid outputs."""
    runner = CliRunner()
    with patch("src.main.check_system_health", return_value={"endpoint": "/health", "status": "healthy"}):
        health_res = runner.invoke(cli, ["health"])
    assert health_res.exit_code == 0
    assert json.loads(health_res.output)["status"] == "healthy"

    sample = tmp_path / "clip.mp3"
    sample.write_bytes(b"1234")
    with patch.object(
        AudioProcessor,
        "get_metadata",
        return_value={"path": str(sample), "format": "mp3", "size_bytes": 4, "duration_seconds": 1.2},
    ):
        info_res = runner.invoke(cli, ["info", str(sample)])
    assert info_res.exit_code == 0
    assert "MP3" in info_res.output


def test_cli_analyze_and_batch_commands(tmp_path: Path) -> None:
    """Click CLI analyze and analyze-batch commands persist individual and master JSON reports."""
    runner = CliRunner()
    audio_file = tmp_path / "demo.mp3"
    audio_file.write_bytes(b"audio")

    fake_report = {
        "file_name": "demo.mp3",
        "duration_seconds": 5.0,
        "audio_quality": {"silence_ratio": 0.0, "clipping_detected": False, "avg_volume_db": -16.0},
        "issues": [],
        "llm_insights": {"summary": "Kualitas audio baik.", "recommended_actions": ["Siap arsip"]},
    }

    with patch("src.main.AudioAnalysisAgent") as mock_agent_cls:
        mock_agent_cls.return_value.analyze.return_value = fake_report
        single_res = runner.invoke(cli, ["analyze", str(audio_file), "--no-cache"])
        batch_res = runner.invoke(cli, ["analyze-batch", str(tmp_path), "--no-cache"])

    assert single_res.exit_code == 0
    assert (tmp_path / "demo_report.json").is_file()
    assert batch_res.exit_code == 0
    master = json.loads((tmp_path / "batch_insights_report.json").read_text(encoding="utf-8"))
    assert len(master) == 1
    assert master[0]["file_name"] == "demo.mp3"
