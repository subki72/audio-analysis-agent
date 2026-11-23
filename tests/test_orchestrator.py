"""Offline unit tests for AudioAnalysisAgent ReAct orchestration, parsing, and caching."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent.orchestrator import AudioAnalysisAgent
from src.config import AppConfig


def _build_mock_response(content: str = "", tool_calls: list[Any] | None = None) -> SimpleNamespace:
    """Build a lightweight stub matching the Groq chat completion response shape."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _build_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """Build a stub Groq tool call object."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent(tmp_path: Path, mock_client: MagicMock, *, use_cache: bool = True) -> AudioAnalysisAgent:
    """Instantiate AudioAnalysisAgent with offline injected config, client, and cache dir."""
    config = AppConfig(groq_api_key="test_token_placeholder", max_iterations=4)
    return AudioAnalysisAgent(
        settings={},
        use_cache=use_cache,
        config=config,
        client=mock_client,
        cache_dir=str(tmp_path / ".cache"),
    )


def test_orchestrator_requires_api_key_when_no_client() -> None:
    """AudioAnalysisAgent raises ValueError if GROQ_API_KEY is absent and no client is injected."""
    config = AppConfig(groq_api_key=None)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        AudioAnalysisAgent(config=config)


def test_orchestrator_rejects_missing_and_invalid_paths(tmp_path: Path) -> None:
    """Analyze returns structured error payloads for missing or null-byte file paths."""
    agent = _make_agent(tmp_path, MagicMock())
    missing = agent.analyze(str(tmp_path / "missing.mp3"))
    assert "error" in missing

    null_byte = agent.analyze("bad\x00audio.mp3")
    assert "error" in null_byte


def test_orchestrator_react_loop_and_cache_hit(tmp_path: Path) -> None:
    """ReAct loop invokes registered tools, normalizes final JSON, and serves cached repeats."""
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"fake-audio-bytes-for-hash")

    final_payload = {
        "file_name": "sample.mp3",
        "duration_seconds": 12.5,
        "audio_quality": {"silence_ratio": 0.05, "clipping_detected": False, "avg_volume_db": -18.0},
        "issues": [],
        "llm_insights": {"summary": "Audio bersih.", "recommended_actions": ["Pertahankan gain"]},
    }
    tool_call = _build_tool_call("call_1", "get_audio_metadata", json.dumps({"file_path": str(audio_file)}))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _build_mock_response(tool_calls=[tool_call]),
        _build_mock_response(content=f"```json\n{json.dumps(final_payload)}\n```"),
    ]

    agent = _make_agent(tmp_path, mock_client, use_cache=True)
    with patch.dict("src.agent.orchestrator.TOOL_REGISTRY", {"get_audio_metadata": lambda **_: {"duration": 12.5}}):
        first_result = agent.analyze(str(audio_file), verbose=True)
        second_result = agent.analyze(str(audio_file), verbose=True)

    assert first_result == final_payload
    assert second_result == final_payload
    assert mock_client.chat.completions.create.call_count == 2


def test_orchestrator_logs_warning_when_cache_write_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cache write failures emit a WARNING log instead of being silently swallowed."""
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"audio-content")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _build_mock_response(
        content=json.dumps({"file_name": "clip.mp3", "duration_seconds": 5.0, "issues": []})
    )
    agent = _make_agent(tmp_path, mock_client, use_cache=True)

    with caplog.at_level(logging.WARNING):
        with patch("src.agent.orchestrator.open", side_effect=[open(audio_file, "rb"), OSError("disk full")]):
            result = agent.analyze(str(audio_file))

    assert result["file_name"] == "clip.mp3"
    assert any("Failed to write analysis cache" in record.message for record in caplog.records)


def test_orchestrator_logs_warning_on_invalid_embedded_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed JSON inside LLM text logs a WARNING and returns safe fallback schema."""
    agent = _make_agent(tmp_path, MagicMock())
    with caplog.at_level(logging.WARNING):
        report = agent._parse_final_report("Hasil analisis: {not valid json at all}")

    assert report["file_name"] == "unknown"
    assert report["llm_insights"]["_raw_response"].startswith("Hasil analisis")
    assert any("Failed to parse extracted JSON block" in record.message for record in caplog.records)


def test_orchestrator_normalize_schema_legacy_payload_generates_issues(tmp_path: Path) -> None:
    """Legacy file_metadata/audio_metrics payloads normalize cleanly with synthesized issues."""
    agent = _make_agent(tmp_path, MagicMock())
    legacy_raw = {
        "file_metadata": {"filename": "deposition.mp3", "duration_seconds": 90.0},
        "audio_metrics": {
            "silence_ratio": 0.25,
            "volume_clipping": True,
            "max_volume_db": -0.2,
            "mean_volume_db": -40.5,
        },
        "llm_insights": {"summary": "Banyak keheningan dan clipping."},
    }
    tool_results = {
        "detect_silence": {
            "silence_segments": [
                {"start": 5.0, "end": 15.0, "duration": 10.0},
                {"start": 30.0, "end": 42.5, "duration": 12.5},
            ]
        }
    }

    normalized = agent._normalize_schema(legacy_raw, tool_results)
    assert normalized["file_name"] == "deposition.mp3"
    assert normalized["duration_seconds"] == 90.0
    assert normalized["audio_quality"]["clipping_detected"] is True
    assert normalized["audio_quality"]["avg_volume_db"] == -40.5
    assert len(normalized["issues"]) == 3
    assert "30.0s" in normalized["issues"][0]


def test_orchestrator_handles_unknown_tool_and_invalid_arguments(tmp_path: Path) -> None:
    """Unknown tool calls and invalid JSON tool arguments return tool error messages cleanly."""
    audio_file = tmp_path / "voice.mp3"
    audio_file.write_bytes(b"bytes")

    calls = [
        _build_tool_call("c1", "nonexistent_tool", json.dumps({"file_path": str(audio_file)})),
        _build_tool_call("c2", "get_audio_metadata", "{invalid-json"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _build_mock_response(tool_calls=calls),
        _build_mock_response(content=json.dumps({"file_name": "voice.mp3", "duration_seconds": 1.0, "issues": []})),
    ]

    agent = _make_agent(tmp_path, mock_client, use_cache=False)
    result = agent.analyze(str(audio_file))
    assert result["file_name"] == "voice.mp3"


def test_orchestrator_handles_api_error_and_max_iterations(tmp_path: Path) -> None:
    """API exceptions and iteration exhaustion return explicit error payloads."""
    audio_file = tmp_path / "loop.mp3"
    audio_file.write_bytes(b"bytes")

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = RuntimeError("network timeout")
    agent_err = _make_agent(tmp_path, failing_client, use_cache=False)
    err_res = agent_err.analyze(str(audio_file))
    assert "Gagal memanggil API" in err_res["error"]

    loop_client = MagicMock()
    loop_call = _build_tool_call("c1", "unknown_tool", "{}")
    loop_client.chat.completions.create.return_value = _build_mock_response(tool_calls=[loop_call])
    agent_loop = _make_agent(tmp_path, loop_client, use_cache=False)
    max_res = agent_loop.analyze(str(audio_file))
    assert "batas iterasi maksimum" in max_res["error"]
