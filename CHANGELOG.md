# Changelog

All notable changes to `audio-analysis-agent` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-23

### Added
- Validated Pydantic configuration (`src/config.py`) supporting environment overrides, Groq base URL normalization, structured logging setup, and audio file path guards.
- System readiness probe (`src/health.py`) exposing the `/health` contract (`check_system_health()`), centralized `AudioAnalysisErrorHandler`, and sliding-window `SimpleRateLimiter`.
- FastMCP `health_check_tool()` endpoint in `mcp_server.py` and `voicescript health` CLI subcommand in `src/main.py`.
- Expanded offline unit and integration test suites (`tests/test_orchestrator.py`, `tests/test_mcp_and_health.py`, `tests/test_cli_and_helpers.py`) achieving >80% coverage across `src/`.
- Deterministic dependency lockfiles (`requirements.txt` exact pins, `requirements-lock.txt`, `uv.lock`) and `pyproject.toml` with Ruff and Pytest settings.
- Multi-stage GitHub Actions CI pipeline (Flake8, Ruff, Pytest + FFmpeg coverage gate, Semgrep SAST, Docker build, Trivy container scan), Dependabot configuration, and `SECURITY.md`.
- Operational tooling: `Makefile`, multi-stage non-root `Dockerfile` (UID 10001 with FFmpeg and `HEALTHCHECK`), `docker-compose.yml`, `.pre-commit-config.yaml`, and `docs/RUNBOOK.md`.
- Governance baseline: `LICENSE` (MIT), `CONTRIBUTING.md`, and `CHANGELOG.md`.

### Changed
- Decomposed `AudioAnalysisAgent.analyze()`, `_normalize_schema()`, CLI commands, and FFmpeg tool modules into cohesive `< 50`-line functions with nesting depth `<= 3` and full Python type hints.
- Replaced raw `print()` diagnostics in core modules with leveled `logging.getLogger(__name__)` instrumentation.
- Added explicit API request timeouts to Groq chat and Whisper transcription clients.

### Fixed
- Replaced two silently swallowed exceptions in `src/agent/orchestrator.py` (cache write failure and fallback JSON decode error) with structured `logger.warning()` diagnostics.
- Removed unused local variable `api_settings` in `src/agent/orchestrator.py` and resolved all 220 `flake8` violations.
- Handled parentless output paths safely in `Transcriber.save_output()`.

## [0.1.0] - 2026-05-30

### Added
- Initial VoiceScript Audio Analysis ReAct Agent with Groq tool calling, FFmpeg tool registry (`get_audio_metadata`, `detect_silence`, `detect_clipping`, `detect_noise`), Click CLI, FastMCP server, sample audio assets, and baseline tests.
