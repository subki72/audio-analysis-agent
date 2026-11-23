# VoiceScript — Audio Analysis Agent (FFmpeg + Groq LLM)

[![CI Pipeline](https://github.com/subki72/audio-analysis-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/audio-analysis-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](mcp_server.py)

Production-grade AI audio quality inspection service combining **FFmpeg/FFprobe** signal diagnostics, a **ReAct (Reason + Act)** tool-calling agent powered by **Groq LLM**, a **FastMCP** server, and a **Click** CLI for single-file, batch, and Whisper transcription workflows.

## Architecture

Instead of hardcoding fixed signal-processing pipelines, the orchestrator runs a bounded ReAct loop where the LLM dynamically selects which FFmpeg tools to invoke, collects deterministic metrics, and synthesizes a schema-normalized JSON quality report.

```text
CLI (src/main.py) / MCP Client (mcp_server.py)
            |
            v
+----------------------------------------+
|  AppConfig + Path Guards (src/config)  |
|  Health & Error Boundary (src/health)  |
+----------------------------------------+
            |
            v
+----------------------------------------+
|  AudioAnalysisAgent (orchestrator.py)  |
|  - Groq Tool Calling + Rate Limiter    |
|  - SHA-256 Disk Cache (--no-cache)     |
|  - Deterministic Schema Normalization  |
+----------------------------------------+
      |         |         |         |
      v         v         v         v
  metadata   silence   volume     noise     <-- FFmpeg/FFprobe Tools (src/tools/)
  (ffprobe) (silencedet)(volumedet)(astats)
      \         |         |         /
       v        v         v        v
+----------------------------------------+
|       Structured JSON Report           |
+----------------------------------------+
```

### Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Groq API (`llama-3.3-70b-versatile`)** | Low-latency native tool-calling inference with explicit request timeouts and sliding-window rate limiting |
| **Deterministic post-processing** | `_normalize_schema()` guarantees uniform output structure regardless of LLM formatting variance |
| **`@register_tool` auto-discovery** | Generates OpenAI/Groq JSON tool schemas automatically from typed function signatures and docstrings |
| **FFmpeg/FFprobe auto-discovery** | Resolves binaries via `FFMPEG_PATH`/`FFPROBE_PATH`, system `PATH`, and Conda environments |
| **SHA-256 content caching** | Avoids redundant FFmpeg + LLM calls on unchanged files; bypass with `--no-cache` |
| **Centralized health & error boundary** | `/health` probe (`check_system_health()`) and `AudioAnalysisErrorHandler` prevent stack-trace leaks |

## Project Structure

```text
audio-analysis-agent/
├── .github/
│   ├── workflows/ci.yml        # Lint, Pytest + FFmpeg coverage, Semgrep SAST, Docker, Trivy
│   └── dependabot.yml          # Weekly pip, docker, and GitHub Actions security updates
├── config/
│   └── settings.yaml           # Thresholds, model defaults, and output settings
├── data/                       # Sample MP3 fixtures and generated example JSON reports
├── docs/
│   └── RUNBOOK.md              # Operational procedures, health checks, and incident playbooks
├── src/
│   ├── config.py               # Pydantic AppConfig, logging setup, and path validation guards
│   ├── health.py               # /health readiness probe, ErrorHandler, and SimpleRateLimiter
│   ├── main.py                 # Click CLI (analyze, analyze-batch, transcribe, info, health)
│   ├── agent/orchestrator.py   # Modular ReAct loop, caching, and schema normalization
│   ├── tools/                  # @register_tool FFmpeg suite (metadata, silence, volume, noise)
│   ├── audio/processor.py      # Local pydub metadata, conversion, and chunk splitting
│   ├── transcription/          # Groq Whisper speech-to-text client
│   └── utils/helpers.py        # YAML loader and SRT/WebVTT formatters
├── tests/                      # Offline unit & FFmpeg integration tests (>80% coverage)
├── mcp_server.py               # FastMCP server exposing 4 audio tools + health_check_tool
├── Dockerfile                  # Multi-stage non-root (UID 10001) container with FFmpeg + HEALTHCHECK
├── docker-compose.yml          # Containerized MCP service definition
├── Makefile                    # Standardized developer automation targets
├── pyproject.toml              # Project metadata, Ruff linter, and Pytest config
├── requirements.txt            # Pinned direct dependencies (==)
├── requirements-lock.txt       # Full pinned dependency lockfile
├── uv.lock                     # Cross-platform deterministic uv lockfile
├── SECURITY.md                 # Vulnerability disclosure policy and controls
├── CONTRIBUTING.md             # Contribution workflow and coding standards
├── CHANGELOG.md                # Semantic release history
└── LICENSE                     # MIT License
```

## Quickstart (Makefile)

### Prerequisites

- Python 3.10+
- `ffmpeg` and `ffprobe` installed (`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu)
- Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### Local Setup

```bash
# 1. Install pinned dependencies
make install

# 2. Configure environment variables
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# 3. Verify system readiness (FFmpeg, FFprobe, and configuration)
make health

# 4. Run linter and automated tests
make lint
make test-cov
```

Alternative Conda environment setup:

```bash
conda env create -f environment.yml
conda activate voicescript
```

## Usage

### CLI Commands

```bash
# Analyze a single audio file (writes <filename>_report.json)
python -m src.main analyze data/bad_audio.mp3

# Verbose reasoning logs and cache bypass
python -m src.main analyze data/bad_audio.mp3 --verbose --no-cache

# Batch-analyze all supported audio files in a directory (writes batch_insights_report.json)
python -m src.main analyze-batch data/

# Transcribe audio via Groq Whisper (formats: txt, srt, vtt, json)
python -m src.main transcribe data/bad_audio.mp3 --output-format srt --language id

# Inspect local file metadata
python -m src.main info data/bad_audio.mp3

# Run readiness health check
python -m src.main health
```

### MCP Server (FastMCP)

Start the Model Context Protocol server locally or via Makefile:

```bash
make run-mcp
# or: python mcp_server.py
```

Exposed MCP tools:

- `get_audio_metadata_tool(file_path)` — Format, duration, sample rate, channels, codec, and bitrate via `ffprobe`
- `detect_silence_tool(file_path)` — Silent segments, total silence duration, and silence ratio via `silencedetect`
- `detect_clipping_tool(file_path)` — Mean/max volume, clipping flag, severity, and histogram via `volumedetect`
- `detect_noise_tool(file_path)` — RMS/peak levels, dynamic range, crest factor, and noise classification via `astats`
- `health_check_tool()` — Readiness probe verifying FFmpeg, FFprobe, and validated runtime configuration (`/health` contract)

### Docker Deployment

Build and run the hardened non-root container (includes FFmpeg and built-in `HEALTHCHECK`):

```bash
make docker-build
make docker-up
```

## Output Report Schema

```json
{
  "file_name": "bad_audio.mp3",
  "duration_seconds": 121.11,
  "audio_quality": {
    "silence_ratio": 0.1007,
    "clipping_detected": true,
    "avg_volume_db": -16.6
  },
  "issues": [
    "High silence ratio (10.1%) - audio contains significant silent segments",
    "Audio clipping detected (max volume: -0.4 dB)"
  ],
  "llm_insights": {
    "summary": "Kualitas audio buruk karena segmen keheningan panjang dan distorsi clipping.",
    "recommended_actions": [
      "Terapkan limiter atau turunkan gain input",
      "Lakukan trimming otomatis pada segmen hening"
    ]
  }
}
```

Sample outputs are committed in `data/bad_audio_report.json`, `data/moonlight-plaza_report.json`, and `data/batch_insights_report.json`.

## Security and Governance

- **Security Policy**: See [SECURITY.md](SECURITY.md) for private vulnerability reporting and defense-in-depth controls (Semgrep SAST, Trivy image scanning, Dependabot).
- **Operations**: See [docs/RUNBOOK.md](docs/RUNBOOK.md) for health checks, environment variables, and incident response playbooks.
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and PR requirements.
- **Release History**: See [CHANGELOG.md](CHANGELOG.md).
- **License**: Released under the [MIT License](LICENSE).
