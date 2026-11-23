# Operational Runbook — Audio Analysis Agent

This runbook documents operational procedures, readiness diagnostics, troubleshooting steps, and incident response playbooks for `audio-analysis-agent` (CLI and FastMCP server).

## 1. Service Overview

| Component | Entrypoint | Responsibility |
|---|---|---|
| MCP Server | `python mcp_server.py` | Exposes 4 FFmpeg audio tools + `health_check_tool()` via FastMCP |
| CLI Runner | `python -m src.main <cmd>` | Single-file analysis (`analyze`), batch (`analyze-batch`), transcription (`transcribe`), metadata (`info`), and `health` |
| ReAct Orchestrator | `src/agent/orchestrator.py` | Coordinates Groq LLM tool-calling loop, SHA-256 caching, rate limiting, and deterministic schema normalization |
| Readiness & Boundary | `src/health.py` | `/health` readiness contract, centralized `AudioAnalysisErrorHandler`, and `SimpleRateLimiter` |

## 2. Health Checks & Readiness Verification

### Local / Container Health Probe

```bash
python -m src.main health
# or via Makefile
make health
```

Expected healthy response (`exit code 0`):

```json
{
  "endpoint": "/health",
  "status": "healthy",
  "checks": {
    "ffmpeg": { "available": true, "path": "/usr/bin/ffmpeg" },
    "ffprobe": { "available": true, "path": "/usr/bin/ffprobe" },
    "configuration": { "valid": true, "api_key_configured": true }
  }
}
```

### MCP Tool Probe

Invoke `health_check_tool()` from any connected MCP client to verify binary availability and configuration validity remotely.

### Docker Container Healthcheck

The container image embeds a `HEALTHCHECK` running every 30s (timeout 5s, 3 retries). Inspect status with:

```bash
docker inspect --format='{{json .State.Health}}' audio-analysis-agent
```

## 3. Environment Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (for LLM/Whisper) | — | Groq API authentication secret (never logged) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | LLM model for ReAct tool calling |
| `GROQ_BASE_URL` | No | Groq SDK default | Custom API base URL (`/openai/v1` stripped automatically) |
| `MAX_ITERATIONS` | No | `10` | Upper bound on ReAct iterations (`1..50`) |
| `API_TIMEOUT_SECONDS` | No | `60` | Explicit network timeout per Groq API call (`5..300`) |
| `RATE_LIMIT_CALLS_PER_MINUTE` | No | `30` | Sliding-window outbound Groq call throttle |
| `FFMPEG_PATH` / `FFPROBE_PATH` | No | Auto-discovered | Explicit binary overrides when not on `PATH` |
| `LOG_LEVEL` | No | `INFO` | Structured logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

## 4. Incident Response Playbooks

### Playbook A: Health Status `degraded` — FFmpeg/FFprobe Not Found

- **Symptom**: `health` exits `1`; `checks.ffmpeg.available` or `checks.ffprobe.available` is `false`.
- **Cause**: FFmpeg is not installed on the host or `FFMPEG_PATH` points to an invalid path.
- **Resolution**:
  1. Linux/Container: `apt-get update && apt-get install -y ffmpeg` (already included in official `Dockerfile`).
  2. Windows: `winget install Gyan.FFmpeg` or activate Conda env `conda activate voicescript`.
  3. Set explicit `FFMPEG_PATH` and `FFPROBE_PATH` in `.env` if installed in a non-standard directory.
  4. Re-run `make health` to confirm `status: "healthy"`.

### Playbook B: Groq API Rate Limiting or Timeouts

- **Symptom**: `Gagal memanggil API` errors or elevated latency in structured logs.
- **Resolution**:
  1. Inspect logs with `LOG_LEVEL=DEBUG` to confirm whether `SimpleRateLimiter` is throttling or upstream Groq returned HTTP 429/5xx.
  2. Lower `RATE_LIMIT_CALLS_PER_MINUTE` (e.g. `15`) to stay within Groq tier quotas.
  3. Increase `API_TIMEOUT_SECONDS` (up to `120`) for large multi-tool reasoning sessions.
  4. Rely on SHA-256 disk caching (enabled by default) to avoid re-querying previously analyzed audio files.

### Playbook C: Corrupted or Stale Analysis Cache

- **Symptom**: Warning log `Failed to read cached report at .cache/...` or outdated analysis results after changing thresholds in `config/settings.yaml`.
- **Resolution**:
  1. Bypass cache for a single run using `--no-cache`: `python -m src.main analyze data/bad_audio.mp3 --no-cache`.
  2. Clear local cache safely: remove files inside `.cache/` and re-run analysis.

### Playbook D: FFmpeg Subprocess Timeout on Large Media Files

- **Symptom**: `ffmpeg timeout setelah 120 detik.` returned by silence/volume/noise tools.
- **Resolution**:
  1. Check file size and duration via `python -m src.main info <file>`.
  2. Split oversized recordings (>25 MB / multi-hour audio) via `AudioProcessor.split_audio()` prior to batch analysis.

## 5. Deployment & Rollback

1. Build and verify locally: `make lint && make test && make docker-build`.
2. Deploy container: `docker compose up -d --build`.
3. Rollback: pin image tag to the previous semantic release (`git checkout v1.0.0 && make docker-build && docker compose up -d`).
