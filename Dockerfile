# Stage 1: Dependency Builder
FROM python:3.15-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Hardened Non-Root Runtime with FFmpeg
FROM python:3.15-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --shell /sbin/nologin --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup mcp_server.py ./mcp_server.py

RUN mkdir -p /app/.cache /app/data/transcripts \
    && chown -R appuser:appgroup /app/.cache /app/data

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys; from src.health import check_system_health; res = check_system_health(); sys.exit(0 if res['status'] == 'healthy' else 1)"]

CMD ["python", "mcp_server.py"]
