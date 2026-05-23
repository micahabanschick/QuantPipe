# ============================================================================
# QUANTPIPE — Dockerfile
# ============================================================================
# Containerizes the existing QuantPipe Python pipeline WITHOUT code changes.
#
# What runs inside this container:
#   1. Cron: run_pipeline.py at 10pm ET weekdays (ingest + signal generation)
#   2. Streamlit: health_dashboard.py on port 3001 (Caddy proxies this)
#   3. uvicorn: mobile/api.py on port 3002 (mobile PWA, Caddy proxies this)
#   4. All Parquet data in /app/data/ (persisted via Docker volume)
#
# The build context is the QuantPipe repo root — clone it into
# apps/quantpipe/ or mount it. See docker-compose.yml for volume config.
# ============================================================================

FROM python:3.12-slim AS base

# ── System packages ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ── Install dependencies (cached layer — only re-runs if pyproject changes)
COPY pyproject.toml ./
RUN uv sync 2>/dev/null || true

# ── Copy full QuantPipe codebase ──────────────────────────────────────────
COPY . .

# ── Sync all dependency groups ────────────────────────────────────────────
RUN uv sync

# ── Create directories for runtime data ───────────────────────────────────
RUN mkdir -p /app/data /app/logs

# ── Set up cron for nightly pipeline ──────────────────────────────────────
# Runs run_pipeline.py at 10:00 PM ET (02:00 UTC next day) on weekdays.
# run_pipeline chains ingest_daily → generate_signals → ancillary pulls, and
# writes pipeline.log + .pipeline_heartbeat.json (consumed by the dashboard).
# All env vars are written to /etc/environment so cron can read them.
COPY <<'CRONTAB' /etc/cron.d/quantpipe-pipeline
# QuantPipe nightly pipeline — Mon-Fri at 10pm ET (02:00 UTC)
0 2 * * 1-5 root cd /app && /root/.local/bin/uv run python orchestration/run_pipeline.py >> /app/logs/pipeline.log 2>&1
CRONTAB

RUN chmod 0644 /etc/cron.d/quantpipe-pipeline && crontab /etc/cron.d/quantpipe-pipeline

# ── Entrypoint: start cron + mobile API + Streamlit ──────────────────────
COPY <<'ENTRYPOINT' /app/entrypoint.sh
#!/bin/bash
set -e

# Export env vars so cron jobs can access them
printenv | grep -v "no_proxy" >> /etc/environment

# Start cron daemon in background
cron
echo "✓ Cron daemon started (nightly pipeline at 10pm ET)"

# Start mobile PWA API in background — must be & so it doesn't block Streamlit
uv run uvicorn mobile.api:app \
    --host 0.0.0.0 \
    --port 3002 \
    --workers 1 \
    >> /app/logs/mobile.log 2>&1 &
echo "✓ Mobile API started on port 3002 (pid $!)"

echo "✓ Starting Streamlit dashboard on port 3001..."

# Start Streamlit as the foreground process
exec uv run streamlit run app.py \
    --server.port=3001 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
ENTRYPOINT

RUN chmod +x /app/entrypoint.sh

EXPOSE 3001
EXPOSE 3002

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:3001/_stcore/health || exit 1

CMD ["/app/entrypoint.sh"]
