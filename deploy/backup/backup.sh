#!/usr/bin/env bash
# QuantPipe → Backblaze B2 incremental backup
# Runs daily via systemd timer. Configure B2 credentials in /opt/quantpipe/.env
# Required env vars: B2_ACCOUNT_ID, B2_APPLICATION_KEY, B2_BUCKET
set -euo pipefail

APP_DIR="/opt/quantpipe"
LOG_FILE="/var/log/quantpipe/backup.log"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

source "$APP_DIR/.env" 2>/dev/null || true

: "${B2_ACCOUNT_ID:?B2_ACCOUNT_ID not set in .env}"
: "${B2_APPLICATION_KEY:?B2_APPLICATION_KEY not set in .env}"
: "${B2_BUCKET:?B2_BUCKET not set in .env}"

log() { echo "[$TIMESTAMP] $*" | tee -a "$LOG_FILE"; }

log "Starting backup to b2:${B2_BUCKET}"

# Configure rclone B2 on the fly (no config file needed)
export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_ACCOUNT_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APPLICATION_KEY"

# Sync data directory (Parquet files, heartbeat, logs)
rclone sync \
    "$APP_DIR/data" \
    "b2:${B2_BUCKET}/data" \
    --exclude "**/.tmp/**" \
    --exclude "**/__pycache__/**" \
    --transfers 4 \
    --checkers 8 \
    --log-level INFO \
    --log-file "$LOG_FILE" \
    2>&1

# Backup .env encrypted with age (if age is installed)
# This gives you a recoverable secrets backup
if command -v age &>/dev/null && [[ -f "$APP_DIR/.age-pubkey" ]]; then
    PUBKEY=$(cat "$APP_DIR/.age-pubkey")
    age --recipient "$PUBKEY" "$APP_DIR/.env" | \
        rclone rcat "b2:${B2_BUCKET}/secrets/env.age"
    log "Encrypted .env backed up"
fi

log "Backup complete"
