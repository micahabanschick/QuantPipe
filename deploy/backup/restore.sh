#!/usr/bin/env bash
# Restore QuantPipe data from Backblaze B2
# Usage: bash restore.sh [--date 2025-01-15]
set -euo pipefail

APP_DIR="/opt/quantpipe"
source "$APP_DIR/.env" 2>/dev/null || true

: "${B2_ACCOUNT_ID:?}" "${B2_APPLICATION_KEY:?}" "${B2_BUCKET:?}"

RESTORE_DIR="${APP_DIR}/data"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "[!] This will overwrite $RESTORE_DIR with data from b2:${B2_BUCKET}/data"
echo "    Dry run: ${DRY_RUN:-no}"
read -rp "    Continue? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_ACCOUNT_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APPLICATION_KEY"

systemctl stop quantpipe-streamlit.service 2>/dev/null || true

rclone sync \
    "b2:${B2_BUCKET}/data" \
    "$RESTORE_DIR" \
    $DRY_RUN \
    --transfers 8 \
    --progress

systemctl start quantpipe-streamlit.service 2>/dev/null || true
echo "[+] Restore complete."
