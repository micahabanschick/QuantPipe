#!/usr/bin/env bash
# Pull latest code and restart services — run on server as root
# Usage: bash /opt/quantpipe/deploy/update.sh
set -euo pipefail

APP_DIR="/opt/quantpipe"
APP_USER="quantpipe"

echo "[+] Pulling latest code..."
sudo -u "$APP_USER" git -C "$APP_DIR" fetch origin
sudo -u "$APP_USER" git -C "$APP_DIR" reset --hard origin/main

echo "[+] Syncing dependencies..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    uv sync --extra execution --extra portfolio --extra backtest
"

echo "[+] Reloading systemd services..."
systemctl daemon-reload
systemctl restart quantpipe-streamlit.service

echo "[+] Done. Pipeline timer unchanged (next run at scheduled time)."
systemctl status quantpipe-streamlit.service --no-pager -l
