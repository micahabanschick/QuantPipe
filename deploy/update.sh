#!/usr/bin/env bash
# Pull latest code and restart services — run on server as root.
# Invoked automatically by the GitHub Actions self-hosted runner on every push.
# Usage: sudo bash /opt/quantpipe/deploy/update.sh
set -euo pipefail

APP_DIR="/opt/quantpipe"
APP_USER="quantpipe"
DEPLOY_SYSTEMD="$APP_DIR/deploy/systemd"
SYSTEMD_DIR="/etc/systemd/system"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

# ── 1. Pull latest code ───────────────────────────────────────────────────────
log "Pulling latest code..."
sudo -u "$APP_USER" git -C "$APP_DIR" pull

# ── 2. Sync Python dependencies ───────────────────────────────────────────────
log "Syncing dependencies..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    uv sync --extra execution --extra portfolio --extra backtest
"

# ── 3. Sync systemd unit files from repo ─────────────────────────────────────
# IMPORTANT: daemon-reload alone does NOT pick up changes to unit files that
# live in the repo — they must be copied to /etc/systemd/system/ first.
# Skipping this step is why the server ran on the old timer schedule.
log "Syncing systemd unit files from repo..."

UNITS=(
    quantpipe-pipeline.service
    quantpipe-pipeline.timer
    quantpipe-rebalance.service
    quantpipe-rebalance.timer
    quantpipe-streamlit.service
    quantpipe-backup.service
    quantpipe-backup.timer
)

changed=0
for unit in "${UNITS[@]}"; do
    src="$DEPLOY_SYSTEMD/$unit"
    dst="$SYSTEMD_DIR/$unit"
    if [[ -f "$src" ]]; then
        if ! cmp -s "$src" "$dst" 2>/dev/null; then
            cp "$src" "$dst"
            log "  Updated $unit"
            changed=1
        fi
    fi
done

# ── 4. Reload systemd if any unit file changed ────────────────────────────────
log "Reloading systemd..."
systemctl daemon-reload

# ── 5. Restart Streamlit (always — picks up new Python code) ─────────────────
log "Restarting Streamlit..."
systemctl restart quantpipe-streamlit.service

# ── 6. Re-enable and reset timers if unit files changed ──────────────────────
if [[ $changed -eq 1 ]]; then
    warn "Unit files changed — resetting timers so new schedules take effect."
    for timer in quantpipe-pipeline.timer quantpipe-rebalance.timer quantpipe-backup.timer; do
        if systemctl list-unit-files "$timer" &>/dev/null; then
            systemctl enable "$timer" --now 2>/dev/null || true
            # Reset the timer's last-trigger timestamp so Persistent=false
            # doesn't carry over a stale "already fired" state.
            systemctl stop "$timer" 2>/dev/null || true
            systemctl start "$timer" 2>/dev/null || true
        fi
    done
fi

# ── 7. Summary ────────────────────────────────────────────────────────────────
log "Done."
echo ""
systemctl status quantpipe-streamlit.service --no-pager -l
echo ""
echo "Timer schedule:"
systemctl list-timers --no-pager | grep -E "quantpipe|NEXT|LAST" || true
