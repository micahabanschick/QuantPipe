#!/usr/bin/env bash
# Install a GitHub Actions self-hosted runner on the QuantPipe server.
# Run as root after setup_server.sh has completed.
#
# Usage:
#   bash /opt/quantpipe/deploy/install_runner.sh \
#       <repo-owner> <repo-name> <runner-registration-token>
#
# Get the token from:
#   https://github.com/<owner>/<repo>/settings/actions/runners/new
#   (click "Linux x64" — copy the token from the "Configure" step)
#
# Example:
#   bash install_runner.sh micahabanschick QuantPipe AXXXXXXXXXXXXXXXXXX
set -euo pipefail

OWNER="${1:?Usage: $0 <owner> <repo> <token>}"
REPO="${2:?Usage: $0 <owner> <repo> <token>}"
TOKEN="${3:?Usage: $0 <owner> <repo> <token>}"

APP_USER="quantpipe"
RUNNER_DIR="/opt/quantpipe/actions-runner"
RUNNER_VERSION="2.321.0"   # bump to latest from github.com/actions/runner/releases

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[x]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Must be run as root"

# ── 1. Allow the runner (quantpipe user) to call update.sh via sudo ───────────
log "Adding sudoers rule for runner..."
SUDOERS_FILE="/etc/sudoers.d/quantpipe-runner"
cat > "$SUDOERS_FILE" << 'EOF'
# Allow the quantpipe service account (GitHub Actions runner) to run the
# deploy script without a password.  Scope is intentionally narrow.
quantpipe ALL=(root) NOPASSWD: /opt/quantpipe/deploy/update.sh
EOF
chmod 440 "$SUDOERS_FILE"
visudo -cf "$SUDOERS_FILE" || die "sudoers syntax error — check $SUDOERS_FILE"
log "Sudoers rule written to $SUDOERS_FILE"

# ── 2. Download the runner tarball ────────────────────────────────────────────
log "Downloading GitHub Actions runner v${RUNNER_VERSION}..."
mkdir -p "$RUNNER_DIR"
chown "$APP_USER:$APP_USER" "$RUNNER_DIR"

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  RUNNER_ARCH="x64"  ;;
    aarch64) RUNNER_ARCH="arm64" ;;
    armv7l)  RUNNER_ARCH="arm"  ;;
    *)       die "Unsupported architecture: $ARCH" ;;
esac

TARBALL="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"

sudo -u "$APP_USER" bash -c "
    cd $RUNNER_DIR
    curl -fsSL '$URL' -o runner.tar.gz
    tar xzf runner.tar.gz
    rm runner.tar.gz
"

# ── 3. Register the runner ────────────────────────────────────────────────────
log "Registering runner with github.com/$OWNER/$REPO..."
sudo -u "$APP_USER" bash -c "
    cd $RUNNER_DIR
    ./config.sh \
        --url 'https://github.com/$OWNER/$REPO' \
        --token '$TOKEN' \
        --name 'quantpipe-server' \
        --labels 'self-hosted,linux' \
        --work '_work' \
        --unattended \
        --replace
"

# ── 4. Install as a systemd service ──────────────────────────────────────────
log "Installing runner as systemd service..."
sudo -u "$APP_USER" bash -c "cd $RUNNER_DIR && sudo ./svc.sh install $APP_USER"
systemctl start actions.runner.*.service 2>/dev/null || \
    systemctl start "actions.runner.${OWNER}-${REPO}.quantpipe-server.service" 2>/dev/null || \
    (cd "$RUNNER_DIR" && ./svc.sh start)

log "Done."
echo ""
echo "================================================================"
echo "  GitHub Actions self-hosted runner installed and running."
echo "  Push to laptop-dev or main → dashboard updates automatically."
echo ""
echo "  Check status:  systemctl status 'actions.runner.*'"
echo "  View logs:     journalctl -u 'actions.runner.*' -f"
echo "================================================================"
