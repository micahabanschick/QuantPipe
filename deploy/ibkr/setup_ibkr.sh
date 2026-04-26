#!/usr/bin/env bash
# Install headless IB Gateway + IBC on Ubuntu server
# Run as root: bash /opt/quantpipe/deploy/ibkr/setup_ibkr.sh
set -euo pipefail

APP_USER="quantpipe"
IBC_DIR="/opt/ibc"
GW_DIR="/opt/ibgateway"
IBC_CONFIG="$IBC_DIR/config.ini"
LOG_DIR="/var/log/quantpipe"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Must run as root"

# ── 1. Dependencies ───────────────────────────────────────────────────────────
log "Installing Java, Xvfb, and display utilities..."
apt-get update -qq
apt-get install -y -qq default-jre xvfb x11-utils wget unzip curl

java -version 2>&1 | head -1 && log "Java OK"

# ── 2. Temporary Xvfb for the IB Gateway installer (needs a display) ─────────
log "Starting temporary virtual display for installer..."
Xvfb :99 -screen 0 1024x768x24 -ac &
XVFB_PID=$!
sleep 3
export DISPLAY=:99

# ── 3. Download and install IB Gateway ────────────────────────────────────────
GW_INSTALLER="/tmp/ibgateway-installer.sh"
GW_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"

log "Downloading IB Gateway (stable)..."
wget -q --show-progress "$GW_URL" -O "$GW_INSTALLER"
chmod +x "$GW_INSTALLER"

log "Installing IB Gateway to $GW_DIR..."
mkdir -p "$GW_DIR"
# -q = quiet, -dir = install directory, -console = non-GUI install
"$GW_INSTALLER" -q -dir "$GW_DIR" || {
    warn "Installer exited non-zero — checking if Gateway was installed anyway..."
    ls "$GW_DIR" | head -5
}

# Stop temp display
kill "$XVFB_PID" 2>/dev/null || true
unset DISPLAY
rm -f "$GW_INSTALLER"

# ── 4. Download and install IBC ───────────────────────────────────────────────
log "Fetching latest IBC release from GitHub..."
IBC_VERSION=$(curl -fsSL https://api.github.com/repos/IbcAlpha/IBC/releases/latest \
    | grep '"tag_name"' | cut -d'"' -f4)
log "IBC version: $IBC_VERSION"

IBC_ZIP="/tmp/ibc.zip"
wget -q --show-progress \
    "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
    -O "$IBC_ZIP"

mkdir -p "$IBC_DIR"
unzip -q -o "$IBC_ZIP" -d "$IBC_DIR"
chmod +x "$IBC_DIR/scripts/"*.sh 2>/dev/null || true
rm -f "$IBC_ZIP"

# ── 5. Create IBC config from template ───────────────────────────────────────
if [[ ! -f "$IBC_CONFIG" ]]; then
    log "Creating IBC config from template..."
    cp "/opt/quantpipe/deploy/ibkr/ibc.ini.template" "$IBC_CONFIG"
    chmod 600 "$IBC_CONFIG"
    warn "Fill in your IBKR credentials: nano $IBC_CONFIG"
else
    warn "$IBC_CONFIG already exists — not overwriting"
fi

# ── 6. Fix IBC script permissions (unzip doesn't preserve +x) ────────────────
log "Fixing IBC script permissions..."
chmod +x "$IBC_DIR"/*.sh "$IBC_DIR/scripts/"*.sh 2>/dev/null || true

# ── 7. Patch gatewaystart.sh with correct server paths ───────────────────────
# IBC hardcodes ~/Jts and version 1019 — patch to match our install
log "Patching gatewaystart.sh paths..."
GW_VERSION=$(ls "$GW_DIR/jars/" | grep 'jts4launch-' | grep -o '[0-9]*' | head -1)
GW_VERSION=${GW_VERSION:-1037}
log "  Detected IB Gateway version: $GW_VERSION"

sed -i \
    -e "s|^TWS_MAJOR_VRSN=.*|TWS_MAJOR_VRSN=${GW_VERSION}|" \
    -e "s|^IBC_INI=.*|IBC_INI=${IBC_CONFIG}|" \
    -e "s|^TWS_PATH=.*|TWS_PATH=/opt/Jts|" \
    -e "s|^TWS_SETTINGS_PATH=.*|TWS_SETTINGS_PATH=${GW_DIR}|" \
    -e "s|^LOG_PATH=.*|LOG_PATH=${LOG_DIR}|" \
    "$IBC_DIR/gatewaystart.sh"

# IBC expects: TWS_PATH/ibgateway/<version>/jars
# Create that symlink structure pointing at our actual install
log "Creating IBC directory structure symlink..."
mkdir -p /opt/Jts/ibgateway
ln -sfn "$GW_DIR" "/opt/Jts/ibgateway/${GW_VERSION}"

# ── 8. Ownership ──────────────────────────────────────────────────────────────
chown -R "$APP_USER:$APP_USER" "$IBC_DIR" "$GW_DIR" /opt/Jts

# ── 9. Install systemd services ───────────────────────────────────────────────
log "Installing systemd services..."
DEPLOY_DIR="/opt/quantpipe/deploy"
cp "$DEPLOY_DIR/systemd/quantpipe-xvfb.service"      /etc/systemd/system/
cp "$DEPLOY_DIR/systemd/quantpipe-ibgateway.service"  /etc/systemd/system/
systemctl daemon-reload
systemctl enable quantpipe-xvfb quantpipe-ibgateway

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  IB Gateway + IBC Installation Complete"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  IB Gateway : $GW_DIR"
echo "  IBC        : $IBC_DIR"
echo "  Config     : $IBC_CONFIG"
echo ""
echo "  Next steps:"
echo "  1. Fill in IBKR credentials:"
echo "     nano $IBC_CONFIG"
echo "     (set IbLoginId, IbPassword, TradingMode)"
echo ""
echo "  2. Start services:"
echo "     systemctl start quantpipe-xvfb"
echo "     systemctl start quantpipe-ibgateway"
echo ""
echo "  3. Watch startup log (first login takes ~30s):"
echo "     journalctl -fu quantpipe-ibgateway"
echo ""
echo "  4. Test connection from QuantPipe dashboard:"
echo "     http://10.0.0.1:8501  → Paper/Live Trading → Test Connection"
echo "════════════════════════════════════════════════════════"
