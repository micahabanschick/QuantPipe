#!/usr/bin/env bash
# QuantPipe Server Setup — run once as root on a fresh Ubuntu 24.04 VPS
# Usage: bash setup_server.sh <client-wg-pubkey> <git-repo-url>
# Example: bash setup_server.sh "abc123...pubkey..." "https://github.com/you/QuantPipe.git"
set -euo pipefail

CLIENT_WG_PUBKEY="${1:?Usage: $0 <client-wg-pubkey> <git-repo-url>}"
REPO_URL="${2:?Usage: $0 <client-wg-pubkey> <git-repo-url>}"

APP_DIR="/opt/quantpipe"
APP_USER="quantpipe"
WG_IFACE="wg0"
WG_PORT=51820
VPN_SERVER_IP="10.0.0.1"
VPN_CLIENT_IP="10.0.0.2"
VPN_SUBNET="10.0.0.0/24"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Must be run as root"
grep -qi "ubuntu" /etc/os-release || warn "This script targets Ubuntu — other distros may need adjustments"

NET_IFACE=$(ip route get 8.8.8.8 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
log "Detected network interface: $NET_IFACE"

# ── 1. System update ───────────────────────────────────────────────────────────
log "Updating system packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
apt-get install -y -qq \
    wireguard ufw curl git unzip \
    nginx rclone \
    unattended-upgrades apt-listchanges \
    fail2ban

# ── 2. Dedicated app user ─────────────────────────────────────────────────────
log "Creating $APP_USER user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --shell /bin/bash --home "$APP_DIR" --create-home "$APP_USER"
fi

# ── 3. SSH hardening ──────────────────────────────────────────────────────────
log "Hardening SSH..."
SSHD_CONF="/etc/ssh/sshd_config.d/99-quantpipe.conf"
cat > "$SSHD_CONF" << 'EOF'
PasswordAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
X11Forwarding no
AllowTcpForwarding no
MaxAuthTries 3
LoginGraceTime 20
EOF
# Ubuntu 22.04+ uses ssh.service; older systems use sshd.service
systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true

# ── 4. Firewall (UFW) ─────────────────────────────────────────────────────────
log "Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh comment "SSH"
ufw allow "${WG_PORT}/udp" comment "WireGuard"
# Streamlit and Vault only reachable via VPN (added after wg0 comes up)
ufw --force enable

# ── 5. WireGuard ──────────────────────────────────────────────────────────────
log "Generating WireGuard server keys..."
wg genkey | tee /etc/wireguard/server.key | wg pubkey > /etc/wireguard/server.pub
chmod 600 /etc/wireguard/server.key

SERVER_PRIV=$(cat /etc/wireguard/server.key)
SERVER_PUB=$(cat /etc/wireguard/server.pub)

log "Writing WireGuard server config..."
cat > /etc/wireguard/${WG_IFACE}.conf << EOF
[Interface]
Address = ${VPN_SERVER_IP}/24
ListenPort = ${WG_PORT}
PrivateKey = ${SERVER_PRIV}

# Open VPN-internal ports for Streamlit and Vault after tunnel comes up
PostUp   = ufw allow in on ${WG_IFACE} to any port 8501 comment "Streamlit (VPN only)"
PostUp   = ufw allow in on ${WG_IFACE} to any port 8200 comment "Vault (VPN only)"
PostDown = ufw delete allow in on ${WG_IFACE} to any port 8501
PostDown = ufw delete allow in on ${WG_IFACE} to any port 8200

[Peer]
# Laptop (peer 0.2)
PublicKey = ${CLIENT_WG_PUBKEY}
AllowedIPs = ${VPN_CLIENT_IP}/32
EOF
chmod 600 /etc/wireguard/${WG_IFACE}.conf

# Enable IP forwarding for VPN
echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/99-wireguard.conf
sysctl -p /etc/sysctl.d/99-wireguard.conf

systemctl enable --now wg-quick@${WG_IFACE}

# ── 6. Unattended upgrades ────────────────────────────────────────────────────
log "Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
cat > /etc/apt/apt.conf.d/51unattended-upgrades-security << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

# ── 7. fail2ban ───────────────────────────────────────────────────────────────
log "Configuring fail2ban..."
cat > /etc/fail2ban/jail.d/sshd.conf << 'EOF'
[sshd]
enabled = true
maxretry = 3
bantime  = 3600
findtime = 600
EOF
systemctl enable --now fail2ban

# ── 8. Install uv ─────────────────────────────────────────────────────────────
log "Installing uv for $APP_USER..."
sudo -u "$APP_USER" bash -c '
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
'

# ── 9. Clone and install QuantPipe ────────────────────────────────────────────
log "Cloning QuantPipe from $REPO_URL..."
if [[ -d "$APP_DIR/.git" ]]; then
    warn "$APP_DIR already has a repo — pulling latest"
    sudo -u "$APP_USER" git -C "$APP_DIR" pull
else
    sudo -u "$APP_USER" git clone "$REPO_URL" "$APP_DIR"
fi

log "Installing Python dependencies..."
sudo -u "$APP_USER" bash -c "
    cd $APP_DIR
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    uv sync --extra execution --extra portfolio --extra backtest
"

# ── 10. Create .env from template ─────────────────────────────────────────────
if [[ ! -f "$APP_DIR/.env" ]]; then
    log "Creating .env from .env.example — fill in your secrets before starting services"
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
fi

# ── 11. Log directory ─────────────────────────────────────────────────────────
mkdir -p /var/log/quantpipe
chown "$APP_USER:$APP_USER" /var/log/quantpipe

# ── 12. Install systemd services ──────────────────────────────────────────────
log "Installing systemd services..."
DEPLOY_DIR="$APP_DIR/deploy"
for f in quantpipe-pipeline.service quantpipe-pipeline.timer quantpipe-streamlit.service; do
    cp "$DEPLOY_DIR/systemd/$f" "/etc/systemd/system/$f"
done
systemctl daemon-reload

# Enable pipeline timer (runs Mon-Fri 06:15 UTC)
systemctl enable --now quantpipe-pipeline.timer

# Enable Streamlit (auto-starts, accessible on VPN only)
systemctl enable --now quantpipe-streamlit.service

# ── 13. Vault ─────────────────────────────────────────────────────────────────
log "Installing HashiCorp Vault..."
bash "$DEPLOY_DIR/vault/setup_vault.sh"

# ── 14. Summary ───────────────────────────────────────────────────────────────
SERVER_IP=$(curl -s4 https://api.ipify.org)
echo ""
echo "════════════════════════════════════════════════════════"
echo "  QuantPipe Server Setup Complete"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Server public IP  : $SERVER_IP"
echo "  WireGuard pub key : $SERVER_PUB"
echo "  VPN server IP     : $VPN_SERVER_IP"
echo ""
echo "  Next steps:"
echo "  1. Fill in secrets: nano $APP_DIR/.env"
echo "  2. (Optional) init Vault: bash $DEPLOY_DIR/vault/init_secrets.sh"
echo "  3. Run backfill:  sudo -u $APP_USER bash -c 'cd $APP_DIR && .venv/bin/python orchestration/backfill_history.py'"
echo "  4. Add more peers: bash $DEPLOY_DIR/add_peer.sh <name> <pubkey> <vpn-ip>"
echo ""
echo "  Dashboard URL (VPN required): http://$VPN_SERVER_IP:8501"
echo "  Vault URL     (VPN required): http://$VPN_SERVER_IP:8200"
echo "════════════════════════════════════════════════════════"
