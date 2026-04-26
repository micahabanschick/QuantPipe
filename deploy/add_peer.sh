#!/usr/bin/env bash
# Add a new WireGuard peer (phone, second laptop, etc.)
# Usage: bash add_peer.sh <peer-name> <peer-pubkey> <vpn-ip>
# Example: bash add_peer.sh phone "abc123...pubkey..." 10.0.0.3
set -euo pipefail

PEER_NAME="${1:?Usage: $0 <peer-name> <peer-pubkey> <vpn-ip>}"
PEER_PUBKEY="${2:?}"
PEER_VPN_IP="${3:?}"   # e.g. 10.0.0.3

[[ $EUID -eq 0 ]] || { echo "Must be run as root"; exit 1; }

WG_IFACE="wg0"
WG_CONF="/etc/wireguard/${WG_IFACE}.conf"
SERVER_PUB=$(cat /etc/wireguard/server.pub)
SERVER_IP=$(curl -s4 https://api.ipify.org)

# Add peer to live config
wg set "$WG_IFACE" peer "$PEER_PUBKEY" allowed-ips "${PEER_VPN_IP}/32"

# Persist to config file
cat >> "$WG_CONF" << EOF

[Peer]
# ${PEER_NAME}
PublicKey = ${PEER_PUBKEY}
AllowedIPs = ${PEER_VPN_IP}/32
EOF

echo ""
echo "Peer '${PEER_NAME}' added. Client config to paste into WireGuard app:"
echo ""
echo "────────────────────────────────────────────"
echo "[Interface]"
echo "PrivateKey = <${PEER_NAME}-private-key>"
echo "Address = ${PEER_VPN_IP}/32"
echo "DNS = 1.1.1.1"
echo ""
echo "[Peer]"
echo "PublicKey = ${SERVER_PUB}"
echo "Endpoint = ${SERVER_IP}:51820"
echo "AllowedIPs = 10.0.0.0/24"
echo "PersistentKeepalive = 25"
echo "────────────────────────────────────────────"
