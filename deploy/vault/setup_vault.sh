#!/usr/bin/env bash
# Install and configure HashiCorp Vault
set -euo pipefail

VAULT_VERSION="1.17.2"
VAULT_DIR="/opt/vault"
VAULT_DATA="$VAULT_DIR/data"
VAULT_CONFIG="$VAULT_DIR/vault.hcl"

[[ $EUID -eq 0 ]] || { echo "Must be run as root"; exit 1; }

echo "[+] Installing Vault $VAULT_VERSION..."
ARCH=$(dpkg --print-architecture)
curl -fsSLo /tmp/vault.zip \
    "https://releases.hashicorp.com/vault/${VAULT_VERSION}/vault_${VAULT_VERSION}_linux_${ARCH}.zip"
unzip -o /tmp/vault.zip -d /usr/local/bin vault
chmod +x /usr/local/bin/vault
rm /tmp/vault.zip

# Allow vault to use mlock without root
setcap cap_ipc_lock=+ep /usr/local/bin/vault

echo "[+] Creating vault user and directories..."
if ! id vault &>/dev/null; then
    useradd --system --shell /bin/false vault
fi
mkdir -p "$VAULT_DATA"
chown -R vault:vault "$VAULT_DIR"

echo "[+] Installing Vault config..."
cp /opt/quantpipe/deploy/vault/vault.hcl "$VAULT_CONFIG"
chown vault:vault "$VAULT_CONFIG"

echo "[+] Installing Vault systemd service..."
cat > /etc/systemd/system/vault.service << EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://developer.hashicorp.com/vault/docs
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=vault
Group=vault
ExecStart=/usr/local/bin/vault server -config=${VAULT_CONFIG}
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
KillSignal=SIGINT
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
LimitMEMLOCK=infinity
StandardOutput=append:/var/log/quantpipe/vault.log
StandardError=append:/var/log/quantpipe/vault.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vault

echo ""
echo "[!] Vault is running. IMPORTANT: Initialize it now:"
echo "    export VAULT_ADDR='http://127.0.0.1:8200'"
echo "    vault operator init -key-shares=1 -key-threshold=1"
echo ""
echo "    Save the UNSEAL KEY and ROOT TOKEN somewhere safe (offline)."
echo "    Then unseal: vault operator unseal <unseal-key>"
echo "    Then run:    bash /opt/quantpipe/deploy/vault/init_secrets.sh"
