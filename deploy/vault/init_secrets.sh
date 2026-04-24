#!/usr/bin/env bash
# Load all QuantPipe secrets from .env into Vault
# Run once after: vault operator init && vault operator unseal
# Usage: VAULT_TOKEN=<root-token> bash init_secrets.sh
set -euo pipefail

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
: "${VAULT_TOKEN:?Set VAULT_TOKEN to your Vault root (or policy) token}"

APP_DIR="/opt/quantpipe"
ENV_FILE="$APP_DIR/.env"

[[ -f "$ENV_FILE" ]] || { echo ".env not found at $ENV_FILE"; exit 1; }

echo "[+] Enabling KV secrets engine..."
vault secrets enable -path=secret kv-v2 2>/dev/null || echo "    (already enabled)"

echo "[+] Creating quantpipe policy..."
vault policy write quantpipe - << 'EOF'
path "secret/data/quantpipe/*" {
  capabilities = ["read", "list"]
}
EOF

echo "[+] Writing secrets from .env to Vault..."
# Parse .env and write all non-empty, non-comment values
declare -A SECRETS
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    [[ -z "$val" ]] && continue
    SECRETS["$key"]="$val"
done < "$ENV_FILE"

if [[ ${#SECRETS[@]} -eq 0 ]]; then
    echo "[!] No secrets found in .env — fill it in first"
    exit 1
fi

# Build vault kv put arguments
KV_ARGS=()
for key in "${!SECRETS[@]}"; do
    KV_ARGS+=("${key}=${SECRETS[$key]}")
done
vault kv put secret/quantpipe/config "${KV_ARGS[@]}"

echo "[+] Creating app token with quantpipe policy..."
APP_TOKEN=$(vault token create -policy=quantpipe -orphan -renewable=true \
    -ttl=8760h -format=json | python3 -c "import sys,json; print(json.load(sys.stdin)['auth']['client_token'])")

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Vault initialized with QuantPipe secrets"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  App token (add to .env as VAULT_TOKEN):"
echo "  VAULT_TOKEN=$APP_TOKEN"
echo "  VAULT_ADDR=http://10.0.0.1:8200"
echo ""
echo "  Verify: vault kv get secret/quantpipe/config"
echo "════════════════════════════════════════════════════════"
