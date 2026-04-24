# HashiCorp Vault config — listens on localhost only (VPN-accessible via port forward)
# Secrets never leave the server unencrypted

storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = true   # Safe — localhost only; TLS handled by WireGuard tunnel
}

# Also listen on VPN interface so you can reach it from your laptop over WireGuard
listener "tcp" {
  address     = "10.0.0.1:8200"
  tls_disable = true
}

api_addr     = "http://10.0.0.1:8200"
cluster_addr = "http://127.0.0.1:8201"

ui = true

# Seal type — auto-unseal on restart (acceptable for single-node personal use)
# For higher security, remove this block and manually unseal after each restart
disable_mlock = true
