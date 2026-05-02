# QuantPipe — Claude Code Runbook

## Branch Workflow (automatic — no prompting needed)

**Always develop on `desktop-dev`.** At the start of any session, if the current branch is not `desktop-dev`, switch to it before doing any work:

```bash
git checkout desktop-dev
```

**Deploy = open a PR from `desktop-dev` to `main`.** Whenever the user asks to deploy, push to the server, or says anything like "ship it", "deploy", "push to prod":

```bash
git push origin desktop-dev
gh pr create --base main --head desktop-dev \
  --title "feat|fix|docs: <one-line summary of what this deploy contains>" \
  --body "## Summary
- <bullet per meaningful change>

## Test plan
- [ ] Dashboard loads at http://10.0.0.1:8501
- [ ] Pipeline runs cleanly (check ntfy or pipeline health page)

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
# Sourcery reviews the PR automatically (usually within seconds)
# Once reviewed, merge via GitHub UI or:
gh pr merge <number> --merge
```

After the PR merges, GitHub Actions pulls `main` and restarts the server automatically. Stay on `desktop-dev`.

**Prerequisite:** `gh` CLI must be installed and authenticated (`gh auth status`). It is already set up on this desktop — if running on a new machine, run `gh auth login` first.

**Never commit directly to main.** All commits go on `desktop-dev` first.

**Never merge `desktop-dev` → `main` directly.** Always go through a PR so Sourcery can review.

### PR title convention

Use a conventional commit prefix matching the primary change:

| Prefix | When |
|---|---|
| `feat:` | New feature or dashboard capability |
| `fix:` | Bug fix |
| `docs:` | README, CLAUDE.md, comments only |
| `refactor:` | Code restructuring, no behaviour change |
| `chore:` | Deps, config, tooling |

### Decision rules at a glance

| Situation | Action |
|---|---|
| Session starts, not on `desktop-dev` | `git checkout desktop-dev` |
| User says "commit" / makes code changes | Commit on `desktop-dev` |
| User says "deploy" / "push" / "ship" | Push `desktop-dev` → open PR → merge PR |
| After PR merged | Stay on `desktop-dev` — continue working |

---

## Server Access

```bash
# SSH
ssh -i ~/.ssh/quantpipe_server root@87.99.133.129

# Check all services
systemctl is-active \
  quantpipe-streamlit \
  quantpipe-pipeline.timer \
  quantpipe-rebalance.timer \
  quantpipe-backup.timer \
  quantpipe-ibgateway \
  quantpipe-xvfb \
  vault \
  vault-unseal \
  wg-quick@wg0 \
  fail2ban \
  actions.runner.micahabanschick-QuantPipe.quantpipe-server

# Restart dashboard
systemctl restart quantpipe-streamlit

# Trigger pipeline now (outside of timer)
systemctl start quantpipe-pipeline.service

# Trigger rebalance now (outside of timer)
systemctl start quantpipe-rebalance.service

# Watch logs
journalctl -u quantpipe-streamlit  -n 30 --no-pager
journalctl -u quantpipe-ibgateway  -n 30 --no-pager
tail -f /var/log/quantpipe/pipeline.log
tail -f /var/log/quantpipe/rebalance.log
tail -f /var/log/quantpipe/backup.log
```

Dashboard URL (WireGuard VPN required): http://10.0.0.1:8501

---

## CI/CD

Push to `main` → GitHub Actions (self-hosted runner on server) → runs `deploy/update.sh` → `git fetch && git reset --hard origin/main` + `uv sync` + restart Streamlit.

Workflow file: [.github/workflows/deploy.yml](.github/workflows/deploy.yml)

Runner service: `actions.runner.micahabanschick-QuantPipe.quantpipe-server`
Runner path: `/opt/quantpipe/actions-runner/`

---

## Server Schedule

| Timer | Schedule | Service it fires |
|---|---|---|
| `quantpipe-pipeline.timer` | Mon–Fri 21:30 UTC | `quantpipe-pipeline.service` — ingest + signals |
| `quantpipe-rebalance.timer` | Mon–Fri 22:30 UTC | `quantpipe-rebalance.service` — IBKR paper orders |
| `quantpipe-backup.timer` | Daily 02:00 UTC | `quantpipe-backup.service` — B2 sync |

Systemd unit files tracked in `deploy/systemd/`. Changes to unit files must be applied manually on the server (`systemctl daemon-reload`) — the CI/CD deploy does not auto-install them.

---

## WireGuard VPN

Desktop VPN is configured and auto-starts on boot (Windows service: `WireGuardTunnel$quantpipe`).

| | Value |
|---|---|
| Desktop IP | `10.0.0.3` |
| Server IP | `10.0.0.1` |
| Server endpoint | `87.99.133.129:51820` |
| Server pubkey | `bQ3IIr4ov8frIWdpKZjA8znrdhPoOb/wuMQQDat65RE=` |
| Desktop config | `C:\Users\micha\.wireguard\quantpipe.conf` |

> **Note:** Always use `wg show wg0` on the server to get the authoritative pubkey — the `server.pub` file may differ.

### Checking VPN status

```bash
# On server
wg show wg0

# On desktop (PowerShell)
Get-Service WireGuardTunnel$quantpipe
```

### Adding a new peer (new machine)

1. Generate keys on the new machine
2. Run `bash deploy/add_peer.sh <pubkey> <vpn-ip>` on the server
3. Import the WireGuard config on the new machine and activate

---

## IB Gateway (IBKR Paper Trading)

IB Gateway runs headlessly on the server via IBC + Xvfb. It logs into IBKR paper account **DUQ368627** automatically on boot.

```bash
# Check status
systemctl status quantpipe-ibgateway
systemctl status quantpipe-xvfb

# View IBC diagnostic log (most detailed)
ls /var/log/quantpipe/ibc-*.txt
tail -40 /var/log/quantpipe/ibc-3.23.0_GATEWAY-1037_$(date +%A).txt

# Restart gateway
systemctl restart quantpipe-xvfb quantpipe-ibgateway

# Verify API port is open
ss -tlnp | grep 4002
```

Key config: `/opt/ibc/config.ini`
- `IbLoginId` / `IbPassword` — IBKR credentials
- `TradingMode=paper`
- `OverrideTwsApiPort=4002`
- `AcceptIncomingConnectionAction=accept`
- `ExistingSessionDetectedAction=primaryoverride`

IBKR credentials are stored in Dashlane under **"QuantPipe — Server Secrets"**.

---

## HashiCorp Vault

Vault stores all QuantPipe secrets and auto-unseals on server reboot via `vault-unseal.service`.

```bash
# Check Vault status
VAULT_ADDR=http://127.0.0.1:8200 vault status

# Read a secret (as app)
VAULT_ADDR=http://127.0.0.1:8200 \
VAULT_TOKEN=$(grep VAULT_TOKEN /opt/quantpipe/.env | cut -d= -f2) \
vault kv get secret/quantpipe/config

# Manually unseal (if vault-unseal.service failed)
VAULT_ADDR=http://127.0.0.1:8200 vault operator unseal <unseal-key>

# Re-load secrets from .env into Vault (e.g. after adding new keys)
VAULT_ADDR=http://127.0.0.1:8200 \
VAULT_TOKEN=<root-token> \
bash /opt/quantpipe/deploy/vault/init_secrets.sh
```

Unseal key and root token are stored in Dashlane under **"QuantPipe — Server Secrets"**.

The app token (in `/opt/quantpipe/.env` as `VAULT_TOKEN`) has a 1-year TTL — renew before **2027-04-27** using the root token:

```bash
VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=<root-token> \
vault token create -policy=quantpipe -orphan -renewable=true -ttl=8760h
```

Auto-unseal key file: `/opt/vault/unseal.key` (root-only, `chmod 400`)

---

## Backblaze B2 Backup

Daily incremental backup of `data/` to the `quantpipe-backup` bucket via rclone.

```bash
# Run backup manually
systemctl start quantpipe-backup.service

# Check last backup
tail -20 /var/log/quantpipe/backup.log

# Verify B2 credentials
source /opt/quantpipe/.env
curl -s -u "$B2_ACCOUNT_ID:$B2_APPLICATION_KEY" \
  https://api.backblazeb2.com/b2api/v2/b2_authorize_account | python3 -m json.tool
```

B2 credentials are stored in Dashlane under **"QuantPipe — Server Secrets"**.

Script: `deploy/backup/backup.sh`
Restore script: `deploy/backup/restore.sh`

---

## ntfy Alerts

The pipeline and rebalance scripts send notifications to `ntfy.sh/quantpipe-micah`.

- Subscribe: open `https://ntfy.sh/quantpipe-micah` in a browser, or add the topic in the ntfy app
- No token required (public topic)
- Fires on: pipeline success, pipeline failure, pre-trade block, position drift

Test alert from server:
```bash
curl -d "test message" https://ntfy.sh/quantpipe-micah
```

---

## Key Commands (local dev)

```bash
uv run streamlit run app.py                    # run dashboard locally
uv run python orchestration/run_pipeline.py    # run full pipeline locally
uv run python orchestration/rebalance.py --broker ibkr --dry-run   # dry-run rebalance
uv run pytest tests/ -m "not network"          # run tests (no network)
```

---

## Secrets Reference

All sensitive credentials are stored in **Dashlane** under the note **"QuantPipe — Server Secrets"**. The note contains:

| Secret | Where used |
|---|---|
| Vault unseal key | Manual unseal after server reboot; also in `/opt/vault/unseal.key` |
| Vault root token | Vault admin operations only — not in any file |
| Vault app token | `/opt/quantpipe/.env` and desktop `.env` as `VAULT_TOKEN` |
| IBKR username + password | `/opt/ibc/config.ini` on server |
| IBKR paper account ID (`DUQ368627`) | Reference only |
| Backblaze B2 key ID + secret | `/opt/quantpipe/.env` and desktop `.env` |
| FRED API key | `/opt/quantpipe/.env` and desktop `.env` |
| Alpha Vantage API key | `/opt/quantpipe/.env` and desktop `.env` as `ALPHA_VANTAGE_API_KEY` — see Dashlane |
| BLS API key | Not yet registered — optional free key at data.bls.gov/registrationEngine |

Non-sensitive config (VPN IPs, ports, service names) lives in this runbook.
