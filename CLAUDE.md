# QuantPipe — Claude Code Runbook

## Branch Workflow (automatic — no prompting needed)

**Always develop on `desktop-dev`.** At the start of any session, if the current branch is not `desktop-dev`, switch to it before doing any work:

```bash
git checkout desktop-dev
```

**Deploy = merge to main then switch back.** Whenever the user asks to deploy, push to the server, or says anything like "ship it", "deploy", "push to prod":

```bash
git checkout main
git merge desktop-dev
git push origin main          # triggers GitHub Actions → server pulls main + restarts
git checkout desktop-dev      # always return here after deploying
```

**Never commit directly to main.** All commits go on `desktop-dev` first.

**Never leave the session on `main`.** After any deploy, switch back to `desktop-dev` immediately.

### Decision rules at a glance

| Situation | Action |
|---|---|
| Session starts, not on `desktop-dev` | `git checkout desktop-dev` |
| User says "commit" / makes code changes | Commit on `desktop-dev` |
| User says "deploy" / "push" / "ship" | Merge → push main → switch back to `desktop-dev` |
| Finished deploy | Already back on `desktop-dev` — continue working |

---

## Server Access

```bash
# SSH
ssh -i ~/.ssh/quantpipe_server root@87.99.133.129

# Check services
systemctl is-active quantpipe-streamlit quantpipe-pipeline.timer wg-quick@wg0 fail2ban

# Restart dashboard
systemctl restart quantpipe-streamlit

# Trigger pipeline now
systemctl start quantpipe-pipeline.service

# Watch logs
journalctl -u quantpipe-streamlit -n 30 --no-pager
```

Dashboard URL (VPN required): http://10.0.0.1:8501

---

## WireGuard VPN Setup (deferred — do when ready)

### Step 1 — Generate your desktop keys

**Option A — WireGuard GUI (no extra tools needed):**
1. Open the WireGuard app
2. Click the dropdown arrow next to **Add Tunnel** → **Create from scratch**
3. The config editor opens with a pre-generated key pair:
   - `PrivateKey = <your-private-key>` — in the `[Interface]` block
   - `Public key: <your-public-key>` — shown in the window title/header
4. Copy both values, then **Cancel** (Claude will write the config file correctly)

**Option B — Command line (Git Bash):**
```bash
"/c/Program Files/WireGuard/wg.exe" genkey | tee ~/desktop_wg_private.key | \
  "/c/Program Files/WireGuard/wg.exe" pubkey > ~/desktop_wg_public.key
cat ~/desktop_wg_private.key   # private key
cat ~/desktop_wg_public.key    # public key
```

### Step 2 — Tell Claude your keys

Paste both keys into the chat. Claude will:
1. Write `C:\Users\micha\.wireguard\quantpipe.conf` with your private key
2. SSH to the server and register your public key as peer `10.0.0.3`
3. Prompt you to import the config in WireGuard and activate it
4. Run `setup_autoconnect.ps1` instructions for auto-connect at login

### Reference (server side)
- Server WireGuard pubkey: `CIZfR+iULCMmJXX/A+rd0BLrZCAFJ16DnTr2C/LYBDk=`
- Server endpoint: `87.99.133.129:51820`
- Desktop VPN IP: `10.0.0.3`
- Server VPN IP: `10.0.0.1`

---

## CI/CD

Push to `main` → GitHub Actions (self-hosted runner on server) → runs `deploy/update.sh` → pulls latest + restarts Streamlit.

Workflow file: [.github/workflows/deploy.yml](.github/workflows/deploy.yml)

---

## Key Commands

```bash
uv run streamlit run app.py              # run locally
uv run python orchestration/run_pipeline.py   # run pipeline locally
uv run pytest tests/ -m "not network"   # run tests
```
