"""Central settings — loads from Vault (production) or .env (local dev).

Priority: Vault > environment variable > default.
Set VAULT_ADDR + VAULT_TOKEN in the environment to enable Vault.
Without those, behaviour is identical to the original .env-only approach.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Vault client (lazy, cached) ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _vault_secrets() -> dict[str, str]:
    """Fetch all secrets from Vault KV in one call and cache them.

    Returns an empty dict if Vault is not configured or unreachable,
    so every caller falls back to os.getenv transparently.
    """
    addr = os.getenv("VAULT_ADDR", "")
    token = os.getenv("VAULT_TOKEN", "")
    if not (addr and token):
        return {}
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            f"{addr}/v1/secret/data/quantpipe/config",
            headers={"X-Vault-Token": token},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read())
        return payload["data"]["data"]
    except Exception:
        return {}


def _get(key: str, default: str = "") -> str:
    """Read a secret: Vault → env var → default."""
    return _vault_secrets().get(key) or os.getenv(key, default)


# ── Alternative data ───────────────────────────────────────────────────────────
FRED_API_KEY: str = _get("FRED_API_KEY")   # Federal Reserve Economic Data

# ── Data providers ─────────────────────────────────────────────────────────────
EODHD_API_KEY: str          = _get("EODHD_API_KEY")
ALPACA_API_KEY: str         = _get("ALPACA_API_KEY")
ALPHA_VANTAGE_API_KEY: str  = _get("ALPHA_VANTAGE_API_KEY")
ALPACA_SECRET_KEY: str = _get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL: str = _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Crypto exchanges ───────────────────────────────────────────────────────────
KRAKEN_API_KEY: str = _get("KRAKEN_API_KEY")
KRAKEN_SECRET: str = _get("KRAKEN_SECRET")
KRAKEN_API_SECRET = KRAKEN_SECRET  # alias — rebalance.py imports this name
COINBASE_API_KEY: str = _get("COINBASE_API_KEY")
COINBASE_SECRET: str = _get("COINBASE_SECRET")

# ── IBKR ───────────────────────────────────────────────────────────────────────
IBKR_HOST: str = _get("IBKR_HOST", "127.0.0.1")
IBKR_PORT: int = int(_get("IBKR_PORT", "7497"))
IBKR_CLIENT_ID: int = int(_get("IBKR_CLIENT_ID", "1"))
IBKR_PAPER: bool = _get("IBKR_PAPER", "true").lower() in ("1", "true", "yes")

# ── Alerting ───────────────────────────────────────────────────────────────────
PUSHOVER_TOKEN: str = _get("PUSHOVER_TOKEN")
PUSHOVER_USER: str = _get("PUSHOVER_USER")
NTFY_TOPIC: str = _get("NTFY_TOPIC")

# ── Backblaze B2 backup ────────────────────────────────────────────────────────
B2_ACCOUNT_ID: str = _get("B2_ACCOUNT_ID")
B2_APPLICATION_KEY: str = _get("B2_APPLICATION_KEY")
B2_BUCKET: str = _get("B2_BUCKET", "quantpipe-backup")

# ── Pipeline defaults ──────────────────────────────────────────────────────────
DEFAULT_EQUITY_FREQ: str = "1d"
DEFAULT_CRYPTO_FREQ: str = "1d"
INGESTION_LOOKBACK_DAYS: int = 30
HISTORICAL_BACKFILL_DAYS: int = 365 * 7
