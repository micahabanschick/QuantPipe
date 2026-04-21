"""Central settings loaded from environment variables and .env file.

Copy .env.example to .env and fill in your values. Never commit .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data providers ─────────────────────────────────────────────────────────────
EODHD_API_KEY: str = os.getenv("EODHD_API_KEY", "")
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ── Crypto exchanges ───────────────────────────────────────────────────────────
KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_SECRET: str = os.getenv("KRAKEN_SECRET", "")
KRAKEN_API_SECRET = KRAKEN_SECRET  # alias — rebalance.py imports this name
COINBASE_API_KEY: str = os.getenv("COINBASE_API_KEY", "")
COINBASE_SECRET: str = os.getenv("COINBASE_SECRET", "")

# ── IBKR ───────────────────────────────────────────────────────────────────────
IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT: int = int(os.getenv("IBKR_PORT", "7497"))   # TWS paper=7497, live=7496; Gateway paper=4002, live=4001
IBKR_CLIENT_ID: int = int(os.getenv("IBKR_CLIENT_ID", "1"))
IBKR_PAPER: bool = os.getenv("IBKR_PAPER", "true").lower() in ("1", "true", "yes")

# ── Alerting ───────────────────────────────────────────────────────────────────
PUSHOVER_TOKEN: str = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER: str = os.getenv("PUSHOVER_USER", "")
NTFY_TOPIC: str = os.getenv("NTFY_TOPIC", "")  # ntfy.sh alternative

# ── Pipeline defaults ──────────────────────────────────────────────────────────
DEFAULT_EQUITY_FREQ: str = "1d"
DEFAULT_CRYPTO_FREQ: str = "1d"
INGESTION_LOOKBACK_DAYS: int = 30   # overlap window for revisions on nightly runs
HISTORICAL_BACKFILL_DAYS: int = 365 * 7  # 7 years for initial backfill
