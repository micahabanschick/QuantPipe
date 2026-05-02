"""Pull quarterly earnings surprise data for sector ETF proxy stocks.

Fetches historical EPS actual vs estimate for representative holdings of each
sector ETF and caches the results in data/alt/earnings/{symbol}.parquet.

Earnings are quarterly so data is stale only after 90 days — the pipeline
skips a symbol if its cache is fresh, keeping daily API usage minimal.

The earnings_surprise_drift strategy reads this output to generate signals.

Usage (standalone):
    uv run python orchestration/pull_earnings.py

Pipeline integration:
    Called by run_pipeline.py when ALPHA_VANTAGE_API_KEY is set.
    Best-effort — failures do not abort the pipeline.
"""

import logging
import time
from datetime import date, datetime
from pathlib import Path

import polars as pl

from config.settings import DATA_DIR, ALPHA_VANTAGE_API_KEY

log = logging.getLogger(__name__)

# ── Sector proxy map ───────────────────────────────────────────────────────────
# 3 liquid, high-coverage stocks per sector ETF.
# These are used as earnings proxies for the entire sector.
# Update annually as index compositions change.
SECTOR_PROXY_MAP: dict[str, list[str]] = {
    "XLK":  ["AAPL", "MSFT", "NVDA"],   # Technology
    "XLE":  ["XOM",  "CVX",  "COP"],    # Energy
    "XLF":  ["JPM",  "BAC",  "WFC"],    # Financials
    "XLU":  ["NEE",  "DUK",  "SO"],     # Utilities
    "XLI":  ["CAT",  "DE",   "HON"],    # Industrials
    "XLV":  ["JNJ",  "UNH",  "PFE"],    # Health Care
    "XLP":  ["PG",   "KO",   "PEP"],    # Consumer Staples
    "XLC":  ["META", "GOOGL","VZ"],      # Communication Services
    "XLY":  ["AMZN", "TSLA", "HD"],     # Consumer Discretionary
    "XLRE": ["AMT",  "PLD",  "EQIX"],   # Real Estate
}

EARNINGS_DIR = DATA_DIR / "alt" / "earnings"
CACHE_DAYS   = 90    # re-fetch only when data is older than this
RATE_LIMIT_S = 13    # seconds between requests to stay under 5/min free tier


def _cache_path(symbol: str) -> Path:
    return EARNINGS_DIR / f"{symbol.lower()}.parquet"


def _is_fresh(symbol: str) -> bool:
    """Return True if the cached file is younger than CACHE_DAYS."""
    p = _cache_path(symbol)
    if not p.exists():
        return False
    age_days = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
    return age_days < CACHE_DAYS


def _fetch_and_cache(symbol: str, adapter) -> bool:
    """Fetch earnings for one symbol and write to parquet. Returns success."""
    try:
        df = adapter.get_earnings(symbol)
        if df.is_empty():
            log.warning("pull_earnings: no data for %s", symbol)
            return False
        p = _cache_path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
        log.info("pull_earnings: cached %d rows for %s", len(df), symbol)
        return True
    except Exception as exc:
        log.warning("pull_earnings: failed for %s: %s", symbol, exc)
        return False


def main() -> int:
    """Fetch stale earnings data for all sector proxy stocks.

    Returns 0 on full success, 1 if any fetches failed.
    """
    if not ALPHA_VANTAGE_API_KEY:
        log.info("pull_earnings: ALPHA_VANTAGE_API_KEY not set — skipping")
        return 0

    from data_adapters.alphavantage_adapter import AlphaVantageAdapter
    adapter = AlphaVantageAdapter(ALPHA_VANTAGE_API_KEY)

    # Collect all unique proxy symbols across all sectors
    all_symbols: list[str] = sorted({
        sym for proxies in SECTOR_PROXY_MAP.values() for sym in proxies
    })

    stale   = [s for s in all_symbols if not _is_fresh(s)]
    fresh   = len(all_symbols) - len(stale)
    failures = 0

    log.info(
        "pull_earnings: %d symbols total — %d fresh, %d to fetch",
        len(all_symbols), fresh, len(stale),
    )

    for i, sym in enumerate(stale):
        ok = _fetch_and_cache(sym, adapter)
        if not ok:
            failures += 1
        # Rate-limit between calls (free tier: ~5 req/min)
        if i < len(stale) - 1:
            time.sleep(RATE_LIMIT_S)

    if failures:
        log.warning("pull_earnings: %d/%d fetches failed", failures, len(stale))
        return 1

    log.info("pull_earnings: complete — %d fetched, %d already fresh", len(stale), fresh)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
