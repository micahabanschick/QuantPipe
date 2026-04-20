"""One-shot historical backfill — run once to seed the data store.

Pulls 7 years of daily bars for all universes. Takes ~5-15 minutes
depending on rate limits. Safe to re-run: write_bars is idempotent.

Usage (from project root):
    uv run python orchestration/backfill_history.py
    uv run python orchestration/backfill_history.py --asset-class equity
    uv run python orchestration/backfill_history.py --start 2018-01-01
"""

import argparse
import logging
import sys
from datetime import date

from config.settings import HISTORICAL_BACKFILL_DAYS, LOGS_DIR
from config.universes import CRYPTO_UNIVERSE, EQUITY_UNIVERSE
from data_adapters.ccxt_adapter import CCXTAdapter
from data_adapters.yfinance_adapter import YFinanceAdapter
from storage.parquet_store import write_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "backfill.log"),
    ],
)
log = logging.getLogger(__name__)


def backfill_equities(start: date, end: date) -> None:
    adapter = YFinanceAdapter()
    log.info(f"Backfilling {len(EQUITY_UNIVERSE)} equity symbols from {start} to {end}")
    for i, symbol in enumerate(EQUITY_UNIVERSE, 1):
        try:
            df = adapter.get_bars(symbol, start, end)
            n = write_bars(df, "equity", symbol)
            log.info(f"  [{i}/{len(EQUITY_UNIVERSE)}] {symbol}: {n} rows")
        except Exception as exc:
            log.error(f"  [{i}/{len(EQUITY_UNIVERSE)}] {symbol}: FAILED — {exc}")


def backfill_crypto(start: date, end: date) -> None:
    adapter = CCXTAdapter(exchange_id="kraken")
    log.info(f"Backfilling {len(CRYPTO_UNIVERSE)} crypto symbols from {start} to {end}")
    for i, symbol in enumerate(CRYPTO_UNIVERSE, 1):
        try:
            df = adapter.get_bars(symbol, start, end)
            n = write_bars(df, "crypto", symbol.replace("/", "_"))
            log.info(f"  [{i}/{len(CRYPTO_UNIVERSE)}] {symbol}: {n} rows")
        except Exception as exc:
            log.error(f"  [{i}/{len(CRYPTO_UNIVERSE)}] {symbol}: FAILED — {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical data backfill")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--asset-class", choices=["equity", "crypto", "all"], default="all")
    args = parser.parse_args()

    from datetime import timedelta
    start = args.start or (args.end - timedelta(days=HISTORICAL_BACKFILL_DAYS))
    end = args.end

    log.info(f"=== Backfill | {start} → {end} | asset-class={args.asset_class} ===")

    if args.asset_class in ("equity", "all"):
        backfill_equities(start, end)
    if args.asset_class in ("crypto", "all"):
        backfill_crypto(start, end)

    log.info("=== Backfill complete ===")


if __name__ == "__main__":
    main()
