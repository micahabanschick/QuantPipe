"""Nightly ingestion script — pulled by cron at 10pm ET on weekdays.

Runs incrementally: fetches the last LOOKBACK_DAYS window for each symbol
so any vendor revisions are picked up. A full historical backfill is a
separate one-shot operation (see backfill_history.py).

Exit codes: 0 = all symbols succeeded, 1 = partial failures, 2 = all failed.
"""

import logging
import sys

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError in log handlers)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import date, timedelta

from config.settings import INGESTION_LOOKBACK_DAYS, LOGS_DIR
from config.universes import CRYPTO_UNIVERSE, EQUITY_UNIVERSE
from data_adapters.ccxt_adapter import CCXTAdapter
from data_adapters.yfinance_adapter import YFinanceAdapter
from storage.parquet_store import write_bars
from storage.validators import format_results, has_failures, validate_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "ingest.log"),
    ],
)
log = logging.getLogger(__name__)


def _send_alert(message: str) -> None:
    """Send a Pushover / ntfy alert if configured, otherwise log only."""
    from config.settings import NTFY_TOPIC, PUSHOVER_TOKEN, PUSHOVER_USER

    if PUSHOVER_TOKEN and PUSHOVER_USER:
        try:
            import requests
            requests.post("https://api.pushover.net/1/messages.json", data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "QuantPipe Alert",
            }, timeout=10)
        except Exception as exc:
            log.warning(f"Pushover alert failed: {exc}")

    elif NTFY_TOPIC:
        try:
            import requests
            requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode(), timeout=10)
        except Exception as exc:
            log.warning(f"ntfy alert failed: {exc}")

    log.warning(f"ALERT: {message}")


def ingest_equities(start: date, end: date) -> tuple[int, int]:
    """Returns (succeeded, failed) counts."""
    adapter = YFinanceAdapter()
    succeeded, failed = 0, 0

    for symbol in EQUITY_UNIVERSE:
        try:
            df = adapter.get_bars(symbol, start, end)
            if df.is_empty():
                log.warning(f"[equity] {symbol}: empty response")
                failed += 1
                continue

            results = validate_ingestion(df)
            n_written = write_bars(df, "equity", symbol)

            if has_failures(results, "error"):
                _send_alert(f"Validation ERROR for equity/{symbol}:\n{format_results(results)}")
                failed += 1
            else:
                log.info(f"[equity] {symbol}: {n_written} rows written")
                succeeded += 1

        except Exception as exc:
            log.error(f"[equity] {symbol}: EXCEPTION — {exc}")
            failed += 1

    return succeeded, failed


def ingest_crypto(start: date, end: date) -> tuple[int, int]:
    """Returns (succeeded, failed) counts."""
    adapter = CCXTAdapter(exchange_id="kraken")
    succeeded, failed = 0, 0

    for symbol in CRYPTO_UNIVERSE:
        try:
            df = adapter.get_bars(symbol, start, end)
            if df.is_empty():
                log.warning(f"[crypto] {symbol}: empty response")
                failed += 1
                continue

            results = validate_ingestion(df)
            safe_sym = symbol.replace("/", "_")
            n_written = write_bars(df, "crypto", safe_sym)

            if has_failures(results, "error"):
                _send_alert(f"Validation ERROR for crypto/{symbol}:\n{format_results(results)}")
                failed += 1
            else:
                log.info(f"[crypto] {symbol}: {n_written} rows written")
                succeeded += 1

        except Exception as exc:
            log.error(f"[crypto] {symbol}: EXCEPTION — {exc}")
            failed += 1

    return succeeded, failed


def main() -> int:
    today = date.today()
    start = today - timedelta(days=INGESTION_LOOKBACK_DAYS)

    log.info(f"=== QuantPipe nightly ingestion | {today} | window: {start} → {today} ===")

    eq_ok, eq_fail = ingest_equities(start, today)
    cr_ok, cr_fail = ingest_crypto(start, today)

    total_ok = eq_ok + cr_ok
    total_fail = eq_fail + cr_fail
    total = total_ok + total_fail

    log.info(f"=== Ingestion complete: {total_ok}/{total} succeeded ({total_fail} failed) ===")

    if total_fail > 0:
        _send_alert(f"Ingestion finished with {total_fail}/{total} failures. Check logs.")

    if total_ok == 0:
        return 2
    if total_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
