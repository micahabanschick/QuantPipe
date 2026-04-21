"""CCXT adapter for crypto OHLCV daily bars.

Supports any CCXT-compatible exchange. Defaults to Kraken.
Symbol format: "BTC/USDT" (CCXT unified format).

Retry policy: up to 3 attempts with exponential backoff on any transient
exchange error. Failed symbols are logged to logs/dead_letters.log.

Pagination guard: the `while` loop has a hard max-iteration cap to prevent
infinite looping when an exchange returns the same cursor repeatedly.
"""

import logging
import time
from datetime import date, datetime, timezone

import ccxt
import polars as pl

from config.settings import LOGS_DIR
from .base import OHLCV_SCHEMA, DataAdapter

log = logging.getLogger(__name__)

_DEAD_LETTER_LOG = LOGS_DIR / "dead_letters.log"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_MAX_PAGES = 2_000   # hard cap: 2000 pages × 500 bars = 1 000 000 bars max


def _log_dead_letter(symbol: str, reason: str) -> None:
    try:
        _DEAD_LETTER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEAD_LETTER_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.utcnow().isoformat(timespec="seconds")
            f.write(f"{ts}\tccxt\t{symbol}\t{reason}\n")
    except Exception:
        pass


class CCXTAdapter:
    """Fetch OHLCV bars from a crypto exchange via CCXT."""

    name = "ccxt"
    asset_class = "crypto"

    def __init__(self, exchange_id: str = "kraken", api_key: str = "", secret: str = "") -> None:
        exchange_class = getattr(ccxt, exchange_id)
        kwargs: dict = {"enableRateLimit": True}
        if api_key:
            kwargs["apiKey"] = api_key
        if secret:
            kwargs["secret"] = secret
        self.exchange: ccxt.Exchange = exchange_class(kwargs)
        self.exchange_id = exchange_id

    def get_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        freq: str = "1d",
    ) -> pl.DataFrame:
        if not self.exchange.has.get("fetchOHLCV"):
            raise RuntimeError(f"{self.exchange_id} does not support fetchOHLCV")

        timeframe_map = {"1d": "1d", "1h": "1h", "1w": "1w"}
        if freq not in timeframe_map:
            raise ValueError(f"CCXTAdapter: unsupported freq {freq!r}")
        timeframe = timeframe_map[freq]

        since_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                all_ohlcv = []
                fetch_since = since_ms
                pages = 0
                prev_last_ts: int | None = None

                while pages < _MAX_PAGES:
                    batch = self.exchange.fetch_ohlcv(
                        symbol, timeframe=timeframe, since=fetch_since, limit=500
                    )
                    if not batch:
                        break

                    last_ts = batch[-1][0]

                    # Guard against exchange returning the same cursor repeatedly
                    if last_ts == prev_last_ts:
                        log.warning(
                            f"[ccxt/{self.exchange_id}] {symbol}: pagination cursor stuck at "
                            f"{last_ts} — stopping early after {pages} pages."
                        )
                        break
                    prev_last_ts = last_ts

                    all_ohlcv.extend(batch)
                    pages += 1

                    if last_ts >= end_ms:
                        break
                    fetch_since = last_ts + 1

                if pages >= _MAX_PAGES:
                    log.warning(
                        f"[ccxt/{self.exchange_id}] {symbol}: hit {_MAX_PAGES}-page limit — "
                        "data may be incomplete."
                    )

                if not all_ohlcv:
                    return pl.DataFrame(schema=OHLCV_SCHEMA)

                rows = [
                    {
                        "date": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date(),
                        "symbol": symbol,
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                        "adj_close": float(c),
                    }
                    for ts, o, h, l, c, v in all_ohlcv
                ]

                df = pl.DataFrame(rows).with_columns([pl.col("date").cast(pl.Date)])
                df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
                return df.sort("date")

            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        f"[ccxt/{self.exchange_id}] {symbol} attempt {attempt}/{_MAX_RETRIES} "
                        f"failed: {exc}. Retrying in {delay:.0f}s."
                    )
                    time.sleep(delay)

        log.error(f"[ccxt/{self.exchange_id}] {symbol} failed after {_MAX_RETRIES} attempts: {last_exc}")
        _log_dead_letter(symbol, str(last_exc))
        raise RuntimeError(
            f"[ccxt/{self.exchange_id}] {symbol} permanently failed: {last_exc}"
        ) from last_exc

    def get_bars_batch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: str = "1d",
    ) -> pl.DataFrame:
        frames = []
        for sym in symbols:
            try:
                df = self.get_bars(sym, start, end, freq)
                if not df.is_empty():
                    frames.append(df)
            except Exception as exc:
                log.error(f"[ccxt/{self.exchange_id}] batch: {sym} permanently failed — {exc}")
        return pl.concat(frames) if frames else pl.DataFrame(schema=OHLCV_SCHEMA)
