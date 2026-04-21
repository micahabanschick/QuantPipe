"""yfinance adapter for equity and ETF daily bars.

Uses auto-adjusted prices for adj_close and also stores the raw close
so both are available in storage. Free tier — suitable for Phase 1.

Retry policy: up to 3 attempts with exponential backoff (2s, 4s) on any
transient failure. Failed symbols are logged to logs/dead_letters.log so
the health dashboard can surface them without scanning all pipeline output.
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

from config.settings import LOGS_DIR
from .base import OHLCV_SCHEMA, DataAdapter

log = logging.getLogger(__name__)

_DEAD_LETTER_LOG = LOGS_DIR / "dead_letters.log"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt


def _log_dead_letter(symbol: str, reason: str) -> None:
    """Append a failure record to the dead-letter log."""
    import datetime
    try:
        _DEAD_LETTER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEAD_LETTER_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
            f.write(f"{ts}\tyfinance\t{symbol}\t{reason}\n")
    except Exception:
        pass   # dead-letter logging must never crash the pipeline


class YFinanceAdapter:
    """Fetch OHLCV bars from Yahoo Finance via yfinance."""

    name = "yfinance"
    asset_class = "equity"

    def get_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        freq: str = "1d",
    ) -> pl.DataFrame:
        if freq not in ("1d", "1wk", "1mo"):
            raise ValueError(f"YFinanceAdapter only supports 1d/1wk/1mo, got {freq!r}")

        # yfinance end is exclusive — add one day so we always include `end`
        end_exclusive = end + timedelta(days=1)

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(
                    start=str(start),
                    end=str(end_exclusive),
                    interval=freq,
                    auto_adjust=False,
                    actions=True,
                )

                if raw.empty:
                    return pl.DataFrame(schema=OHLCV_SCHEMA)

                raw = raw.reset_index()
                raw.columns = [c.lower() for c in raw.columns]

                if "adj close" in raw.columns:
                    raw = raw.rename(columns={"adj close": "adj_close"})
                elif "adjclose" in raw.columns:
                    raw = raw.rename(columns={"adjclose": "adj_close"})
                else:
                    raw["adj_close"] = raw["close"]

                raw["symbol"] = symbol
                raw["date"] = raw["date"].dt.date

                df = pl.from_pandas(
                    raw[["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]]
                ).with_columns([
                    pl.col("date").cast(pl.Date),
                    pl.col("symbol").cast(pl.Utf8),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                    pl.col("adj_close").cast(pl.Float64),
                ])

                return df.sort("date")

            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log.warning(
                        f"[yfinance] {symbol} attempt {attempt}/{_MAX_RETRIES} failed: {exc}. "
                        f"Retrying in {delay:.0f}s."
                    )
                    time.sleep(delay)

        # All retries exhausted — log to dead-letter file and re-raise
        log.error(f"[yfinance] {symbol} failed after {_MAX_RETRIES} attempts: {last_exc}")
        _log_dead_letter(symbol, str(last_exc))
        raise RuntimeError(f"[yfinance] {symbol} permanently failed: {last_exc}") from last_exc

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
                log.error(f"[yfinance] batch: {sym} permanently failed — {exc}")
        return pl.concat(frames) if frames else pl.DataFrame(schema=OHLCV_SCHEMA)
