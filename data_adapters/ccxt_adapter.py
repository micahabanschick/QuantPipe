"""CCXT adapter for crypto OHLCV daily bars.

Supports any CCXT-compatible exchange. Defaults to Kraken.
Symbol format: "BTC/USDT" (CCXT unified format).
"""

from datetime import date, datetime, timezone

import ccxt
import polars as pl

from .base import OHLCV_SCHEMA, DataAdapter


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

        all_ohlcv = []
        fetch_since = since_ms
        while True:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=500)
            if not batch:
                break
            all_ohlcv.extend(batch)
            if batch[-1][0] >= end_ms:
                break
            fetch_since = batch[-1][0] + 1

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
                "adj_close": float(c),  # crypto has no adjustments
            }
            for ts, o, h, l, c, v in all_ohlcv
        ]

        df = pl.DataFrame(rows).with_columns([
            pl.col("date").cast(pl.Date),
        ])

        # Filter to requested date range (CCXT may return extra rows)
        df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        return df.sort("date")

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
                print(f"[ccxt/{self.exchange_id}] WARNING: {sym} failed — {exc}")
        return pl.concat(frames) if frames else pl.DataFrame(schema=OHLCV_SCHEMA)
