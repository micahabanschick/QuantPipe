"""yfinance adapter for equity and ETF daily bars.

Uses auto-adjusted prices for adj_close and also stores the raw close
so both are available in storage. Free tier — suitable for Phase 1.
"""

from datetime import date, timedelta

import polars as pl
import yfinance as yf

from .base import OHLCV_SCHEMA, DataAdapter


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

        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=str(start),
            end=str(end_exclusive),
            interval=freq,
            auto_adjust=False,  # we handle adjustments ourselves
            actions=True,
        )

        if raw.empty:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]

        # adj_close comes back as "adj close" from yfinance
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
                print(f"[yfinance] WARNING: {sym} failed — {exc}")
        return pl.concat(frames) if frames else pl.DataFrame(schema=OHLCV_SCHEMA)
