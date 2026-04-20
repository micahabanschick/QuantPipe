"""Adapter protocol shared by all data providers.

Every adapter must implement get_bars() and return a DataFrame that
conforms to OHLCV_SCHEMA. Downstream code depends only on this protocol
so providers can be swapped without touching any other module.
"""

from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl

# Canonical column schema for OHLCV bars
OHLCV_SCHEMA: dict[str, type] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "adj_close": pl.Float64,
}

# Named tuple alias for single-row type hints in tests
OHLCVRow = dict[str, object]


@runtime_checkable
class DataAdapter(Protocol):
    """Protocol every data adapter must satisfy."""

    name: str
    asset_class: str  # "equity" | "crypto" | "futures" | "options"

    def get_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        freq: str = "1d",
    ) -> pl.DataFrame:
        """Return OHLCV bars for *symbol* in the closed interval [start, end].

        Returned DataFrame must have columns matching OHLCV_SCHEMA.
        Raises ValueError for unsupported freq values.
        Raises RuntimeError on provider API errors.
        """
        ...

    def get_bars_batch(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: str = "1d",
    ) -> pl.DataFrame:
        """Fetch multiple symbols and return a single concatenated DataFrame."""
        ...


def validate_ohlcv(df: pl.DataFrame, source: str = "") -> None:
    """Raise ValueError if df is missing required OHLCV columns."""
    missing = set(OHLCV_SCHEMA.keys()) - set(df.columns)
    if missing:
        tag = f" ({source})" if source else ""
        raise ValueError(f"DataFrame{tag} missing columns: {missing}")
