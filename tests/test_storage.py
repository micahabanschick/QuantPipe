"""Tests for the Parquet storage layer.

Uses a temporary directory so tests don't touch real data.

Run:
    uv run pytest tests/test_storage.py -v
"""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from data_adapters.base import OHLCV_SCHEMA


def _make_sample_df(symbol: str = "SPY", n_rows: int = 5) -> pl.DataFrame:
    start_date = date(2024, 1, 2)
    dates = [date(2024, 1, 2 + i) for i in range(n_rows)]
    return pl.DataFrame({
        "date": dates,
        "symbol": [symbol] * n_rows,
        "open": [100.0 + i for i in range(n_rows)],
        "high": [101.0 + i for i in range(n_rows)],
        "low":  [99.0 + i for i in range(n_rows)],
        "close": [100.5 + i for i in range(n_rows)],
        "volume": [1_000_000.0] * n_rows,
        "adj_close": [100.5 + i for i in range(n_rows)],
    }).with_columns(pl.col("date").cast(pl.Date))


class TestParquetStore:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_dir = Path(self._tmpdir.name)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def _patch_data_dir(self):
        """Context manager patching DATA_DIR in storage.parquet_store."""
        import storage.parquet_store as ps
        return patch.object(ps, "DATA_DIR", self._data_dir)

    def test_write_and_read_roundtrip(self):
        import storage.parquet_store as ps
        with self._patch_data_dir():
            df = _make_sample_df("SPY")
            n = ps.write_bars(df, "equity", "SPY")
            assert n == len(df)

            loaded = ps.load_bars("SPY", date(2024, 1, 2), date(2024, 1, 10), "equity")
            assert not loaded.is_empty()
            assert len(loaded) == len(df)

    def test_write_is_idempotent(self):
        import storage.parquet_store as ps
        with self._patch_data_dir():
            df = _make_sample_df("SPY")
            ps.write_bars(df, "equity", "SPY")
            ps.write_bars(df, "equity", "SPY")   # write same data twice

            loaded = ps.load_bars("SPY", date(2024, 1, 2), date(2024, 1, 10), "equity")
            assert len(loaded) == len(df)   # no duplicates

    def test_write_empty_df_returns_zero(self):
        import storage.parquet_store as ps
        with self._patch_data_dir():
            empty = pl.DataFrame(schema=OHLCV_SCHEMA)
            n = ps.write_bars(empty, "equity", "SPY")
            assert n == 0

    def test_list_symbols(self):
        import storage.parquet_store as ps
        with self._patch_data_dir():
            ps.write_bars(_make_sample_df("SPY"), "equity", "SPY")
            ps.write_bars(_make_sample_df("QQQ"), "equity", "QQQ")

            symbols = ps.list_symbols("equity")
            assert "SPY" in symbols
            assert "QQQ" in symbols
