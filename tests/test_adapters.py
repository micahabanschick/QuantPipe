"""Tests for data adapter interfaces.

These tests hit the live network and are marked @pytest.mark.network.
They are excluded from the fast CI suite by default.

Run all:
    uv run pytest tests/test_adapters.py -v
Run only network tests:
    uv run pytest tests/test_adapters.py -v -m network
Skip network tests:
    uv run pytest -m "not network"
"""

from datetime import date

import polars as pl
import pytest

from data_adapters.base import OHLCV_SCHEMA, validate_ohlcv
from data_adapters.yfinance_adapter import YFinanceAdapter


REQUIRED_COLS = list(OHLCV_SCHEMA.keys())
START = date(2024, 1, 2)
END = date(2024, 1, 31)

pytestmark = pytest.mark.network   # all tests in this file require live network


class TestYFinanceAdapter:
    def setup_method(self):
        self.adapter = YFinanceAdapter()

    def test_get_bars_returns_correct_schema(self):
        df = self.adapter.get_bars("SPY", START, END)
        assert not df.is_empty(), "Expected non-empty DataFrame for SPY"
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_get_bars_date_range(self):
        df = self.adapter.get_bars("SPY", START, END)
        assert df["date"].min() >= START
        assert df["date"].max() <= END

    def test_get_bars_no_nulls_in_prices(self):
        df = self.adapter.get_bars("SPY", START, END)
        for col in ["open", "high", "low", "close", "adj_close"]:
            assert df[col].is_null().sum() == 0, f"Nulls in {col}"

    def test_get_bars_symbol_column_correct(self):
        df = self.adapter.get_bars("QQQ", START, END)
        assert (df["symbol"] == "QQQ").all()

    def test_get_bars_invalid_freq_raises(self):
        with pytest.raises(ValueError, match="only supports"):
            self.adapter.get_bars("SPY", START, END, freq="5m")

    def test_get_bars_empty_for_invalid_symbol(self):
        df = self.adapter.get_bars("ZZZZNOTREAL99", START, END)
        assert df.is_empty() or len(df) == 0

    def test_validate_ohlcv_passes_for_valid_df(self):
        df = self.adapter.get_bars("SPY", START, END)
        validate_ohlcv(df, "SPY")   # should not raise

    def test_get_bars_batch_returns_multiple_symbols(self):
        df = self.adapter.get_bars_batch(["SPY", "QQQ"], START, END)
        symbols = df["symbol"].unique().to_list()
        assert "SPY" in symbols
        assert "QQQ" in symbols


class TestValidateOhlcv:
    def test_raises_on_missing_columns(self):
        df = pl.DataFrame({"date": [], "close": []})
        with pytest.raises(ValueError, match="missing columns"):
            validate_ohlcv(df, "test")
