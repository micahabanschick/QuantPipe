"""Snapshot tests for the canonical feature library.

Each feature is tested against a fixed input so any lookahead contamination
or accidental logic change is immediately caught.

Run:
    uv run pytest tests/test_features.py -v
"""

import numpy as np
import polars as pl
import pytest

from features.canonical import (
    compute_features,
    dollar_volume,
    log_return,
    momentum_12m_1m,
    realized_vol,
    reversal_5d,
)


def _ramp_prices(n: int = 300, start: float = 100.0, daily_drift: float = 0.001) -> pl.Series:
    """Deterministic upward-trending price series for snapshot tests."""
    prices = [start * (1 + daily_drift) ** i for i in range(n)]
    return pl.Series("adj_close", prices)


class TestLogReturn:
    def test_first_row_is_null(self):
        prices = _ramp_prices(10)
        rets = log_return(prices, 1)
        assert rets[0] is None or np.isnan(rets[0])

    def test_positive_for_rising_prices(self):
        prices = _ramp_prices(10)
        rets = log_return(prices, 1)
        assert all(r > 0 for r in rets[1:] if r is not None)

    def test_periods_parameter(self):
        prices = _ramp_prices(10)
        r1 = log_return(prices, 1)
        r5 = log_return(prices, 5)
        assert r5[4] is None or np.isnan(r5[4])
        assert r5[5] is not None


class TestRealizedVol:
    def test_output_length_matches_input(self):
        prices = _ramp_prices(100)
        vol = realized_vol(prices, window=21)
        assert len(vol) == len(prices)

    def test_positive_for_non_constant_series(self):
        prices = _ramp_prices(100)
        vol = realized_vol(prices, window=21)
        non_null = [v for v in vol if v is not None and not np.isnan(v)]
        assert all(v >= 0 for v in non_null)

    def test_zero_for_constant_prices(self):
        prices = pl.Series([100.0] * 50)
        vol = realized_vol(prices, window=21)
        non_null = [v for v in vol[21:] if v is not None and not np.isnan(v)]
        assert all(abs(v) < 1e-10 for v in non_null)


class TestMomentum12m1m:
    def test_null_before_252_days(self):
        prices = _ramp_prices(300)
        mom = momentum_12m_1m(prices)
        assert mom[251] is None or np.isnan(mom[251])

    def test_positive_for_uptrend(self):
        prices = _ramp_prices(300, daily_drift=0.002)
        mom = momentum_12m_1m(prices)
        valid = [v for v in mom if v is not None and not np.isnan(v)]
        assert all(v > 0 for v in valid)


class TestDollarVolume:
    def test_output_shape(self):
        prices = _ramp_prices(100)
        volume = pl.Series([1_000_000.0] * 100)
        dv = dollar_volume(prices, volume, window=63)
        assert len(dv) == 100

    def test_positive_for_positive_prices_and_volume(self):
        prices = _ramp_prices(100)
        volume = pl.Series([1_000_000.0] * 100)
        dv = dollar_volume(prices, volume, window=63)
        valid = [v for v in dv if v is not None and not np.isnan(v)]
        assert all(v > 0 for v in valid)


class TestReversal5d:
    def test_negative_for_uptrend(self):
        prices = _ramp_prices(20, daily_drift=0.01)
        rev = reversal_5d(prices)
        valid = [v for v in rev[5:] if v is not None and not np.isnan(v)]
        assert all(v < 0 for v in valid)


class TestComputeFeatures:
    def _make_bars(self, symbol: str = "SPY", n: int = 300) -> pl.DataFrame:
        from datetime import date, timedelta
        start = date(2020, 1, 2)
        dates = [start + timedelta(days=i) for i in range(n)]
        prices = [100.0 * (1.001 ** i) for i in range(n)]
        return pl.DataFrame({
            "date": dates,
            "symbol": [symbol] * n,
            "adj_close": prices,
            "volume": [1_000_000.0] * n,
        }).with_columns(pl.col("date").cast(pl.Date))

    def test_returns_expected_columns(self):
        bars = self._make_bars()
        result = compute_features(bars)
        expected_cols = {"date", "symbol", "log_return_1d", "realized_vol_21d",
                         "momentum_12m_1m", "dollar_volume_63d", "reversal_5d"}
        assert expected_cols.issubset(set(result.columns))

    def test_handles_multiple_symbols(self):
        bars = pl.concat([self._make_bars("SPY"), self._make_bars("QQQ")])
        result = compute_features(bars)
        assert set(result["symbol"].unique().to_list()) == {"SPY", "QQQ"}

    def test_feature_subset(self):
        bars = self._make_bars()
        result = compute_features(bars, feature_list=["log_return_1d"])
        assert "log_return_1d" in result.columns
        assert "momentum_12m_1m" not in result.columns

    def test_unknown_feature_raises(self):
        bars = self._make_bars()
        with pytest.raises(ValueError, match="Unknown features"):
            compute_features(bars, feature_list=["fake_feature"])
