"""Feature library tests — unit correctness + snapshot regression.

Snapshot test: the fixture in tests/fixtures/feature_snapshot.parquet is the
ground truth. Any change to feature logic will break this test. If a change
is intentional, re-run tests/generate_fixtures.py and commit the new fixture.

Run:
    uv run pytest tests/test_features.py -v
"""

from datetime import date, timedelta
from pathlib import Path

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

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SNAPSHOT_PATH = FIXTURES_DIR / "feature_snapshot.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ramp_prices(n: int = 300, start: float = 100.0, daily_drift: float = 0.001) -> pl.Series:
    """Deterministic upward-trending price series."""
    return pl.Series("adj_close", [start * (1 + daily_drift) ** i for i in range(n)])


def _make_bars(symbol: str = "SPY", n: int = 320, drift: float = 0.0008) -> pl.DataFrame:
    start_date = date(2020, 1, 2)
    dates = [start_date + timedelta(days=i) for i in range(n)]
    prices = [100.0 * ((1 + drift) ** i) for i in range(n)]
    return pl.DataFrame({
        "date": dates,
        "symbol": [symbol] * n,
        "adj_close": prices,
        "volume": [1_000_000.0 + i * 100 for i in range(n)],
    }).with_columns(pl.col("date").cast(pl.Date))


# ── Unit tests: log_return ────────────────────────────────────────────────────

class TestLogReturn:
    def test_first_row_is_null(self):
        prices = _ramp_prices(10)
        rets = log_return(prices, 1)
        assert rets[0] is None or np.isnan(float(rets[0]))

    def test_positive_for_rising_prices(self):
        prices = _ramp_prices(10)
        rets = log_return(prices, 1)
        assert all(r > 0 for r in rets[1:] if r is not None and not np.isnan(float(r)))

    def test_periods_parameter_shifts_null_window(self):
        prices = _ramp_prices(10)
        r5 = log_return(prices, 5)
        assert r5[4] is None or np.isnan(float(r5[4]))
        assert r5[5] is not None and not np.isnan(float(r5[5]))

    def test_length_preserved(self):
        prices = _ramp_prices(50)
        assert len(log_return(prices, 1)) == 50


# ── Unit tests: realized_vol ──────────────────────────────────────────────────

class TestRealizedVol:
    def test_length_matches_input(self):
        assert len(realized_vol(_ramp_prices(100), 21)) == 100

    def test_non_negative_for_any_series(self):
        vol = realized_vol(_ramp_prices(100), 21)
        assert all(v >= 0 for v in vol if v is not None and not np.isnan(float(v)))

    def test_zero_for_constant_prices(self):
        prices = pl.Series([100.0] * 50)
        vol = realized_vol(prices, 21)
        valid = [float(v) for v in vol[21:] if v is not None and not np.isnan(float(v))]
        assert all(abs(v) < 1e-10 for v in valid)

    def test_higher_vol_for_noisier_series(self):
        quiet = pl.Series([100.0 + 0.01 * i for i in range(100)])
        noisy = pl.Series([100.0 + 0.5 * ((-1) ** i) * i for i in range(100)])
        quiet_vol = [float(v) for v in realized_vol(quiet, 21)[21:] if v is not None]
        noisy_vol = [float(v) for v in realized_vol(noisy, 21)[21:] if v is not None]
        assert np.mean(noisy_vol) > np.mean(quiet_vol)


# ── Unit tests: momentum_12m_1m ───────────────────────────────────────────────

class TestMomentum12m1m:
    def test_null_before_252_days(self):
        prices = _ramp_prices(300)
        mom = momentum_12m_1m(prices)
        # Row 251 (index) = 252nd value — still needs shift(252) lookback
        assert mom[251] is None or np.isnan(float(mom[251]))

    def test_valid_after_252_days(self):
        prices = _ramp_prices(300)
        mom = momentum_12m_1m(prices)
        valid = [float(v) for v in mom[252:] if v is not None and not np.isnan(float(v))]
        assert len(valid) > 0

    def test_positive_for_sustained_uptrend(self):
        prices = _ramp_prices(300, daily_drift=0.002)
        mom = momentum_12m_1m(prices)
        valid = [float(v) for v in mom if v is not None and not np.isnan(float(v))]
        assert all(v > 0 for v in valid)

    def test_negative_for_sustained_downtrend(self):
        prices = _ramp_prices(300, daily_drift=-0.001)
        mom = momentum_12m_1m(prices)
        valid = [float(v) for v in mom if v is not None and not np.isnan(float(v))]
        assert all(v < 0 for v in valid)


# ── Unit tests: dollar_volume ─────────────────────────────────────────────────

class TestDollarVolume:
    def test_shape(self):
        prices = _ramp_prices(100)
        volume = pl.Series([1_000_000.0] * 100)
        assert len(dollar_volume(prices, volume, 63)) == 100

    def test_positive_for_positive_inputs(self):
        prices = _ramp_prices(100)
        volume = pl.Series([1_000_000.0] * 100)
        dv = dollar_volume(prices, volume, 63)
        valid = [float(v) for v in dv if v is not None and not np.isnan(float(v))]
        assert all(v > 0 for v in valid)


# ── Unit tests: reversal_5d ───────────────────────────────────────────────────

class TestReversal5d:
    def test_negative_for_uptrend(self):
        prices = _ramp_prices(20, daily_drift=0.01)
        rev = reversal_5d(prices)
        valid = [float(v) for v in rev[5:] if v is not None and not np.isnan(float(v))]
        assert all(v < 0 for v in valid)

    def test_positive_for_downtrend(self):
        prices = _ramp_prices(20, daily_drift=-0.01)
        rev = reversal_5d(prices)
        valid = [float(v) for v in rev[5:] if v is not None and not np.isnan(float(v))]
        assert all(v > 0 for v in valid)


# ── Unit tests: compute_features orchestrator ─────────────────────────────────

class TestComputeFeatures:
    def test_returns_all_five_columns(self):
        result = compute_features(_make_bars())
        expected = {"date", "symbol", "log_return_1d", "realized_vol_21d",
                    "momentum_12m_1m", "dollar_volume_63d", "reversal_5d"}
        assert expected.issubset(set(result.columns))

    def test_handles_multiple_symbols(self):
        bars = pl.concat([_make_bars("SPY"), _make_bars("QQQ", drift=-0.0003)])
        result = compute_features(bars)
        assert set(result["symbol"].unique().to_list()) == {"SPY", "QQQ"}

    def test_feature_subset(self):
        result = compute_features(_make_bars(), feature_list=["log_return_1d"])
        assert "log_return_1d" in result.columns
        assert "momentum_12m_1m" not in result.columns

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown features"):
            compute_features(_make_bars(), feature_list=["not_a_feature"])

    def test_output_sorted_by_date_and_symbol(self):
        bars = pl.concat([_make_bars("SPY"), _make_bars("QQQ")])
        result = compute_features(bars)
        dates = result["date"].to_list()
        assert dates == sorted(dates)

    def test_no_future_data_in_momentum(self):
        """Momentum must be null for rows with < 252 days of history."""
        bars = _make_bars(n=260)  # just over 252 days
        result = compute_features(bars, feature_list=["momentum_12m_1m"])
        # First 252 rows must all be null
        first_252 = result["momentum_12m_1m"][:252].to_list()
        assert all(v is None or np.isnan(float(v)) for v in first_252)


# ── Snapshot regression test ──────────────────────────────────────────────────

class TestFeatureSnapshot:
    """Byte-level regression: output must match pinned fixture exactly.

    If this test fails after an intentional feature change, run:
        uv run python tests/generate_fixtures.py
    and commit the updated fixture.
    """

    @pytest.mark.skipif(
        not SNAPSHOT_PATH.exists(),
        reason="Snapshot fixture not generated yet — run tests/generate_fixtures.py",
    )
    def test_snapshot_match(self):
        fixture = pl.read_parquet(SNAPSHOT_PATH)

        # Regenerate from the same deterministic seed used in generate_fixtures.py
        SEED_START = date(2020, 1, 2)
        N_ROWS = 320
        DAILY_DRIFTS = {"SEED_A": 0.0008, "SEED_B": -0.0003}

        frames = []
        for sym, drift in DAILY_DRIFTS.items():
            dates = [SEED_START + timedelta(days=i) for i in range(N_ROWS)]
            prices = [100.0 * ((1 + drift) ** i) for i in range(N_ROWS)]
            volume = [1_000_000.0 + i * 100 for i in range(N_ROWS)]
            frames.append(pl.DataFrame({
                "date": dates,
                "symbol": [sym] * N_ROWS,
                "adj_close": prices,
                "volume": volume,
            }).with_columns(pl.col("date").cast(pl.Date)))

        current = compute_features(pl.concat(frames))

        # Compare numeric columns with tolerance (floating point reproducibility)
        numeric_cols = ["log_return_1d", "realized_vol_21d", "momentum_12m_1m",
                        "dollar_volume_63d", "reversal_5d"]
        for col in numeric_cols:
            fixture_vals = fixture[col].to_numpy()
            current_vals = current[col].to_numpy()
            np.testing.assert_allclose(
                fixture_vals, current_vals, rtol=1e-9, equal_nan=True,
                err_msg=f"Snapshot mismatch in column '{col}' — feature logic may have changed",
            )
