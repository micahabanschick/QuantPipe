"""Tests for the cross-sectional momentum signal module."""

from datetime import date, timedelta

import polars as pl
import pytest

from signals.momentum import (
    cross_sectional_momentum,
    get_monthly_rebalance_dates,
    momentum_weights,
)


def _make_features(n_symbols: int = 10, n_days: int = 300) -> pl.DataFrame:
    """Generate synthetic features with known momentum ordering."""
    start = date(2022, 1, 3)
    frames = []
    for i in range(n_symbols):
        dates = [start + timedelta(days=d) for d in range(n_days)]
        # Symbol i has momentum proportional to i (symbol 9 = highest)
        mom_values = [float(i) / n_symbols + 0.01 * d / n_days for d in range(n_days)]
        vol_values = [0.15 + 0.01 * i] * n_days
        frames.append(pl.DataFrame({
            "date": dates,
            "symbol": [f"SYM{i:02d}"] * n_days,
            "momentum_12m_1m": mom_values,
            "realized_vol_21d": vol_values,
        }).with_columns(pl.col("date").cast(pl.Date)))
    return pl.concat(frames)


def _rebal_dates(features: pl.DataFrame, n: int = 3) -> list[date]:
    all_dates = sorted(features["date"].unique().to_list())
    step = len(all_dates) // (n + 1)
    return [all_dates[step * (i + 1)] for i in range(n)]


class TestCrossSectionalMomentum:
    def setup_method(self):
        self.features = _make_features()
        self.rebal_dates = _rebal_dates(self.features)

    def test_returns_expected_columns(self):
        result = cross_sectional_momentum(self.features, self.rebal_dates, top_n=3)
        for col in ["date", "symbol", "momentum_12m_1m", "rank", "selected", "rebalance_date"]:
            assert col in result.columns

    def test_top_n_selected_per_rebalance(self):
        top_n = 4
        result = cross_sectional_momentum(self.features, self.rebal_dates, top_n=top_n)
        counts = result.filter(pl.col("selected")).group_by("rebalance_date").len()
        assert all(counts["len"] == top_n)

    def test_rank_1_has_highest_momentum(self):
        result = cross_sectional_momentum(self.features, self.rebal_dates, top_n=3)
        for rebal_date in result["rebalance_date"].unique().to_list():
            day = result.filter(pl.col("rebalance_date") == rebal_date)
            rank1_mom = day.filter(pl.col("rank") == 1)["momentum_12m_1m"][0]
            max_mom = day["momentum_12m_1m"].max()
            assert abs(rank1_mom - max_mom) < 1e-9

    def test_missing_momentum_column_raises(self):
        bad_features = self.features.drop("momentum_12m_1m")
        with pytest.raises(ValueError, match="momentum_12m_1m"):
            cross_sectional_momentum(bad_features, self.rebal_dates)

    def test_empty_result_when_dates_before_any_data(self):
        # Dates before the feature data starts should produce no results
        ancient_dates = [date(2000, 1, 1)]
        result = cross_sectional_momentum(self.features, ancient_dates, top_n=3)
        assert result.is_empty()


class TestMomentumWeights:
    def setup_method(self):
        features = _make_features()
        rebal_dates = _rebal_dates(features)
        self.signal = cross_sectional_momentum(features, rebal_dates, top_n=5)
        self.features = features

    def test_equal_weights_sum_to_one(self):
        weights = momentum_weights(self.signal, weight_scheme="equal")
        sums = weights.group_by("rebalance_date").agg(pl.col("weight").sum())
        for s in sums["weight"].to_list():
            assert abs(s - 1.0) < 1e-9

    def test_equal_weight_is_one_over_n(self):
        top_n = 5
        weights = momentum_weights(self.signal, weight_scheme="equal")
        assert all(abs(w - 1.0 / top_n) < 1e-9 for w in weights["weight"].to_list())

    def test_vol_scaled_weights_sum_to_one(self):
        vol_df = self.features.select(["date", "symbol", "realized_vol_21d"])
        weights = momentum_weights(self.signal, weight_scheme="vol_scaled", vol_series=vol_df)
        sums = weights.group_by("rebalance_date").agg(pl.col("weight").sum())
        for s in sums["weight"].to_list():
            assert abs(s - 1.0) < 1e-6

    def test_vol_scaled_raises_without_vol_series(self):
        with pytest.raises(ValueError, match="vol_series required"):
            momentum_weights(self.signal, weight_scheme="vol_scaled")

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown weight_scheme"):
            momentum_weights(self.signal, weight_scheme="mystery")


class TestGetMonthlyRebalanceDates:
    def _trading_days(self, start: date, n: int) -> list[date]:
        days = []
        d = start
        for _ in range(n):
            days.append(d)
            d += timedelta(days=1)
        return days

    def test_first_day_of_each_month(self):
        trading_days = self._trading_days(date(2023, 1, 1), 365)
        rebal = get_monthly_rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), trading_days)
        months = [(d.year, d.month) for d in rebal]
        assert len(months) == len(set(months)), "Duplicate months in rebalance dates"

    def test_respects_start_end_bounds(self):
        trading_days = self._trading_days(date(2023, 1, 1), 365)
        start = date(2023, 3, 1)
        end = date(2023, 9, 30)
        rebal = get_monthly_rebalance_dates(start, end, trading_days)
        assert all(start <= d <= end for d in rebal)
