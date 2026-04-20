"""Tests for portfolio construction and covariance estimation."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from portfolio.covariance import compute_returns, ledoit_wolf_cov, sample_cov, cov_to_corr
from portfolio.optimizer import construct_portfolio, PortfolioConstraints


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_prices(n_symbols: int = 5, n_days: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    start = date(2021, 1, 4)
    rows = []
    for i in range(n_symbols):
        sym = f"ETF{i:02d}"
        price = 100.0
        for d in range(n_days):
            dt = start + timedelta(days=d)
            price *= (1 + rng.normal(0.0003, 0.01))
            rows.append({"date": dt, "symbol": sym, "adj_close": round(price, 4)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _make_signals(symbols: list[str], rebal_dates: list[date]) -> pl.DataFrame:
    rows = []
    for rd in rebal_dates:
        for sym in symbols:
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": 1.0, "selected": True})
    return pl.DataFrame(rows).with_columns(pl.col("rebalance_date").cast(pl.Date))


PRICES = _make_prices()
SYMBOLS = PRICES["symbol"].unique().sort().to_list()
REBAL_DATES = [date(2021, 6, 1), date(2021, 9, 1), date(2021, 12, 1)]


# ── compute_returns ───────────────────────────────────────────────────────────

class TestComputeReturns:
    def test_shape(self):
        ret, syms = compute_returns(PRICES)
        n_dates = PRICES["date"].n_unique()
        assert ret.shape == (n_dates - 1, len(syms))

    def test_symbols_match(self):
        _, syms = compute_returns(PRICES)
        assert set(syms) == set(SYMBOLS)

    def test_no_nan(self):
        ret, _ = compute_returns(PRICES)
        assert not np.isnan(ret).any()

    def test_simple_returns_different_from_log(self):
        log_ret, _ = compute_returns(PRICES, method="log")
        sim_ret, _ = compute_returns(PRICES, method="simple")
        assert not np.allclose(log_ret, sim_ret)


# ── ledoit_wolf_cov ───────────────────────────────────────────────────────────

class TestLedoitWolfCov:
    def test_shape_and_symbols(self):
        cov, syms = ledoit_wolf_cov(PRICES)
        n = len(SYMBOLS)
        assert cov.shape == (n, n)
        assert set(syms) == set(SYMBOLS)

    def test_positive_definite(self):
        cov, _ = ledoit_wolf_cov(PRICES)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues > 0).all(), "Covariance matrix is not positive definite"

    def test_symmetric(self):
        cov, _ = ledoit_wolf_cov(PRICES)
        assert np.allclose(cov, cov.T)

    def test_annualise_scales_by_252(self):
        cov_ann, _ = ledoit_wolf_cov(PRICES, annualize=True)
        cov_daily, _ = ledoit_wolf_cov(PRICES, annualize=False)
        assert np.allclose(cov_ann, cov_daily * 252)

    def test_lookback_respected(self):
        cov_short, _ = ledoit_wolf_cov(PRICES, lookback_days=60)
        cov_long, _ = ledoit_wolf_cov(PRICES, lookback_days=250)
        assert not np.allclose(cov_short, cov_long)


class TestSampleCov:
    def test_shape(self):
        cov, syms = sample_cov(PRICES)
        assert cov.shape == (len(SYMBOLS), len(SYMBOLS))

    def test_symmetric(self):
        cov, _ = sample_cov(PRICES)
        assert np.allclose(cov, cov.T)


class TestCovToCorr:
    def test_diagonal_is_one(self):
        cov, _ = ledoit_wolf_cov(PRICES)
        corr = cov_to_corr(cov)
        assert np.allclose(np.diag(corr), 1.0)

    def test_values_in_range(self):
        cov, _ = ledoit_wolf_cov(PRICES)
        corr = cov_to_corr(cov)
        assert (corr >= -1.0 - 1e-9).all() and (corr <= 1.0 + 1e-9).all()


# ── construct_portfolio ───────────────────────────────────────────────────────

class TestConstructPortfolio:
    def setup_method(self):
        self.cov, self.sym_order = ledoit_wolf_cov(PRICES)
        self.signals = _make_signals(self.sym_order, REBAL_DATES)
        self.expected_returns = np.array([0.10, 0.12, 0.08, 0.11, 0.09])

    def _weights_sum(self, df: pl.DataFrame) -> list[float]:
        return df.group_by("rebalance_date").agg(pl.col("weight").sum())["weight"].to_list()

    def test_equal_weights_sum_to_one(self):
        df = construct_portfolio(self.signals, method="equal")
        for s in self._weights_sum(df):
            assert abs(s - 1.0) < 1e-9

    def test_equal_weight_value(self):
        df = construct_portfolio(self.signals, method="equal")
        expected = 1.0 / len(self.sym_order)
        for w in df["weight"].to_list():
            assert abs(w - expected) < 1e-9

    def test_vol_scaled_sum_to_one(self):
        df = construct_portfolio(self.signals, self.cov, self.sym_order, method="vol_scaled")
        for s in self._weights_sum(df):
            assert abs(s - 1.0) < 1e-6

    def test_vol_scaled_requires_cov(self):
        with pytest.raises(ValueError, match="vol_scaled requires"):
            construct_portfolio(self.signals, method="vol_scaled")

    def test_mean_variance_sum_to_one(self):
        pytest.importorskip("cvxpy")
        df = construct_portfolio(
            self.signals, self.cov, self.sym_order,
            method="mean_variance",
            expected_returns=self.expected_returns,
        )
        for s in self._weights_sum(df):
            assert abs(s - 1.0) < 1e-4

    def test_mean_variance_requires_expected_returns(self):
        with pytest.raises(ValueError, match="expected_returns"):
            construct_portfolio(self.signals, self.cov, self.sym_order, method="mean_variance")

    def test_min_variance_sum_to_one(self):
        pytest.importorskip("pypfopt")
        df = construct_portfolio(self.signals, self.cov, self.sym_order, method="min_variance")
        for s in self._weights_sum(df):
            assert abs(s - 1.0) < 1e-4

    def test_max_sharpe_sum_to_one(self):
        pytest.importorskip("pypfopt")
        df = construct_portfolio(
            self.signals, self.cov, self.sym_order,
            method="max_sharpe",
            expected_returns=self.expected_returns,
        )
        for s in self._weights_sum(df):
            assert abs(s - 1.0) < 1e-4

    def test_position_cap_respected(self):
        constraints = PortfolioConstraints(max_position=0.25)
        df = construct_portfolio(
            self.signals, self.cov, self.sym_order,
            method="vol_scaled",
            constraints=constraints,
        )
        assert all(w <= 0.25 + 1e-9 for w in df["weight"].to_list())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            construct_portfolio(self.signals, method="magic")

    def test_output_columns(self):
        df = construct_portfolio(self.signals, method="equal")
        assert set(df.columns) >= {"rebalance_date", "symbol", "weight"}
