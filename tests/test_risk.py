"""Tests for the risk management module."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from risk.engine import (
    RiskLimits,
    ExposureReport,
    CheckResult,
    compute_exposures,
    historical_var,
    pre_trade_check,
    generate_risk_report,
    EQUITY_SECTOR_MAP,
)
from risk.scenarios import SCENARIOS, apply_scenario, run_all_scenarios


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_returns(n_assets: int = 5, n_days: int = 300) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.0005, 0.01, size=(n_days, n_assets))


def _make_prices_df(symbols: list[str], n_days: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    start = date(2020, 1, 2)
    rows = []
    for sym in symbols:
        price = 100.0
        for d in range(n_days):
            price *= 1 + rng.normal(0.0003, 0.01)
            rows.append({"date": start + timedelta(days=d), "symbol": sym, "adj_close": round(price, 4)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ── compute_exposures ─────────────────────────────────────────────────────────

class TestComputeExposures:
    def test_empty_weights_returns_zeros(self):
        report = compute_exposures({})
        assert report.gross_exposure == 0.0
        assert report.net_exposure == 0.0
        assert report.n_positions == 0

    def test_gross_net_long_only(self):
        weights = {"XLK": 0.30, "XLF": 0.30, "XLV": 0.40}
        r = compute_exposures(weights)
        assert abs(r.gross_exposure - 1.0) < 1e-6
        assert abs(r.net_exposure - 1.0) < 1e-6

    def test_gross_net_with_short(self):
        weights = {"XLK": 0.60, "XLF": -0.20}
        r = compute_exposures(weights)
        assert abs(r.gross_exposure - 0.80) < 1e-6
        assert abs(r.net_exposure - 0.40) < 1e-6

    def test_sector_aggregation(self):
        weights = {"XLK": 0.30, "XLF": 0.30, "XLV": 0.40}
        r = compute_exposures(weights)
        assert "Technology" in r.sector_exposures
        assert "Financials" in r.sector_exposures
        assert abs(r.sector_exposures["Technology"] - 0.30) < 1e-6

    def test_unknown_symbol_goes_to_other(self):
        weights = {"FAKE": 0.50, "XLK": 0.50}
        r = compute_exposures(weights)
        assert "Other" in r.sector_exposures

    def test_top5_concentration(self):
        weights = {f"SYM{i}": 0.1 for i in range(10)}
        r = compute_exposures(weights, sector_map={f"SYM{i}": "Other" for i in range(10)})
        assert abs(r.top_5_concentration - 0.5) < 1e-6

    def test_largest_position(self):
        weights = {"XLK": 0.10, "XLF": 0.60, "XLV": 0.30}
        r = compute_exposures(weights)
        assert r.largest_position[0] == "XLF"
        assert abs(r.largest_position[1] - 0.60) < 1e-6

    def test_n_positions_ignores_dust(self):
        weights = {"XLK": 0.50, "XLF": 1e-9}
        r = compute_exposures(weights)
        assert r.n_positions == 1

    def test_as_of_date_default_is_today(self):
        r = compute_exposures({"XLK": 1.0})
        assert r.as_of == date.today()

    def test_custom_as_of(self):
        r = compute_exposures({"XLK": 1.0}, as_of=date(2023, 6, 1))
        assert r.as_of == date(2023, 6, 1)


# ── historical_var ────────────────────────────────────────────────────────────

class TestHistoricalVar:
    def setup_method(self):
        self.ret = _make_returns(n_assets=5, n_days=300)
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_returns_positive_float(self):
        v = historical_var(self.ret, self.weights)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_95_var_less_than_99_var(self):
        v95 = historical_var(self.ret, self.weights, confidence=0.95)
        v99 = historical_var(self.ret, self.weights, confidence=0.99)
        assert v95 <= v99

    def test_empty_returns_zero(self):
        v = historical_var(np.array([]).reshape(0, 5), self.weights)
        assert v == 0.0

    def test_lookback_uses_tail(self):
        long_ret = np.vstack([
            np.full((200, 5), -0.50),
            _make_returns(n_assets=5, n_days=100),
        ])
        v_full = historical_var(long_ret, self.weights, lookback=300)
        v_recent = historical_var(long_ret, self.weights, lookback=100)
        assert v_full != v_recent

    def test_concentrated_portfolio_higher_var(self):
        equal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        concentrated = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        rng = np.random.default_rng(5)
        ret = rng.normal(0, 0.02, (300, 5))
        v_eq = historical_var(ret, equal)
        v_conc = historical_var(ret, concentrated)
        assert v_conc >= v_eq


# ── pre_trade_check ───────────────────────────────────────────────────────────

class TestPreTradeCheck:
    def test_passes_clean_portfolio(self):
        # 10 equal weights: each 10%, top-5 = 50% (below 80% cap)
        weights = {f"ETF{i}": 0.10 for i in range(10)}
        limits = RiskLimits(max_sector=2.0)
        result = pre_trade_check(weights, limits, sector_map={f"ETF{i}": "Other" for i in range(10)})
        assert result.passed
        assert len(result.violations) == 0

    def test_fails_position_cap(self):
        weights = {"XLK": 0.60, "XLF": 0.40}
        limits = RiskLimits(max_position=0.40)
        result = pre_trade_check(weights, limits)
        assert not result.passed
        assert any("Position cap" in v for v in result.violations)

    def test_fails_sector_cap(self):
        weights = {"XLK": 0.35, "XLF": 0.35, "XLV": 0.30}
        limits = RiskLimits(max_sector=0.30)
        result = pre_trade_check(weights, limits)
        assert not result.passed
        assert any("Sector cap" in v for v in result.violations)

    def test_fails_gross_cap(self):
        weights = {"XLK": 0.60, "XLF": 0.60}
        limits = RiskLimits(max_gross=1.00, max_position=1.0, max_sector=2.0)
        result = pre_trade_check(weights, limits)
        assert not result.passed
        assert any("Gross cap" in v for v in result.violations)

    def test_fails_net_cap(self):
        weights = {"XLK": 0.60, "XLF": 0.60}
        limits = RiskLimits(max_net=0.50, max_position=1.0, max_sector=2.0, max_gross=2.0)
        result = pre_trade_check(weights, limits)
        assert not result.passed
        assert any("Net cap" in v for v in result.violations)

    def test_fails_top5_concentration(self):
        weights = {f"ETF{i}": 0.20 for i in range(5)}
        limits = RiskLimits(max_top5_concentration=0.50, max_sector=2.0)
        result = pre_trade_check(weights, limits, sector_map={f"ETF{i}": "Other" for i in range(5)})
        assert not result.passed
        assert any("Concentration" in v for v in result.violations)

    def test_var_check_with_returns(self):
        symbols = ["XLK", "XLF", "XLV", "XLY", "XLP"]
        weights = {s: 0.2 for s in symbols}
        ret = _make_returns(n_assets=5, n_days=300) * 5  # inflate vol to breach limit
        limits = RiskLimits(var_limit_pct=0.001)
        result = pre_trade_check(weights, limits, returns_matrix=ret, symbol_order=symbols)
        assert not result.passed
        assert any("VaR cap" in v for v in result.violations)

    def test_var_check_skipped_without_returns(self):
        # 10 equal weights pass all non-VaR checks; VaR check must be skipped without returns
        weights = {f"ETF{i}": 0.10 for i in range(10)}
        limits = RiskLimits(var_limit_pct=0.001, max_sector=2.0)
        result = pre_trade_check(weights, limits, sector_map={f"ETF{i}": "Other" for i in range(10)})
        assert result.passed

    def test_str_passed(self):
        r = CheckResult(passed=True)
        assert "PASSED" in str(r)

    def test_str_failed(self):
        r = CheckResult(passed=False, violations=["Position cap: XLK 60.0% > limit 40.0%"])
        assert "FAILED" in str(r)
        assert "Position cap" in str(r)


# ── generate_risk_report ──────────────────────────────────────────────────────

class TestGenerateRiskReport:
    def test_returns_risk_report(self):
        from risk.engine import RiskReport
        symbols = list(EQUITY_SECTOR_MAP.keys())[:5]
        weights = {s: 0.2 for s in symbols}
        prices = _make_prices_df(symbols)
        report = generate_risk_report(weights, prices)
        assert isinstance(report, RiskReport)

    def test_var_is_positive(self):
        symbols = list(EQUITY_SECTOR_MAP.keys())[:5]
        weights = {s: 0.2 for s in symbols}
        prices = _make_prices_df(symbols)
        report = generate_risk_report(weights, prices)
        assert report.var_1d_95 >= 0.0
        assert report.var_1d_99 >= report.var_1d_95

    def test_stress_results_passed_through(self):
        symbols = list(EQUITY_SECTOR_MAP.keys())[:5]
        weights = {s: 0.2 for s in symbols}
        prices = _make_prices_df(symbols)
        stress = {"test_scenario": -0.10}
        report = generate_risk_report(weights, prices, stress_results=stress)
        assert report.stress_results == {"test_scenario": -0.10}

    def test_empty_prices_still_returns_report(self):
        weights = {"XLK": 0.5, "XLF": 0.5}
        empty_prices = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "adj_close": pl.Float64})
        report = generate_risk_report(weights, empty_prices)
        assert report.var_1d_95 == 0.0


# ── scenarios ─────────────────────────────────────────────────────────────────

class TestScenarios:
    def test_known_scenarios_exist(self):
        for name in ("2008_GFC", "2020_COVID", "2022_RATES", "2000_DOTCOM"):
            assert name in SCENARIOS

    def test_apply_scenario_long_only_loss(self):
        weights = {"SPY": 1.0}
        pnl = apply_scenario(weights, "2008_GFC")
        assert pnl < 0

    def test_apply_scenario_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            apply_scenario({"SPY": 1.0}, "1970_UNKNOWN")

    def test_apply_scenario_unknown_symbol_is_zero(self):
        weights = {"UNKNOWN_SYM": 1.0}
        pnl = apply_scenario(weights, "2008_GFC")
        assert pnl == 0.0

    def test_apply_scenario_short_position_gains_in_crash(self):
        weights = {"SPY": -1.0}
        pnl = apply_scenario(weights, "2008_GFC")
        assert pnl > 0

    def test_run_all_scenarios_returns_all(self):
        weights = {"SPY": 1.0}
        results = run_all_scenarios(weights)
        assert set(results.keys()) == set(SCENARIOS.keys())

    def test_run_all_scenarios_sorted_worst_first(self):
        weights = {"SPY": 1.0}
        results = run_all_scenarios(weights)
        values = list(results.values())
        assert values == sorted(values)

    def test_bonds_positive_in_2008(self):
        weights = {"TLT": 1.0}
        pnl = apply_scenario(weights, "2008_GFC")
        assert pnl > 0

    def test_custom_scenarios_override(self):
        custom = {"MY_CRASH": {"SPY": -0.50}}
        pnl = apply_scenario({"SPY": 1.0}, "MY_CRASH", scenarios=custom)
        assert abs(pnl - (-0.50)) < 1e-9
