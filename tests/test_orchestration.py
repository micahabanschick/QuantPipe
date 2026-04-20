"""Tests for orchestration helpers — signal generation and pipeline utilities.

These are unit tests for the pure functions; they do not hit the network
or write to the real data directory.
"""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_prices(symbols: list[str], n_days: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(12)
    start = date(2019, 1, 2)
    rows = []
    for sym in symbols:
        price = 100.0
        for d in range(n_days):
            price *= 1 + rng.normal(0.0003, 0.01)
            rows.append({"date": start + timedelta(days=d), "symbol": sym, "adj_close": round(price, 4)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _make_features(symbols: list[str], n_days: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(34)
    start = date(2019, 1, 2)
    rows = []
    for i, sym in enumerate(symbols):
        for d in range(n_days):
            rows.append({
                "date": start + timedelta(days=d),
                "symbol": sym,
                "momentum_12m_1m": float(i) / len(symbols) + rng.normal(0, 0.01),
                "realized_vol_21d": 0.15 + rng.normal(0, 0.01),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


SYMBOLS_5 = ["SPY", "QQQ", "IWM", "XLK", "XLF"]


# ── generate_signals helpers ──────────────────────────────────────────────────

class TestUpsertTargetWeights:
    def test_creates_file_on_first_write(self, tmp_path):
        from orchestration.generate_signals import _upsert_target_weights, TARGET_WEIGHTS_PATH
        path = tmp_path / "target_weights.parquet"
        df = pl.DataFrame({
            "date": [date(2024, 1, 15)] * 3,
            "symbol": ["SPY", "QQQ", "IWM"],
            "weight": [0.4, 0.35, 0.25],
            "rebalance_date": [date(2024, 1, 2)] * 3,
        }).with_columns([pl.col("date").cast(pl.Date), pl.col("rebalance_date").cast(pl.Date)])

        with patch("orchestration.generate_signals.TARGET_WEIGHTS_PATH", path):
            _upsert_target_weights(df)

        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 3

    def test_upsert_replaces_same_date(self, tmp_path):
        from orchestration.generate_signals import _upsert_target_weights

        path = tmp_path / "tw.parquet"
        base = pl.DataFrame({
            "date": [date(2024, 1, 15)] * 2,
            "symbol": ["SPY", "QQQ"],
            "weight": [0.5, 0.5],
            "rebalance_date": [date(2024, 1, 2)] * 2,
        }).with_columns([pl.col("date").cast(pl.Date), pl.col("rebalance_date").cast(pl.Date)])
        base.write_parquet(path)

        updated = pl.DataFrame({
            "date": [date(2024, 1, 15)] * 2,
            "symbol": ["SPY", "QQQ"],
            "weight": [0.6, 0.4],
            "rebalance_date": [date(2024, 1, 2)] * 2,
        }).with_columns([pl.col("date").cast(pl.Date), pl.col("rebalance_date").cast(pl.Date)])

        with patch("orchestration.generate_signals.TARGET_WEIGHTS_PATH", path):
            _upsert_target_weights(updated)

        loaded = pl.read_parquet(path)
        assert len(loaded) == 2
        spy_row = loaded.filter(pl.col("symbol") == "SPY")
        assert abs(spy_row["weight"][0] - 0.6) < 1e-9

    def test_upsert_appends_new_date(self, tmp_path):
        from orchestration.generate_signals import _upsert_target_weights

        path = tmp_path / "tw.parquet"
        base = pl.DataFrame({
            "date": [date(2024, 1, 15)] * 2,
            "symbol": ["SPY", "QQQ"],
            "weight": [0.5, 0.5],
            "rebalance_date": [date(2024, 1, 2)] * 2,
        }).with_columns([pl.col("date").cast(pl.Date), pl.col("rebalance_date").cast(pl.Date)])
        base.write_parquet(path)

        new_day = pl.DataFrame({
            "date": [date(2024, 1, 16)] * 2,
            "symbol": ["SPY", "QQQ"],
            "weight": [0.5, 0.5],
            "rebalance_date": [date(2024, 1, 2)] * 2,
        }).with_columns([pl.col("date").cast(pl.Date), pl.col("rebalance_date").cast(pl.Date)])

        with patch("orchestration.generate_signals.TARGET_WEIGHTS_PATH", path):
            _upsert_target_weights(new_day)

        loaded = pl.read_parquet(path)
        assert len(loaded) == 4


class TestUpsertPortfolioLog:
    def test_creates_log_on_first_write(self, tmp_path):
        from orchestration.generate_signals import _upsert_portfolio_log

        path = tmp_path / "portfolio_log.parquet"
        snapshot = {
            "date": date(2024, 1, 15),
            "n_positions": 5,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
            "top5_concentration": 1.0,
            "var_1d_95": 0.012,
            "var_1d_99": 0.018,
            "pre_trade_passed": True,
            "worst_stress_scenario": "2008_GFC",
            "worst_stress_pnl": -0.35,
            "rebalance_date": date(2024, 1, 2),
        }
        with patch("orchestration.generate_signals.PORTFOLIO_LOG_PATH", path):
            _upsert_portfolio_log(snapshot)

        assert path.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
        assert loaded["n_positions"][0] == 5

    def test_upsert_replaces_same_date(self, tmp_path):
        from orchestration.generate_signals import _upsert_portfolio_log

        path = tmp_path / "portfolio_log.parquet"
        snap1 = {
            "date": date(2024, 1, 15), "n_positions": 5, "gross_exposure": 1.0,
            "net_exposure": 1.0, "top5_concentration": 1.0, "var_1d_95": 0.01,
            "var_1d_99": 0.015, "pre_trade_passed": True,
            "worst_stress_scenario": "2008_GFC", "worst_stress_pnl": -0.35,
            "rebalance_date": date(2024, 1, 2),
        }
        with patch("orchestration.generate_signals.PORTFOLIO_LOG_PATH", path):
            _upsert_portfolio_log(snap1)
            snap2 = dict(snap1)
            snap2["var_1d_95"] = 0.02
            _upsert_portfolio_log(snap2)

        loaded = pl.read_parquet(path)
        assert len(loaded) == 1
        assert abs(loaded["var_1d_95"][0] - 0.02) < 1e-9


# ── run_pipeline helpers ──────────────────────────────────────────────────────

class TestRunStep:
    def test_run_step_success(self):
        from orchestration.run_pipeline import _run_step

        code, elapsed = _run_step("test_ok", lambda: 0)
        assert code == 0
        assert elapsed >= 0

    def test_run_step_failure(self):
        from orchestration.run_pipeline import _run_step

        code, elapsed = _run_step("test_fail", lambda: 1)
        assert code == 1

    def test_run_step_exception(self):
        from orchestration.run_pipeline import _run_step

        def boom():
            raise RuntimeError("bang")

        code, elapsed = _run_step("test_exc", boom)
        assert code == 2


class TestRunPipeline:
    def test_skip_ingest_calls_generate_signals(self):
        """With --skip-ingest, only generate_signals runs."""
        from orchestration.run_pipeline import run_pipeline

        with patch("orchestration.run_pipeline.run_generate_signals", return_value=0) as mock_gs:
            code = run_pipeline(skip_ingest=True, as_of=date(2024, 1, 15))
            assert mock_gs.called
            assert code == 0

    def test_generate_signals_failure_returns_1(self):
        from orchestration.run_pipeline import run_pipeline

        with patch("orchestration.run_pipeline.run_generate_signals", return_value=1):
            with patch("orchestration.run_pipeline._send_alert"):
                code = run_pipeline(skip_ingest=True, as_of=date(2024, 1, 15))
                assert code == 1

    def test_total_ingest_failure_aborts(self):
        from orchestration.run_pipeline import run_pipeline

        with patch("orchestration.run_pipeline._run_step", return_value=(2, 0.1)):
            with patch("orchestration.run_pipeline._send_alert"):
                code = run_pipeline(skip_ingest=False, as_of=date(2024, 1, 15))
                assert code == 2
