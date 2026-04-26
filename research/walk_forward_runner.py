"""Walk-forward validation runner — data loading + backtest.walk_forward orchestration.

Pure orchestration: no Streamlit, no Plotly.
The dashboard is responsible for caching and rendering results.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd
import polars as pl

from backtest.tearsheet import tearsheet_dict
from backtest.walk_forward import WalkForwardResult


@dataclass
class WFVConfig:
    train_years: int  = 3
    test_months: int  = 12
    top_n:       int  = 5
    cost_bps:    float = 5.0


def run(
    symbols: list[str],
    start: date,
    end: date,
    config: WFVConfig | None = None,
) -> WalkForwardResult:
    """Load price + feature data and run expanding-window walk-forward validation.

    Parameters
    ----------
    symbols : equity universe symbols
    start   : backtest start date (should provide at least train_years of history)
    end     : backtest end date
    config  : WFVConfig; uses defaults if None

    Returns
    -------
    WalkForwardResult — see backtest.walk_forward for full schema.

    Raises
    ------
    RuntimeError if data cannot be loaded or no complete folds are produced.
    """
    from backtest.walk_forward import walk_forward
    from features.compute import load_features
    from signals.momentum import cross_sectional_momentum, momentum_weights
    from storage.parquet_store import load_bars

    cfg = config or WFVConfig()

    prices = load_bars(symbols, start, end, "equity")
    if prices.is_empty():
        raise RuntimeError("No price data — run the historical backfill first.")

    features = load_features(symbols, start, end, "equity")
    if features.is_empty():
        raise RuntimeError("No features — run: uv run python features/compute.py")

    result = walk_forward(
        prices, features,
        signal_fn=cross_sectional_momentum,
        weight_fn=momentum_weights,
        train_years=cfg.train_years,
        test_months=cfg.test_months,
        cost_bps=cfg.cost_bps,
        top_n=cfg.top_n,
    )

    if not result.folds:
        raise RuntimeError(
            f"No complete folds produced. Need at least {cfg.train_years}y train + "
            f"{cfg.test_months}m test ({cfg.train_years + cfg.test_months / 12:.1f}y total). "
            "Increase lookback or reduce window sizes."
        )

    return result


def fold_summary(result: WalkForwardResult) -> list[dict]:
    """Flatten per-fold tearsheet metrics into a list of dicts for table display."""
    rows = []
    for fold in result.folds:
        td = tearsheet_dict(fold.result)
        rows.append({
            "Fold":       fold.fold + 1,
            "Test Start": str(fold.test_start),
            "Test End":   str(fold.test_end),
            "IS Sharpe":  getattr(fold, "is_sharpe", 0.0),
            "OOS Sharpe": round(td.get("sharpe", 0.0), 3),
            "OOS CAGR":   td.get("cagr", 0.0),
            "OOS Max DD": td.get("max_drawdown", 0.0),
            "OOS Vol":    td.get("vol", 0.0),
            "Calmar":     round(td.get("calmar", 0.0), 3),
        })
    return rows


def oos_equity_normalised(result: WalkForwardResult, initial: float = 10_000) -> pd.Series:
    """Return the combined OOS equity curve normalised to ``initial``."""
    eq = result.combined_equity
    return initial * eq / eq.iloc[0]
