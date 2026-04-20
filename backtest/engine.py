"""Vectorized backtesting engine backed by VectorBT.

Wraps VectorBT's Portfolio.from_orders() so that the rest of the codebase
depends only on this module's interface, not on VectorBT directly. If we
ever swap VectorBT for something else, only this file changes.

Core function:
    run_backtest(prices, weights, cost_bps) -> BacktestResult

prices  : wide DataFrame [date × symbol] of adjusted close prices
weights : output of signals.momentum_weights() — [rebalance_date, symbol, weight]
cost_bps: round-trip cost in basis points (default 5 = 0.05%)
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt

# Keep a module-level pd reference so the try/except import inside run_backtest
# doesn't shadow it
_pd = pd


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    equity_curve: pd.Series          # index=date, values=portfolio value
    returns: pd.Series               # daily log returns
    positions: pd.DataFrame          # date × symbol position sizes (shares)
    weights_history: pd.DataFrame    # date × symbol weights (0–1)
    trades: pd.DataFrame             # one row per trade
    total_cost: float                # total transaction costs paid
    sharpe: float
    sortino: float
    max_drawdown: float
    cagr: float
    calmar: float
    total_return: float


def _build_target_weight_matrix(
    weights: pl.DataFrame,
    price_dates: list[date],
    symbols: list[str],
) -> pd.DataFrame:
    """Forward-fill monthly weights into a daily date × symbol matrix."""
    w_df = weights.to_pandas().set_index(["rebalance_date", "symbol"])["weight"].unstack(
        fill_value=0.0
    )
    # Reindex to full daily date range and forward-fill
    daily_index = pd.DatetimeIndex([pd.Timestamp(d) for d in price_dates])
    rebal_index = pd.DatetimeIndex([pd.Timestamp(d) for d in w_df.index])
    w_df.index = rebal_index

    # Align to daily index: first reindex to daily, then ffill
    w_daily = w_df.reindex(daily_index).ffill()

    # Fill any remaining NaN (before first rebalance) with 0
    w_daily = w_daily.fillna(0.0)

    # Align columns to provided symbols, filling missing with 0
    for sym in symbols:
        if sym not in w_daily.columns:
            w_daily[sym] = 0.0
    return w_daily[symbols]


def run_backtest(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    cost_bps: float = 5.0,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Run a vectorized backtest using VectorBT.

    Parameters
    ----------
    prices    : Polars DataFrame with columns [date, symbol, adj_close]
    weights   : Polars DataFrame with columns [rebalance_date, symbol, weight]
    cost_bps  : Round-trip transaction cost in basis points (5 = 0.05%)
    initial_cash: Starting portfolio value

    Returns
    -------
    BacktestResult with equity curve, positions, and all summary statistics.
    """
    # --- Pivot prices to wide format (date × symbol) ---
    price_wide = (
        prices.pivot(index="date", on="symbol", values="adj_close")
        .sort("date")
    )
    price_dates = [d for d in price_wide["date"].to_list()]
    symbols = [c for c in price_wide.columns if c != "date"]

    price_pd = price_wide.to_pandas().set_index("date")
    price_pd.index = pd.DatetimeIndex([pd.Timestamp(d) for d in price_dates])

    # --- Build daily target weight matrix ---
    w_matrix = _build_target_weight_matrix(weights, price_dates, symbols)
    w_matrix = w_matrix[symbols]   # ensure column order matches prices

    # --- Compute target sizes in cash terms ---
    # VectorBT from_signals() is verbose; use simulate_from_orders via
    # Portfolio.from_orders with target percentage allocation
    one_way_cost = cost_bps / 10_000 / 2   # split round-trip across entry and exit

    pf = vbt.Portfolio.from_orders(
        close=price_pd[symbols],
        size=w_matrix,
        size_type="targetpercent",
        fees=one_way_cost,
        slippage=0.0,
        init_cash=initial_cash,
        freq="1D",
        cash_sharing=True,   # all symbols share one cash pool
        group_by=True,       # treat as a single portfolio
        call_seq="auto",
    )

    # --- Extract results ---
    # With group_by=True, pf.value() and pf.returns() return Series not DataFrames
    equity = pf.value()
    returns = pf.returns()

    try:
        trades_df = pf.trades.records_readable
    except Exception:
        trades_df = _pd.DataFrame()

    weights_hist = w_matrix.copy()

    try:
        total_cost = float(pf.orders.fees.sum())
    except Exception:
        total_cost = 0.0

    # --- Compute summary statistics ---
    ann_factor = 252
    ret_arr = returns.values if hasattr(returns, "values") else returns
    eq_arr = equity.values if hasattr(equity, "values") else equity

    sharpe = _sharpe(ret_arr, ann_factor)
    sortino = _sortino(ret_arr, ann_factor)
    max_dd = _max_drawdown(eq_arr)
    cagr = _cagr(eq_arr, ann_factor)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    total_return = float(eq_arr[-1] / initial_cash - 1)

    equity_series = equity if hasattr(equity, "iloc") else _pd.Series(equity)
    returns_series = returns if hasattr(returns, "iloc") else _pd.Series(returns)

    return BacktestResult(
        equity_curve=equity_series,
        returns=returns_series,
        positions=pf.assets() if hasattr(pf, "assets") else pd.DataFrame(),
        weights_history=weights_hist,
        trades=trades_df,
        total_cost=total_cost,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        cagr=round(cagr, 4),
        calmar=round(calmar, 3),
        total_return=round(total_return, 4),
    )


# ── Statistical helpers ────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, ann_factor: int = 252, rfr: float = 0.0) -> float:
    excess = returns - rfr / ann_factor
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ann_factor))


def _sortino(returns: np.ndarray, ann_factor: int = 252, rfr: float = 0.0) -> float:
    excess = returns - rfr / ann_factor
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(ann_factor))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def _cagr(equity: np.ndarray, ann_factor: int = 252) -> float:
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    total_return = equity[-1] / equity[0]
    n_years = len(equity) / ann_factor
    return float(total_return ** (1 / n_years) - 1)
