"""Backtesting engine — pure NumPy/Pandas, no external backtest library.

Core function:
    run_backtest(prices, weights, cost_bps) -> BacktestResult

prices  : Polars DataFrame [date, symbol, adj_close|close]
weights : Polars DataFrame [rebalance_date, symbol, weight]
cost_bps: round-trip transaction cost in basis points (5 = 0.05% per side)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

_pd = pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series        # index=date, values=portfolio value
    returns: pd.Series             # daily simple returns
    positions: pd.DataFrame        # date × symbol shares held
    weights_history: pd.DataFrame  # date × symbol target weights (forward-filled)
    trades: pd.DataFrame           # one row per trade
    total_cost: float
    sharpe: float
    sortino: float
    max_drawdown: float
    cagr: float
    calmar: float
    total_return: float


def _build_target_weight_matrix(
    weights: pl.DataFrame,
    price_dates: list,
    symbols: list[str],
) -> pd.DataFrame:
    """Forward-fill rebalance weights into a daily date × symbol matrix."""
    w_pd = weights.to_pandas()
    date_col = "rebalance_date" if "rebalance_date" in w_pd.columns else "date"
    w_df = w_pd.set_index([date_col, "symbol"])["weight"].unstack(fill_value=0.0)

    daily_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in price_dates])
    w_df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in w_df.index])
    w_daily = w_df.reindex(daily_idx).ffill().fillna(0.0)

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
    """Simulate a rebalancing portfolio.

    Algorithm (per day):
      Rebalance day : compute target shares from current NAV, deduct one-way
                      transaction cost from NAV, set shares.
      Other days    : shares drift with prices; NAV = sum(shares × price).
    """
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    price_wide = prices.pivot(index="date", on="symbol", values=price_col).sort("date")
    price_dates = price_wide["date"].to_list()
    symbols = [c for c in price_wide.columns if c != "date"]

    price_pd = price_wide.to_pandas().set_index("date")
    price_pd.index = pd.DatetimeIndex([pd.Timestamp(d) for d in price_dates])
    price_pd = price_pd[symbols].ffill().bfill()

    w_matrix = _build_target_weight_matrix(weights, price_dates, symbols)

    date_col = "rebalance_date" if "rebalance_date" in weights.columns else "date"
    rebal_set = {pd.Timestamp(d) for d in weights[date_col].unique().to_list()}

    one_way = cost_bps / 20_000.0  # fraction of traded value per leg

    n, ns = len(price_pd), len(symbols)
    shares = np.zeros(ns)
    cash = initial_cash

    nav_arr = np.empty(n)
    shares_hist = np.zeros((n, ns))
    trade_records: list[dict] = []

    for i, (ts, row) in enumerate(price_pd.iterrows()):
        px = row.values.astype(float)
        nav_pre = float((shares * px).sum()) + cash

        if ts in rebal_set:
            target_w = w_matrix.loc[ts].values.astype(float)

            # Approximate cost with pre-cost NAV; re-target on effective NAV
            ts_approx = np.where(px > 0, target_w * nav_pre / px, 0.0)
            est_cost = float(np.abs(ts_approx - shares).dot(px)) * one_way
            eff_nav = max(nav_pre - est_cost, 0.0)

            target_sh = np.where(px > 0, target_w * eff_nav / px, 0.0)
            delta = target_sh - shares
            actual_cost = float(np.abs(delta).dot(px)) * one_way

            for j, sym in enumerate(symbols):
                if abs(delta[j]) > 1e-6:
                    trade_records.append({
                        "date":   ts.date(),
                        "symbol": sym,
                        "qty":    round(float(delta[j]), 4),
                        "price":  round(float(px[j]), 2),
                        "value":  round(float(delta[j] * px[j]), 2),
                        "cost":   round(float(abs(delta[j]) * px[j] * one_way), 2),
                        "side":   "BUY" if delta[j] > 0 else "SELL",
                    })

            shares = target_sh
            cash = 0.0
            nav_arr[i] = nav_pre - actual_cost
        else:
            nav_arr[i] = nav_pre

        shares_hist[i] = shares

    equity = pd.Series(nav_arr, index=price_pd.index)
    returns = equity.pct_change().fillna(0.0)
    shares_df = pd.DataFrame(shares_hist, index=price_pd.index, columns=symbols)
    trades_df = (
        pd.DataFrame(trade_records)
        if trade_records
        else pd.DataFrame(columns=["date", "symbol", "qty", "price", "value", "cost", "side"])
    )

    total_cost = float(trades_df["cost"].sum()) if not trades_df.empty else 0.0
    ret_arr = returns.values

    sharpe   = _sharpe(ret_arr)
    sortino  = _sortino(ret_arr)
    max_dd   = _max_drawdown(nav_arr)
    cagr     = _cagr(nav_arr)
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0
    total_ret = float(nav_arr[-1] / initial_cash - 1)

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=shares_df,
        weights_history=w_matrix,
        trades=trades_df,
        total_cost=total_cost,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        cagr=round(cagr, 4),
        calmar=round(calmar, 3),
        total_return=round(total_ret, 4),
    )


# ── Statistical helpers ────────────────────────────────────────────────────────

def _sharpe(ret: np.ndarray, ann: int = 252, rfr: float = 0.0) -> float:
    exc = ret - rfr / ann
    std = exc.std()
    return float(exc.mean() / std * np.sqrt(ann)) if std > 0 else 0.0


def _sortino(ret: np.ndarray, ann: int = 252, rfr: float = 0.0) -> float:
    exc = ret - rfr / ann
    down = exc[exc < 0]
    dstd = down.std()
    return float(exc.mean() / dstd * np.sqrt(ann)) if dstd > 0 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def _cagr(equity: np.ndarray, ann: int = 252) -> float:
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (ann / len(equity)) - 1)
