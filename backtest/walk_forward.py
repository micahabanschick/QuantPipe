"""Walk-forward validation for time-series strategies.

Standard K-Fold is wrong for financial time series — it allows future data
to leak into training. Walk-forward validation uses expanding or rolling
in-sample windows with a fixed out-of-sample test period.

         |-- in-sample --|-- OOS --|
Fold 1:  [==========]    [>>>]
Fold 2:  [================]  [>>>]
Fold 3:  [====================] [>>>]

Each fold returns a BacktestResult on the OOS window only.
Aggregate the OOS equity curves to get a clean walk-forward equity curve.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

import pandas as pd
import polars as pl

from backtest.engine import BacktestResult, run_backtest
from backtest.tearsheet import tearsheet_dict


@dataclass
class WalkForwardFold:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    result: BacktestResult


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    combined_equity: pd.Series    # OOS equity curves stitched together
    combined_sharpe: float
    combined_cagr: float
    combined_max_drawdown: float

    def summary(self) -> list[dict]:
        rows = []
        for f in self.folds:
            d = tearsheet_dict(f.result)
            d["fold"] = f.fold
            d["train_start"] = str(f.train_start)
            d["train_end"] = str(f.train_end)
            d["test_start"] = str(f.test_start)
            d["test_end"] = str(f.test_end)
            rows.append(d)
        return rows


def walk_forward(
    prices: pl.DataFrame,
    features: pl.DataFrame,
    signal_fn,
    weight_fn,
    train_years: int = 3,
    test_months: int = 12,
    cost_bps: float = 5.0,
    top_n: int = 5,
) -> WalkForwardResult:
    """Run expanding-window walk-forward validation.

    Parameters
    ----------
    prices      : [date, symbol, adj_close]
    features    : [date, symbol, momentum_12m_1m, ...] from feature library
    signal_fn   : cross_sectional_momentum (or any signal function)
    weight_fn   : momentum_weights (or any weight function)
    train_years : Minimum in-sample length in years
    test_months : OOS test window length in months
    cost_bps    : Round-trip cost
    top_n       : Symbols in long book

    Returns
    -------
    WalkForwardResult with per-fold results and combined OOS equity curve.
    """
    from signals.momentum import get_monthly_rebalance_dates

    all_dates = sorted(prices["date"].unique().to_list())
    if not all_dates:
        raise ValueError("prices DataFrame is empty")

    data_start = all_dates[0]
    data_end = all_dates[-1]

    folds: list[WalkForwardFold] = []
    fold_idx = 0

    # First test window starts after minimum training period
    test_start = _add_years(data_start, train_years)

    while test_start < data_end:
        test_end_candidate = _add_months(test_start, test_months)
        test_end = min(test_end_candidate, data_end)

        if test_start >= test_end:
            break

        train_start = data_start
        train_end = test_start - timedelta(days=1)

        # Generate signal on full history up to train_end, evaluate on test window
        rebal_dates = get_monthly_rebalance_dates(train_start, test_end, all_dates)

        feat_slice = features.filter(pl.col("date") <= train_end)
        signal = signal_fn(feat_slice, rebal_dates, top_n=top_n)
        weights = weight_fn(signal)

        # Restrict backtest to OOS window only
        oos_prices = prices.filter(
            (pl.col("date") >= test_start) & (pl.col("date") <= test_end)
        )
        oos_weights = weights.filter(
            (pl.col("rebalance_date") >= train_start) & (pl.col("rebalance_date") <= test_end)
        )

        if oos_prices.is_empty() or oos_weights.is_empty():
            test_start = _add_months(test_start, test_months)
            continue

        try:
            result = run_backtest(oos_prices, oos_weights, cost_bps=cost_bps)
            folds.append(WalkForwardFold(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                result=result,
            ))
            fold_idx += 1
        except Exception as exc:
            print(f"[walk_forward] Fold {fold_idx} failed: {exc}")

        test_start = _add_months(test_start, test_months)

    if not folds:
        raise ValueError("No valid walk-forward folds produced — check data range")

    combined_equity = _stitch_equity_curves(folds)
    combined_returns = combined_equity.pct_change().dropna()

    from backtest.engine import _cagr, _max_drawdown, _sharpe
    ret_arr = combined_returns.values
    combined_sharpe = _sharpe(ret_arr)
    combined_cagr = _cagr(combined_equity.values)
    combined_max_dd = _max_drawdown(combined_equity.values)

    return WalkForwardResult(
        folds=folds,
        combined_equity=combined_equity,
        combined_sharpe=round(combined_sharpe, 3),
        combined_cagr=round(combined_cagr, 4),
        combined_max_drawdown=round(combined_max_dd, 4),
    )


def _stitch_equity_curves(folds: list[WalkForwardFold]) -> pd.Series:
    """Stitch OOS equity curves into a single normalised series."""
    curves = []
    base = 100_000.0
    for fold in folds:
        eq = fold.result.equity_curve
        # Rescale so each fold starts where previous ended
        scale = base / eq.iloc[0]
        scaled = eq * scale
        curves.append(scaled)
        base = float(scaled.iloc[-1])
    return pd.concat(curves)


def _add_months(d: date, months: int) -> date:
    """Add calendar months to a date, correctly handling month-end and Feb-29."""
    return (date(d.year, d.month, d.day) + relativedelta(months=months))


def _add_years(d: date, years: int) -> date:
    return _add_months(d, years * 12)
