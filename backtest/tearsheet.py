"""Tearsheet — formats BacktestResult as text or dict; provides analytic helpers.

Intentionally output-agnostic so it works headlessly in the nightly pipeline.
The Streamlit strategy lab uses the helper functions below for richer displays.
"""

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


def print_tearsheet(result: BacktestResult, title: str = "Backtest") -> None:
    eq = result.equity_curve
    n_years = len(eq) / 252
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Period          : {eq.index[0].date()} -> {eq.index[-1].date()} ({n_years:.1f} yrs)")
    print(f"  Starting value  : ${eq.iloc[0]:>12,.0f}")
    print(f"  Ending value    : ${eq.iloc[-1]:>12,.0f}")
    print(f"{'─'*60}")
    print(f"  Total return    : {result.total_return:>+.1%}")
    print(f"  CAGR            : {result.cagr:>+.1%}")
    print(f"  Sharpe ratio    : {result.sharpe:>6.3f}")
    print(f"  Sortino ratio   : {result.sortino:>6.3f}")
    print(f"  Max drawdown    : {result.max_drawdown:>+.1%}")
    print(f"  Calmar ratio    : {result.calmar:>6.3f}")
    print(f"  Transaction cost: ${result.total_cost:>10,.0f}")
    print(f"{'─'*60}")
    if not result.trades.empty:
        print(f"  Trades          : {len(result.trades)}")
    print(f"{'='*60}\n")


def tearsheet_dict(result: BacktestResult) -> dict:
    """Core metrics dict — used by the pipeline and backtest_runner."""
    eq = result.equity_curve
    return {
        "start":        str(eq.index[0].date()),
        "end":          str(eq.index[-1].date()),
        "years":        round(len(eq) / 252, 1),
        "total_return": result.total_return,
        "cagr":         result.cagr,
        "sharpe":       result.sharpe,
        "sortino":      result.sortino,
        "max_drawdown": result.max_drawdown,
        "calmar":       result.calmar,
        "total_cost":   round(result.total_cost, 2),
        "n_trades":     len(result.trades) if not result.trades.empty else 0,
    }


# ── Analytics helpers (used by the Strategy Lab) ───────────────────────────────

_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def monthly_returns_matrix(equity: pd.Series) -> pd.DataFrame:
    """Year × month matrix of monthly returns (NaN where no data)."""
    monthly = equity.resample("ME").last()
    ret = monthly.pct_change().dropna()
    if ret.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"year": ret.index.year, "month": ret.index.month, "ret": ret.values})
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [_MONTH_ABBR.get(m, str(m)) for m in pivot.columns]
    return pivot


def rolling_sharpe_series(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualised Sharpe over `window` trading days."""
    def _s(x: np.ndarray) -> float:
        std = x.std()
        return float(x.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return returns.rolling(window).apply(_s, raw=True).dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from peak, in percent."""
    peak = equity.cummax()
    return (equity - peak) / peak * 100.0


def alpha_beta(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
) -> tuple[float, float]:
    """(annualised alpha, beta) via OLS regression on aligned daily returns."""
    aligned = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0, 1.0
    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values
    beta, alpha_d = np.polyfit(x, y, 1)
    return round(float(alpha_d * 252), 4), round(float(beta), 4)


def tracking_error(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    ann: int = 252,
) -> float:
    """Annualised tracking error vs benchmark."""
    aligned = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return round(float(diff.std() * np.sqrt(ann)), 4)


def information_ratio(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    ann: int = 252,
) -> float:
    """Annualised information ratio (active return / tracking error)."""
    aligned = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = diff.std()
    return round(float(diff.mean() / te * np.sqrt(ann)), 3) if te > 0 else 0.0
