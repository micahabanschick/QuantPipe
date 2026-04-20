"""Tearsheet — formats and prints BacktestResult as a human-readable summary.

Intentionally text-only so it works headlessly in the nightly pipeline.
The Streamlit performance dashboard (Phase 5) will render the charts.
"""

from backtest.engine import BacktestResult


def print_tearsheet(result: BacktestResult, title: str = "Backtest") -> None:
    """Print a formatted text tearsheet to stdout."""
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

    # Trade stats
    if not result.trades.empty:
        n_trades = len(result.trades)
        print(f"  Trades          : {n_trades}")

    print(f"{'='*60}\n")


def tearsheet_dict(result: BacktestResult) -> dict:
    """Return tearsheet metrics as a plain dict (for logging / Streamlit)."""
    eq = result.equity_curve
    return {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "years": round(len(eq) / 252, 1),
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_drawdown": result.max_drawdown,
        "calmar": result.calmar,
        "total_cost": round(result.total_cost, 2),
        "n_trades": len(result.trades) if not result.trades.empty else 0,
    }
