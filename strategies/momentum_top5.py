"""Momentum Top-5 — cross-sectional 12-1 momentum, equal-weight top-5.

Strategy interface (required by tools/backtest_runner.py):
  NAME          : displayed in the Strategy Lab selector
  DESCRIPTION   : one-line summary
  DEFAULT_PARAMS: fallback values used when the UI does not override them
  get_signal()  : (features, rebal_dates, **params) -> signal DataFrame
  get_weights() : (signal, **params) -> weights DataFrame
"""

import polars as pl

from signals.momentum import cross_sectional_momentum, momentum_weights

NAME = "Momentum Top-5"
DESCRIPTION = "Cross-sectional 12-1 momentum on the ETF universe, equal-weight top-5 long-only."
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n": 5,
    "cost_bps": 5.0,
    "weight_scheme": "equal",
}


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    **kwargs,
) -> pl.DataFrame:
    """Rank symbols by 12-1 momentum and select the top_n on each rebalance date."""
    return cross_sectional_momentum(features, rebal_dates, top_n=top_n)


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Convert ranked signal to target weights."""
    return momentum_weights(signal, weight_scheme=weight_scheme)
