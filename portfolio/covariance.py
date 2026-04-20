"""Covariance matrix estimation with Ledoit-Wolf shrinkage.

Raw sample covariance from estimated returns is notoriously unstable —
small changes in inputs cause large swings in optimized weights. Ledoit-Wolf
shrinkage (Ledoit & Wolf 2004) regularises toward a structured estimator,
making downstream optimization far more stable.

All functions return numpy arrays + a symbol list so callers stay
independent of the internal representation.
"""

from datetime import date

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf


def compute_returns(
    prices: pl.DataFrame,
    method: str = "log",
) -> tuple[np.ndarray, list[str]]:
    """Pivot prices to a (T × N) returns matrix.

    Parameters
    ----------
    prices : DataFrame with [date, symbol, adj_close] — multiple symbols
    method : "log" (default) or "simple"

    Returns
    -------
    (returns_matrix, symbols)  — matrix rows = dates, columns = symbols
    """
    wide = (
        prices.pivot(index="date", on="symbol", values="adj_close")
        .sort("date")
    )
    symbols = [c for c in wide.columns if c != "date"]
    price_arr = wide.select(symbols).to_numpy().astype(float)

    if method == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_arr = np.diff(np.log(price_arr), axis=0)
    else:
        ret_arr = np.diff(price_arr, axis=0) / price_arr[:-1]

    # Replace NaN / Inf from zero or negative prices with 0
    ret_arr = np.nan_to_num(ret_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return ret_arr, symbols


def ledoit_wolf_cov(
    prices: pl.DataFrame,
    lookback_days: int = 252,
    annualize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute Ledoit-Wolf shrunk covariance matrix.

    Parameters
    ----------
    prices       : [date, symbol, adj_close]
    lookback_days: Rolling window for estimation (252 = 1 year default)
    annualize    : Scale daily cov to annual (× 252)

    Returns
    -------
    (cov_matrix, symbols)  — (N × N) numpy array
    """
    returns, symbols = compute_returns(prices)

    # Use the most recent lookback_days rows
    if len(returns) > lookback_days:
        returns = returns[-lookback_days:]

    lw = LedoitWolf()
    lw.fit(returns)
    cov = lw.covariance_

    if annualize:
        cov = cov * 252

    return cov, symbols


def sample_cov(
    prices: pl.DataFrame,
    lookback_days: int = 252,
    annualize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Raw sample covariance (no shrinkage). Useful for comparison."""
    returns, symbols = compute_returns(prices)
    if len(returns) > lookback_days:
        returns = returns[-lookback_days:]
    cov = np.cov(returns.T)
    if annualize:
        cov = cov * 252
    return cov, symbols


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
