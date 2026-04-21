"""Factor analysis — time-series stats, distribution, and information coefficient.

Pure functions: (features_df, prices_df, params) -> structured results.
No I/O, no Streamlit, no Plotly. Usable from notebooks and scripts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr


@dataclass
class FactorStats:
    mean: float
    std: float
    skew: float
    kurt: float
    p5: float
    p95: float
    n_obs: int


@dataclass
class ICResult:
    """Output of compute_ic().

    dates  : observation dates (one per sample step)
    values : Spearman rank IC at each date
    rolling_mean : 6-period rolling mean of IC
    mean_ic : grand mean IC
    icir    : mean_ic / std(IC)  — information ratio of the IC series
    """
    dates: list
    values: list
    rolling_mean: list
    mean_ic: float
    icir: float


def factor_pivot_from_features(
    features_df: pl.DataFrame,
    symbols: list[str],
    factor: str,
) -> pd.DataFrame:
    """Pivot a single factor column to (date × symbol) wide format.

    Returns an empty DataFrame if the factor column is absent.
    """
    if factor not in features_df.columns:
        return pd.DataFrame()

    fac = (
        features_df
        .filter(pl.col("symbol").is_in(symbols))
        .select(["date", "symbol", factor])
        .to_pandas()
    )
    fac["date"] = pd.to_datetime(fac["date"])
    return fac.pivot(index="date", columns="symbol", values=factor)


def price_pivot_from_bars(prices_pl: pl.DataFrame) -> pd.DataFrame:
    """Pivot adj_close prices to (date × symbol) wide format."""
    prices_pd = prices_pl.select(["date", "symbol", "adj_close"]).to_pandas()
    prices_pd["date"] = pd.to_datetime(prices_pd["date"])
    return prices_pd.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def compute_factor_stats(values: np.ndarray) -> FactorStats:
    """Descriptive statistics for a 1-D array of factor observations."""
    clean = values[~np.isnan(values)]
    s = pd.Series(clean)
    return FactorStats(
        mean=float(clean.mean()),
        std=float(clean.std()),
        skew=float(s.skew()),
        kurt=float(s.kurtosis()),
        p5=float(np.percentile(clean, 5)),
        p95=float(np.percentile(clean, 95)),
        n_obs=int(len(clean)),
    )


def compute_ic(
    factor_pivot: pd.DataFrame,
    price_pivot: pd.DataFrame,
    forward_window: int,
    step: int | None = None,
    min_symbols: int = 5,
    rolling_periods: int = 6,
) -> ICResult:
    """Compute the rolling Spearman rank information coefficient.

    At each sampled date, computes the cross-sectional Spearman rank
    correlation between factor values and forward ``forward_window``-day
    returns. Samples every ``step`` dates (default: forward_window // 3).

    Parameters
    ----------
    factor_pivot    : (date × symbol) factor values
    price_pivot     : (date × symbol) adj_close prices
    forward_window  : forward return horizon in trading days
    step            : sampling cadence in calendar rows; default forward_window // 3
    min_symbols     : minimum shared symbols required to record an IC observation
    rolling_periods : window for the rolling mean of IC

    Returns
    -------
    ICResult with per-date IC, rolling mean, mean IC, and IC IR.
    """
    if factor_pivot.empty or price_pivot.empty:
        return ICResult([], [], [], 0.0, 0.0)

    step = step or max(1, forward_window // 3)
    fwd_ret = price_pivot.pct_change(forward_window).shift(-forward_window)

    ic_dates: list = []
    ic_vals:  list = []

    for dt in sorted(factor_pivot.index)[::step]:
        if dt not in fwd_ret.index:
            continue
        factor_row = factor_pivot.loc[dt].dropna()
        fwd_row    = fwd_ret.loc[dt].reindex(factor_row.index).dropna()
        shared     = factor_row.index.intersection(fwd_row.index)
        if len(shared) < min_symbols:
            continue
        rho, _ = spearmanr(factor_row[shared], fwd_row[shared])
        if not np.isnan(rho):
            ic_dates.append(dt)
            ic_vals.append(float(rho))

    if not ic_vals:
        return ICResult([], [], [], 0.0, 0.0)

    ic_series   = pd.Series(ic_vals, index=ic_dates)
    roll_mean   = ic_series.rolling(rolling_periods, min_periods=max(1, rolling_periods // 2)).mean()
    mean_ic     = float(ic_series.mean())
    std_ic      = float(ic_series.std())
    icir        = mean_ic / std_ic if std_ic > 1e-10 else 0.0

    return ICResult(
        dates=ic_dates,
        values=ic_vals,
        rolling_mean=roll_mean.tolist(),
        mean_ic=mean_ic,
        icir=icir,
    )
