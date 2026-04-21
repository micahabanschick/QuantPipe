"""Signal analytics — IC decay, turnover, and rank autocorrelation.

Pure functions. Uses pandas and scipy for analytics.
No I/O, no Streamlit, no Plotly.

Public API:
    ic_decay(factor_pivot, price_pivot, horizons, min_symbols) -> list[DecayPoint]
    signal_turnover(signal_df, ...)                            -> TurnoverResult
    rank_autocorrelation(signal_df, ...)                       -> dict[int, float]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class DecayPoint:
    horizon_days: int
    mean_ic: float
    icir: float
    n_obs: int


@dataclass
class TurnoverResult:
    rebal_dates: list        # rebalance dates (starting from 2nd)
    turnover: list           # fraction of portfolio that changed (0–1)
    mean_turnover: float


# ── IC decay ──────────────────────────────────────────────────────────────────

def ic_decay(
    factor_pivot: pd.DataFrame,
    price_pivot: pd.DataFrame,
    horizons: list | None = None,
    min_symbols: int = 5,
) -> list:
    """IC (Spearman rank correlation) at multiple forward horizons.

    For each horizon H, compute IC between factor values and H-day forward
    returns. Samples every ``max(1, H//3)`` dates to avoid autocorrelation.

    Parameters
    ----------
    factor_pivot : (date × symbol) factor values
    price_pivot  : (date × symbol) adjusted close prices
    horizons     : forward horizons in days (default [1, 5, 21, 63, 126])
    min_symbols  : minimum symbols with valid data per observation

    Returns
    -------
    list[DecayPoint], one per horizon
    """
    if horizons is None:
        horizons = [1, 5, 21, 63, 126]

    results: list[DecayPoint] = []

    if factor_pivot.empty or price_pivot.empty:
        for h in horizons:
            results.append(DecayPoint(horizon_days=h, mean_ic=0.0, icir=0.0, n_obs=0))
        return results

    # Align indices
    common_symbols = factor_pivot.columns.intersection(price_pivot.columns)
    common_dates   = factor_pivot.index.intersection(price_pivot.index)
    if len(common_symbols) < min_symbols or len(common_dates) < 2:
        for h in horizons:
            results.append(DecayPoint(horizon_days=h, mean_ic=0.0, icir=0.0, n_obs=0))
        return results

    fp = factor_pivot[common_symbols].reindex(common_dates)
    pp = price_pivot[common_symbols].reindex(common_dates)

    all_dates = sorted(fp.index)

    for h in horizons:
        step = max(1, h // 3)
        ics: list[float] = []

        sample_dates = all_dates[::step]
        for dt in sample_dates:
            # Find target date h trading days ahead
            pos = all_dates.index(dt)
            fwd_pos = pos + h
            if fwd_pos >= len(all_dates):
                continue
            fwd_dt = all_dates[fwd_pos]

            factor_row = fp.loc[dt].dropna()
            if len(factor_row) < min_symbols:
                continue

            price_now = pp.loc[dt, factor_row.index].dropna()
            price_fwd = pp.loc[fwd_dt, factor_row.index].dropna()

            shared = factor_row.index.intersection(price_now.index).intersection(price_fwd.index)
            if len(shared) < min_symbols:
                continue

            fwd_ret = (price_fwd[shared] / price_now[shared] - 1).values
            fac_vals = factor_row[shared].values

            if np.std(fac_vals) < 1e-12 or np.std(fwd_ret) < 1e-12:
                continue

            rho, _ = spearmanr(fac_vals, fwd_ret)
            if not np.isnan(rho):
                ics.append(float(rho))

        if ics:
            arr = np.array(ics)
            mean_ic = float(arr.mean())
            std_ic  = float(arr.std())
            icir    = mean_ic / std_ic if std_ic > 1e-12 else 0.0
        else:
            mean_ic, icir = 0.0, 0.0

        results.append(DecayPoint(
            horizon_days=h,
            mean_ic=round(mean_ic, 6),
            icir=round(icir, 4),
            n_obs=len(ics),
        ))

    return results


# ── Signal turnover ───────────────────────────────────────────────────────────

def signal_turnover(
    signal_df: pl.DataFrame,
    selected_col: str = "selected",
    date_col: str = "rebalance_date",
) -> TurnoverResult:
    """Compute per-rebalance portfolio turnover.

    Turnover at date t =
        |selected(t) XOR selected(t-1)| / max(|selected(t)|, |selected(t-1)|)

    Returns TurnoverResult.
    """
    if signal_df.is_empty() or date_col not in signal_df.columns:
        return TurnoverResult(rebal_dates=[], turnover=[], mean_turnover=0.0)

    dates = sorted(signal_df[date_col].unique().to_list())
    if len(dates) < 2:
        return TurnoverResult(rebal_dates=[], turnover=[], mean_turnover=0.0)

    def _selected_set(d) -> set:
        day = signal_df.filter(pl.col(date_col) == d)
        return set(
            day.filter(pl.col(selected_col) == True)["symbol"].to_list()
        )

    result_dates: list = []
    result_to: list[float] = []

    prev_set = _selected_set(dates[0])
    for d in dates[1:]:
        curr_set = _selected_set(d)
        changed = len(prev_set.symmetric_difference(curr_set))
        denom = max(len(curr_set), len(prev_set))
        to = changed / denom if denom > 0 else 0.0
        result_dates.append(d)
        result_to.append(round(to, 6))
        prev_set = curr_set

    mean_to = float(np.mean(result_to)) if result_to else 0.0
    return TurnoverResult(
        rebal_dates=result_dates,
        turnover=result_to,
        mean_turnover=round(mean_to, 6),
    )


# ── Rank autocorrelation ───────────────────────────────────────────────────────

def rank_autocorrelation(
    signal_df: pl.DataFrame,
    rank_col: str = "rank",
    date_col: str = "rebalance_date",
    lags: list | None = None,
) -> dict:
    """Spearman correlation of symbol ranks at date t vs t-lag.

    Returns {lag: mean_spearman_rho} — how sticky are the rankings?

    Parameters
    ----------
    signal_df : Output of cross_sectional_momentum() or composite_signal()
    rank_col  : Column containing integer ranks
    date_col  : Column containing the rebalance date
    lags      : List of integer lags (default [1, 2, 3])
    """
    if lags is None:
        lags = [1, 2, 3]

    result: dict[int, float] = {}

    if signal_df.is_empty() or rank_col not in signal_df.columns:
        return {lag: 0.0 for lag in lags}

    dates = sorted(signal_df[date_col].unique().to_list())
    if len(dates) < 2:
        return {lag: 0.0 for lag in lags}

    # Build pivot: date × symbol → rank
    rank_pd = (
        signal_df.select([date_col, "symbol", rank_col])
        .to_pandas()
        .pivot(index=date_col, columns="symbol", values=rank_col)
    )

    for lag in lags:
        rhos: list[float] = []
        for i in range(lag, len(dates)):
            dt_now = dates[i]
            dt_lag = dates[i - lag]

            if dt_now not in rank_pd.index or dt_lag not in rank_pd.index:
                continue

            row_now = rank_pd.loc[dt_now].dropna()
            row_lag = rank_pd.loc[dt_lag].dropna()

            shared = row_now.index.intersection(row_lag.index)
            if len(shared) < 3:
                continue

            rho, _ = spearmanr(row_now[shared].values, row_lag[shared].values)
            if not np.isnan(rho):
                rhos.append(float(rho))

        result[lag] = round(float(np.mean(rhos)), 4) if rhos else 0.0

    return result
