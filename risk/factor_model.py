"""Factor model — ETF-proxy factor returns and beta estimation.

Pure analytics. Uses pandas and numpy.
No I/O, no Streamlit, no Plotly.

Factor proxies are ETFs already present in the equity universe price data.

Public API:
    estimate_factor_returns(prices_pl, factor_proxies) -> FactorReturns
    estimate_factor_betas(portfolio_returns, factor_returns, lookback) -> FactorBetas
    rolling_factor_betas(portfolio_returns, factor_returns, window, min_periods) -> pd.DataFrame
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl


# ── Factor proxy definitions ───────────────────────────────────────────────────

# Each factor is defined as:
#   single-ETF  : (ETF,)      → daily pct-change of that ETF
#   two-ETF     : (A, B)      → daily pct-change of A minus daily pct-change of B
FACTOR_PROXIES: dict[str, tuple] = {
    "Market": ("SPY",),
    "Size":   ("IWM", "IWB"),   # small-cap minus large-cap
    "Value":  ("IWD", "IWF"),   # value minus growth (large-cap Russell)
    "LowVol": ("IWS", "IWP"),   # mid-cap value minus mid-cap growth (vol proxy)
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FactorReturns:
    """Daily factor proxy returns."""
    returns: pd.DataFrame     # (date × factor_name) float
    factor_names: list = field(default_factory=list)


@dataclass
class FactorBetas:
    """OLS regression betas of portfolio returns on factor returns."""
    betas: dict = field(default_factory=dict)   # factor_name → beta
    alpha: float = 0.0                            # daily alpha (intercept)
    r_squared: float = 0.0
    systematic_var_pct: float = 0.0               # fraction of variance explained
    idiosyncratic_var_pct: float = 0.0


# ── Core functions ─────────────────────────────────────────────────────────────

def estimate_factor_returns(
    prices_pl: pl.DataFrame,
    factor_proxies: dict | None = None,
) -> FactorReturns:
    """Compute daily factor proxy returns from ETF price data.

    For single-ETF factors: return = daily pct change of that ETF.
    For two-ETF factors (A, B): return = pct_change(A) - pct_change(B).

    Skips any factor whose required ETFs are missing from prices_pl.

    Parameters
    ----------
    prices_pl      : Polars DataFrame with columns [date, symbol, adj_close] or [date, symbol, close]
    factor_proxies : dict mapping factor name → tuple of ETF symbols. Uses FACTOR_PROXIES if None.

    Returns
    -------
    FactorReturns with .returns (date × factor) and .factor_names
    """
    if factor_proxies is None:
        factor_proxies = FACTOR_PROXIES

    # Detect price column
    price_col = "adj_close" if "adj_close" in prices_pl.columns else "close"

    # Build pandas pivot: date × symbol → price
    prices_pd = (
        prices_pl
        .select(["date", "symbol", price_col])
        .to_pandas()
        .pivot(index="date", columns="symbol", values=price_col)
        .sort_index()
    )
    prices_pd.index = pd.to_datetime(prices_pd.index)

    # Daily returns for all available symbols
    rets_all = prices_pd.pct_change()

    factor_series: dict[str, pd.Series] = {}
    for factor_name, etfs in factor_proxies.items():
        if len(etfs) == 1:
            sym = etfs[0]
            if sym not in rets_all.columns:
                continue
            factor_series[factor_name] = rets_all[sym].rename(factor_name)
        elif len(etfs) == 2:
            sym_a, sym_b = etfs
            if sym_a not in rets_all.columns or sym_b not in rets_all.columns:
                continue
            factor_series[factor_name] = (rets_all[sym_a] - rets_all[sym_b]).rename(factor_name)
        # More than 2 ETFs: skip (unsupported)

    if not factor_series:
        empty_df = pd.DataFrame()
        return FactorReturns(returns=empty_df, factor_names=[])

    factor_df = pd.DataFrame(factor_series).dropna(how="all")
    return FactorReturns(
        returns=factor_df,
        factor_names=list(factor_df.columns),
    )


def estimate_factor_betas(
    portfolio_returns: pd.Series,
    factor_returns: FactorReturns,
    lookback: int = 252,
) -> FactorBetas:
    """OLS regression: portfolio_returns ~ factor_returns.

    Uses last ``lookback`` observations where both are available.

    Returns
    -------
    FactorBetas with betas, alpha, r_squared, systematic_var_pct, idiosyncratic_var_pct
    """
    if factor_returns.returns.empty or portfolio_returns.empty:
        return FactorBetas()

    # Align
    fret = factor_returns.returns.copy()
    fret.index = pd.to_datetime(fret.index)
    pret = portfolio_returns.copy()
    pret.index = pd.to_datetime(pret.index)

    common_idx = pret.index.intersection(fret.index)
    if len(common_idx) < max(10, len(factor_returns.factor_names) + 2):
        return FactorBetas()

    pret = pret.loc[common_idx].tail(lookback)
    fret = fret.loc[pret.index].dropna(how="any")
    pret = pret.loc[fret.index]

    if len(pret) < max(10, len(factor_returns.factor_names) + 2):
        return FactorBetas()

    y = pret.values
    X = fret.values
    # Add intercept column
    X_aug = np.column_stack([np.ones(len(X)), X])

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        return FactorBetas()

    alpha = float(coeffs[0])
    betas_arr = coeffs[1:]
    betas_dict = {
        fname: float(b)
        for fname, b in zip(factor_returns.factor_names, betas_arr)
    }

    # R² and variance decomposition
    y_pred = X_aug @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-16 else 0.0

    total_var = float(np.var(y)) if np.var(y) > 1e-16 else 1.0
    systematic_var = total_var * max(r2, 0.0)
    idio_var = total_var - systematic_var

    return FactorBetas(
        betas=betas_dict,
        alpha=round(alpha, 8),
        r_squared=round(max(r2, 0.0), 6),
        systematic_var_pct=round(systematic_var / total_var, 6),
        idiosyncratic_var_pct=round(idio_var / total_var, 6),
    )


def rolling_factor_betas(
    portfolio_returns: pd.Series,
    factor_returns: FactorReturns,
    window: int = 126,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Rolling OLS betas over time.

    Returns a (date × factor_name) DataFrame. Dates align with portfolio_returns.
    """
    if factor_returns.returns.empty or portfolio_returns.empty:
        return pd.DataFrame()

    fret = factor_returns.returns.copy()
    fret.index = pd.to_datetime(fret.index)
    pret = portfolio_returns.copy()
    pret.index = pd.to_datetime(pret.index)

    common_idx = pret.index.intersection(fret.index)
    pret = pret.loc[common_idx]
    fret = fret.loc[common_idx].dropna(how="any")
    pret = pret.loc[fret.index]

    n = len(pret)
    if n < min_periods:
        return pd.DataFrame()

    factor_names = factor_returns.factor_names
    result_data: dict[str, list] = {f: [] for f in factor_names}
    result_index: list = []

    dates = pret.index.tolist()
    y_arr = pret.values
    X_arr = fret.values

    for i in range(n):
        if i + 1 < min_periods:
            for f in factor_names:
                result_data[f].append(np.nan)
            result_index.append(dates[i])
            continue

        start = max(0, i + 1 - window)
        y_w = y_arr[start : i + 1]
        X_w = X_arr[start : i + 1]

        valid_mask = ~np.isnan(X_w).any(axis=1) & ~np.isnan(y_w)
        y_w = y_w[valid_mask]
        X_w = X_w[valid_mask]

        if len(y_w) < min_periods:
            for f in factor_names:
                result_data[f].append(np.nan)
            result_index.append(dates[i])
            continue

        X_aug = np.column_stack([np.ones(len(X_w)), X_w])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_w, rcond=None)
            betas = coeffs[1:]
        except np.linalg.LinAlgError:
            betas = [np.nan] * len(factor_names)

        for f, b in zip(factor_names, betas):
            result_data[f].append(float(b))
        result_index.append(dates[i])

    return pd.DataFrame(result_data, index=result_index)
