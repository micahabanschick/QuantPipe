"""Return attribution — factor and sector decomposition of portfolio P&L.

Pure analytics. Returns structured data for dashboard rendering.
No I/O, no Streamlit, no Plotly.

Public API:
    factor_return_attribution(portfolio_returns, factor_returns, betas) -> FactorAttribution
    sector_return_contribution(weights_history, prices_pl, sector_map)  -> SectorAttribution
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from risk.factor_model import FactorReturns, FactorBetas


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FactorAttribution:
    """Daily and cumulative factor P&L contributions."""
    daily: pd.DataFrame       # (date × factor_name + "Residual") daily contributions
    cumulative: pd.DataFrame  # cumulative sum of daily contributions
    total_return: float
    factor_names: list = field(default_factory=list)


@dataclass
class SectorAttribution:
    """Period sector contribution to portfolio returns."""
    sector_contributions: dict = field(default_factory=dict)  # sector → cumulative return contribution
    total_return: float = 0.0


# ── Factor attribution ────────────────────────────────────────────────────────

def factor_return_attribution(
    portfolio_returns: pd.Series,
    factor_returns: FactorReturns,
    betas: FactorBetas,
) -> FactorAttribution:
    """Decompose portfolio returns into factor contributions.

    daily_contribution[factor] = beta[factor] * factor_return[date]
    residual[date] = portfolio_return[date] - sum(contributions[date])

    Aligns dates between portfolio_returns and factor_returns.

    Parameters
    ----------
    portfolio_returns : pd.Series of daily portfolio returns (DatetimeIndex)
    factor_returns    : FactorReturns from estimate_factor_returns()
    betas             : FactorBetas from estimate_factor_betas()

    Returns
    -------
    FactorAttribution with .daily and .cumulative DataFrames
    """
    empty = FactorAttribution(
        daily=pd.DataFrame(),
        cumulative=pd.DataFrame(),
        total_return=0.0,
        factor_names=[],
    )

    if factor_returns.returns.empty or not betas.betas or portfolio_returns.empty:
        return empty

    fret = factor_returns.returns.copy()
    fret.index = pd.to_datetime(fret.index)
    pret = portfolio_returns.copy()
    pret.index = pd.to_datetime(pret.index)

    # Align to common dates
    common_idx = pret.index.intersection(fret.index)
    if len(common_idx) < 2:
        return empty

    pret = pret.loc[common_idx]
    fret = fret.loc[common_idx]

    factor_names = [f for f in factor_returns.factor_names if f in betas.betas]
    if not factor_names:
        return empty

    # daily contributions
    contrib: dict[str, pd.Series] = {}
    total_factor_contrib = pd.Series(0.0, index=common_idx)

    for fname in factor_names:
        beta = betas.betas[fname]
        if fname in fret.columns:
            c = fret[fname] * beta
        else:
            c = pd.Series(0.0, index=common_idx)
        contrib[fname] = c
        total_factor_contrib = total_factor_contrib + c.fillna(0.0)

    contrib["Residual"] = pret - total_factor_contrib

    daily_df = pd.DataFrame(contrib, index=common_idx)
    cum_df   = daily_df.cumsum()

    # Total return (compounded)
    total_return = float((1 + pret).prod() - 1)

    all_names = factor_names + ["Residual"]
    return FactorAttribution(
        daily=daily_df,
        cumulative=cum_df,
        total_return=round(total_return, 6),
        factor_names=all_names,
    )


# ── Sector attribution ────────────────────────────────────────────────────────

def sector_return_contribution(
    weights_history: pl.DataFrame,
    prices_pl: pl.DataFrame,
    sector_map: dict,
) -> SectorAttribution:
    """Compute each sector's contribution to portfolio returns.

    For each rebalance period [t, t+1]:
      contribution = sum over symbols in sector of (weight * return_over_period)
    Aggregates across all periods for cumulative sector contribution.

    Parameters
    ----------
    weights_history : Polars DataFrame [rebalance_date, symbol, weight]
    prices_pl       : Polars DataFrame [date, symbol, adj_close | close]
    sector_map      : {symbol: sector_name}

    Returns
    -------
    SectorAttribution with sector_contributions and total_return
    """
    if weights_history.is_empty() or prices_pl.is_empty():
        return SectorAttribution()

    price_col = "adj_close" if "adj_close" in prices_pl.columns else "close"

    # Build price pivot: date × symbol
    prices_pd = (
        prices_pl
        .select(["date", "symbol", price_col])
        .to_pandas()
        .pivot(index="date", columns="symbol", values=price_col)
        .sort_index()
    )
    prices_pd.index = pd.to_datetime(prices_pd.index)

    # Get sorted rebalance dates
    wh_pd = weights_history.to_pandas()
    if "rebalance_date" in wh_pd.columns:
        date_col = "rebalance_date"
    else:
        date_col = "date"

    wh_pd[date_col] = pd.to_datetime(wh_pd[date_col])
    rebal_dates = sorted(wh_pd[date_col].unique())

    sector_contrib: dict[str, float] = {}
    portfolio_total: float = 0.0

    for i, rd in enumerate(rebal_dates):
        # Determine end of this period
        if i + 1 < len(rebal_dates):
            next_rd = rebal_dates[i + 1]
        else:
            next_rd = prices_pd.index.max()

        # Weights at this rebalance date
        wts = wh_pd[wh_pd[date_col] == rd].set_index("symbol")["weight"]

        for sym, w in wts.items():
            # Find closest price on or after rebalance date and at period end
            sym_prices = prices_pd.get(sym)
            if sym_prices is None:
                continue

            avail = sym_prices.dropna()
            if avail.empty:
                continue

            # Start: closest date >= rd
            start_dates = avail.index[avail.index >= pd.Timestamp(rd)]
            if start_dates.empty:
                continue
            p_start = float(avail.loc[start_dates[0]])

            # End: closest date <= next_rd
            end_dates = avail.index[avail.index <= pd.Timestamp(next_rd)]
            if end_dates.empty:
                continue
            p_end = float(avail.loc[end_dates[-1]])

            if p_start < 1e-10:
                continue

            period_ret = p_end / p_start - 1.0
            contribution = float(w) * period_ret

            sector = sector_map.get(str(sym), "Other")
            sector_contrib[sector] = sector_contrib.get(sector, 0.0) + contribution
            portfolio_total += contribution

    return SectorAttribution(
        sector_contributions={k: round(v, 6) for k, v in sorted(
            sector_contrib.items(), key=lambda x: -abs(x[1])
        )},
        total_return=round(portfolio_total, 6),
    )
