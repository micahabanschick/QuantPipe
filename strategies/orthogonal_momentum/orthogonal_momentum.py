"""Orthogonal Momentum — market-neutral idiosyncratic momentum.

Mechanism:
  Standard cross-sectional momentum selects ETFs with the highest 12-1 month
  returns.  The problem: in a strong bull market, *all* ETFs have high momentum
  simply because the market went up.  Ranking raw momentum largely selects
  for high-beta assets during uptrends — that is beta-riding, not alpha.

  Orthogonal momentum fixes this by first removing the market's contribution:

      raw_momentum_i  =  alpha_i  +  beta_i × market_momentum  +  epsilon_i

  The residual epsilon_i is the idiosyncratic (market-neutral) momentum.
  We then rank by epsilon_i.  An ETF ranked #1 by orthogonal momentum has
  genuinely outperformed its market-beta-implied path — that is alpha.

  Empirically, idiosyncratic momentum has:
  - Lower correlation with the overall market (reduced beta exposure)
  - Better Sharpe ratio than raw momentum in most academic studies
  - More stable performance across market regimes

Signal: OLS residuals of each ETF's 12-1 month return regressed against
        SPY's 12-1 month return, computed over the full available history.
        Rank by residual descending; select top_n.

No new data required — uses the same features as the existing momentum strategies.
"""

import polars as pl

from signals.momentum import momentum_weights
from signals.orthogonal import orthogonal_cross_sectional_momentum

NAME        = "Orthogonal Momentum"
DESCRIPTION = (
    "Cross-sectional momentum with market beta removed via OLS residualisation. "
    "Selects ETFs that outperformed their beta-implied path — idiosyncratic alpha "
    "rather than beta-riding."
)
DEFAULT_PARAMS = {
    "lookback_years":   6,
    "top_n":            5,
    "cost_bps":         5.0,
    "weight_scheme":    "equal",
    "market_symbol":    "SPY",
    "min_obs":          12,    # minimum OLS observations per symbol
}


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    market_symbol: str = DEFAULT_PARAMS["market_symbol"],
    min_obs: int = DEFAULT_PARAMS["min_obs"],
    **kwargs,
) -> pl.DataFrame:
    """Rank ETFs by market-orthogonalized momentum on each rebalance date.

    Args:
        features:      Standard features DataFrame [date, symbol, momentum_12m_1m, ...].
        rebal_dates:   Rebalance dates.

    Returns:
        DataFrame with columns [date, symbol, momentum_12m_1m_ortho, rank, selected,
        rebalance_date] — same schema as cross_sectional_momentum output.
    """
    return orthogonal_cross_sectional_momentum(
        features,
        rebalance_dates=rebal_dates,
        top_n=top_n,
        market_symbol=market_symbol,
        min_obs=min_obs,
    )


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Convert orthogonal momentum signal to weights.

    Reuses the standard momentum_weights function — compatible because the
    signal DataFrame follows the same schema (has 'selected' and 'rebalance_date').
    """
    # momentum_weights expects a 'momentum_12m_1m' column for vol_scaled;
    # alias ortho column so it works with weight_scheme='equal'
    ortho_col = "momentum_12m_1m_ortho"
    if ortho_col in signal.columns and "momentum_12m_1m" not in signal.columns:
        signal = signal.rename({ortho_col: "momentum_12m_1m"})

    return momentum_weights(signal, weight_scheme=weight_scheme)
