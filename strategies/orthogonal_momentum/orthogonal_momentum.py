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
    """Equal-weight the selected symbols from the orthogonal momentum signal.

    Only 'equal' weighting is implemented; other schemes raise ValueError.
    """
    if weight_scheme != "equal":
        raise ValueError(
            f"Unsupported weight_scheme={weight_scheme!r}; "
            "only 'equal' is implemented for Orthogonal Momentum."
        )

    selected = signal.filter(pl.col("selected"))
    if selected.is_empty():
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    counts = selected.group_by("rebalance_date").agg(pl.len().alias("n"))
    return (
        selected.join(counts, on="rebalance_date")
        .with_columns((1.0 / pl.col("n")).alias("weight"))
        .select(["rebalance_date", "symbol", "weight"])
        .sort(["rebalance_date", "symbol"])
    )
