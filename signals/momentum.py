"""Cross-sectional momentum signal — the canary strategy core.

Signal: rank all symbols in the universe by their 12-1 month momentum on
each rebalance date. Top-N by rank = long positions.

All functions are pure: (features_df, params) -> signal_df.
No I/O, no state. Testable in isolation.
"""

from datetime import date

import numpy as np
import polars as pl


def cross_sectional_momentum(
    features: pl.DataFrame,
    rebalance_dates: list[date],
    top_n: int = 5,
    min_universe_size: int = 10,
) -> pl.DataFrame:
    """Rank symbols by 12-1 momentum on each rebalance date.

    Parameters
    ----------
    features        : DataFrame with columns [date, symbol, momentum_12m_1m, ...]
    rebalance_dates : Dates on which to evaluate the signal (monthly typically)
    top_n           : Number of top-ranked symbols to include in the long book
    min_universe_size: Skip rebalance dates where fewer symbols have valid data

    Returns
    -------
    DataFrame with columns: [date, symbol, momentum_12m_1m, rank, selected]
    - rank: 1 = highest momentum (best), ascending
    - selected: True for the top_n symbols that enter the portfolio
    """
    if "momentum_12m_1m" not in features.columns:
        raise ValueError("features DataFrame must contain 'momentum_12m_1m' column")

    rows = []
    for d in rebalance_dates:
        # Snap to the closest available date on or before rebalance date
        slice_df = features.filter(pl.col("date") <= d)
        if slice_df.is_empty():
            continue
        latest_date = slice_df["date"].max()
        day_df = slice_df.filter(pl.col("date") == latest_date)

        # Drop symbols with null momentum (insufficient history)
        valid = day_df.filter(pl.col("momentum_12m_1m").is_not_null())
        if len(valid) < min_universe_size:
            continue

        # Rank descending: rank 1 = highest momentum
        ranked = valid.with_columns(
            pl.col("momentum_12m_1m")
            .rank(method="ordinal", descending=True)
            .alias("rank")
        ).with_columns(
            (pl.col("rank") <= top_n).alias("selected")
        ).select(["date", "symbol", "momentum_12m_1m", "rank", "selected"])

        # Tag with the rebalance date (may differ from latest_date on weekends)
        ranked = ranked.with_columns(pl.lit(d).cast(pl.Date).alias("rebalance_date"))
        rows.append(ranked)

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.Date, "symbol": pl.Utf8, "momentum_12m_1m": pl.Float64,
            "rank": pl.UInt32, "selected": pl.Boolean, "rebalance_date": pl.Date,
        })

    return pl.concat(rows).sort(["rebalance_date", "rank"])


def momentum_weights(
    signal: pl.DataFrame,
    weight_scheme: str = "equal",
    vol_series: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Convert a momentum signal DataFrame into a target weights DataFrame.

    Parameters
    ----------
    signal       : Output of cross_sectional_momentum()
    weight_scheme: "equal" | "vol_scaled"
    vol_series   : Required for vol_scaled; DataFrame with [date, symbol, realized_vol_21d]

    Returns
    -------
    DataFrame with columns: [rebalance_date, symbol, weight]
    Weights sum to 1.0 on each rebalance date for selected symbols.
    """
    selected = signal.filter(pl.col("selected"))

    if weight_scheme == "equal":
        # Equal weight across selected symbols
        counts = (
            selected.group_by("rebalance_date")
            .agg(pl.len().alias("n_selected"))
        )
        weights = (
            selected.join(counts, on="rebalance_date")
            .with_columns((1.0 / pl.col("n_selected")).alias("weight"))
            .select(["rebalance_date", "symbol", "weight"])
        )

    elif weight_scheme == "vol_scaled":
        if vol_series is None:
            raise ValueError("vol_series required for vol_scaled weighting")

        # Join realized vol onto selected symbols at each rebalance date
        vol_at_rebal = (
            vol_series.rename({"date": "rebalance_date"})
            .select(["rebalance_date", "symbol", "realized_vol_21d"])
        )
        joined = selected.join(vol_at_rebal, on=["rebalance_date", "symbol"], how="left")

        # Fill missing vol with universe median to avoid dropping symbols
        median_vol = joined["realized_vol_21d"].median()
        joined = joined.with_columns(
            pl.col("realized_vol_21d").fill_null(median_vol)
        )

        # Weight = (1/vol) / sum(1/vol) — inverse vol weighting
        joined = joined.with_columns(
            (1.0 / pl.col("realized_vol_21d")).alias("inv_vol")
        )
        inv_vol_sums = (
            joined.group_by("rebalance_date")
            .agg(pl.col("inv_vol").sum().alias("inv_vol_sum"))
        )
        weights = (
            joined.join(inv_vol_sums, on="rebalance_date")
            .with_columns((pl.col("inv_vol") / pl.col("inv_vol_sum")).alias("weight"))
            .select(["rebalance_date", "symbol", "weight"])
        )

    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme!r}. Use 'equal' or 'vol_scaled'")

    return weights.sort(["rebalance_date", "symbol"])


def get_monthly_rebalance_dates(start: date, end: date, trading_dates: list[date]) -> list[date]:
    """Return the first available trading day of each month in [start, end]."""
    trading_set = set(trading_dates)
    rebalance = []
    current_month = None

    for d in sorted(trading_dates):
        if d < start or d > end:
            continue
        month_key = (d.year, d.month)
        if month_key != current_month:
            current_month = month_key
            rebalance.append(d)

    return rebalance
