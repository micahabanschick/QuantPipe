"""Composite factor signal — weighted multi-factor cross-sectional scoring.

Pure functions, no I/O, no Streamlit. Uses Polars.

Public API:
    zscore_cross_section(features_df, factor, date_col, symbol_col) -> pl.DataFrame
    composite_score(features_df, factor_weights, rebal_dates)       -> pl.DataFrame
    composite_signal(features_df, rebal_dates, factor_weights, ...) -> pl.DataFrame
    composite_weights(signal, weight_scheme)                         -> pl.DataFrame
"""

from datetime import date as _date

import polars as pl


def zscore_cross_section(
    features_df: pl.DataFrame,
    factor: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Z-score a factor cross-sectionally within each date.

    Returns the same DataFrame with an additional column ``{factor}_z``.
    NaN values are ignored in mean/std computation (they map to null in Polars).
    """
    if factor not in features_df.columns:
        return features_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{factor}_z"))

    z_col = f"{factor}_z"
    return features_df.with_columns(
        (
            (pl.col(factor) - pl.col(factor).mean().over(date_col))
            / pl.col(factor).std().over(date_col)
        ).alias(z_col)
    )


def composite_score(
    features_df: pl.DataFrame,
    factor_weights: dict[str, float],
    rebal_dates: list,
) -> pl.DataFrame:
    """Compute a weighted composite z-score at each rebalance date.

    Algorithm
    ---------
    1. For each factor in *factor_weights*, z-score cross-sectionally per date.
    2. Weighted sum of z-scores (weights normalised to sum to 1.0).
    3. Return ``[rebalance_date, symbol, composite_score]`` using the closest
       available date <= rebal_date.

    Parameters
    ----------
    features_df   : Wide Polars DataFrame with at least [date, symbol, <factors>]
    factor_weights: {factor_name: weight} — need not sum to 1.
    rebal_dates   : List of target rebalance dates (``date`` or ``datetime``).

    Returns
    -------
    pl.DataFrame with columns [rebalance_date, symbol, composite_score]
    """
    if not factor_weights:
        return pl.DataFrame(schema={
            "rebalance_date": pl.Date,
            "symbol": pl.Utf8,
            "composite_score": pl.Float64,
        })

    # Normalise weights
    total_w = sum(abs(v) for v in factor_weights.values())
    if total_w < 1e-12:
        total_w = 1.0
    norm_weights = {f: w / total_w for f, w in factor_weights.items()}

    # Build z-scored versions for all requested factors that exist
    present = {f: w for f, w in norm_weights.items() if f in features_df.columns}
    if not present:
        return pl.DataFrame(schema={
            "rebalance_date": pl.Date,
            "symbol": pl.Utf8,
            "composite_score": pl.Float64,
        })

    df = features_df
    for factor in present:
        df = zscore_cross_section(df, factor)

    # Weighted composite on the z-scored columns
    z_cols = [f"{f}_z" for f in present]
    w_exprs = [pl.col(z) * w for z, w in zip(z_cols, present.values())]
    combined = w_exprs[0]
    for expr in w_exprs[1:]:
        combined = combined + expr

    df = df.with_columns(combined.alias("_composite_raw"))

    date_col_name = "date"
    rows = []
    for d in rebal_dates:
        # Snap to closest date on or before rebalance date
        slice_df = df.filter(pl.col(date_col_name) <= d)
        if slice_df.is_empty():
            continue
        latest = slice_df[date_col_name].max()
        day_df = slice_df.filter(pl.col(date_col_name) == latest)

        valid = day_df.filter(pl.col("_composite_raw").is_not_null())
        if valid.is_empty():
            continue

        result = (
            valid.select(["symbol", "_composite_raw"])
            .rename({"_composite_raw": "composite_score"})
            .with_columns(pl.lit(d).cast(pl.Date).alias("rebalance_date"))
            .select(["rebalance_date", "symbol", "composite_score"])
        )
        rows.append(result)

    if not rows:
        return pl.DataFrame(schema={
            "rebalance_date": pl.Date,
            "symbol": pl.Utf8,
            "composite_score": pl.Float64,
        })

    return pl.concat(rows).sort(["rebalance_date", "symbol"])


def composite_signal(
    features_df: pl.DataFrame,
    rebal_dates: list,
    factor_weights: dict[str, float],
    top_n: int = 5,
    min_universe_size: int = 8,
) -> pl.DataFrame:
    """Select top-N symbols by composite score on each rebalance date.

    Mirrors the structure of ``cross_sectional_momentum()`` output.

    Returns
    -------
    pl.DataFrame with columns [rebalance_date, symbol, composite_score, rank, selected]
    """
    scores = composite_score(features_df, factor_weights, rebal_dates)

    if scores.is_empty():
        return pl.DataFrame(schema={
            "rebalance_date": pl.Date,
            "symbol": pl.Utf8,
            "composite_score": pl.Float64,
            "rank": pl.UInt32,
            "selected": pl.Boolean,
        })

    rows = []
    for d in rebal_dates:
        day = scores.filter(pl.col("rebalance_date") == d)
        if len(day) < min_universe_size:
            continue

        ranked = day.with_columns(
            pl.col("composite_score")
            .rank(method="ordinal", descending=True)
            .alias("rank")
        ).with_columns(
            (pl.col("rank") <= top_n).alias("selected")
        )
        rows.append(ranked)

    if not rows:
        return pl.DataFrame(schema={
            "rebalance_date": pl.Date,
            "symbol": pl.Utf8,
            "composite_score": pl.Float64,
            "rank": pl.UInt32,
            "selected": pl.Boolean,
        })

    return pl.concat(rows).sort(["rebalance_date", "rank"])


def composite_weights(
    signal: pl.DataFrame,
    weight_scheme: str = "equal",
) -> pl.DataFrame:
    """Convert composite signal to portfolio weights.

    Same interface as ``momentum_weights()``.

    Parameters
    ----------
    signal        : Output of ``composite_signal()``
    weight_scheme : "equal" only (score-proportional future extension)

    Returns
    -------
    pl.DataFrame with columns [rebalance_date, symbol, weight]
    """
    selected = signal.filter(pl.col("selected"))

    if weight_scheme == "equal":
        counts = (
            selected.group_by("rebalance_date")
            .agg(pl.len().alias("n_selected"))
        )
        weights = (
            selected.join(counts, on="rebalance_date")
            .with_columns((1.0 / pl.col("n_selected")).alias("weight"))
            .select(["rebalance_date", "symbol", "weight"])
        )
    elif weight_scheme == "score":
        # Weight proportional to composite score (shift to positive first)
        min_scores = (
            selected.group_by("rebalance_date")
            .agg(pl.col("composite_score").min().alias("_min_score"))
        )
        joined = selected.join(min_scores, on="rebalance_date")
        joined = joined.with_columns(
            (pl.col("composite_score") - pl.col("_min_score") + 1e-6).alias("_adj_score")
        )
        score_sums = (
            joined.group_by("rebalance_date")
            .agg(pl.col("_adj_score").sum().alias("_score_sum"))
        )
        weights = (
            joined.join(score_sums, on="rebalance_date")
            .with_columns((pl.col("_adj_score") / pl.col("_score_sum")).alias("weight"))
            .select(["rebalance_date", "symbol", "weight"])
        )
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme!r}. Use 'equal' or 'score'")

    return weights.sort(["rebalance_date", "symbol"])
