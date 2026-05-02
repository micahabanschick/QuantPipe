"""Signal orthogonalization — remove market beta from cross-sectional signals.

Replaces raw momentum (which is correlated with market beta) with
idiosyncratic momentum: the part of each ETF's return that is NOT
explained by the overall market move.

Method: for each symbol, fit an OLS time-series regression

    signal_i,t  =  alpha_i  +  beta_i × market_signal_t  +  epsilon_i,t

and use epsilon_i,t as the orthogonalized signal.  Symbols with high
residual momentum are outperforming in ways the market doesn't explain —
that is genuine alpha rather than beta-riding.

All functions are pure: (features_df, params) -> signal_df.  No I/O.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def orthogonalize_signal(
    features: pl.DataFrame,
    signal_col: str = "momentum_12m_1m",
    market_symbol: str = "SPY",
    min_obs: int = 12,
) -> pl.DataFrame:
    """Residualize a cross-sectional signal against the market's same signal.

    For each symbol, fits a full-history OLS regression of its time series
    against the market symbol's time series and returns the residuals.
    The residual is the idiosyncratic component that is uncorrelated with beta.

    Parameters
    ----------
    features      : DataFrame with [date, symbol, <signal_col>, ...].
                    Must include the market_symbol in the symbol column.
    signal_col    : Feature column to orthogonalize (default: momentum_12m_1m).
    market_symbol : The benchmark used as the market factor (default: SPY).
    min_obs       : Minimum overlapping observations required to fit OLS;
                    symbols with fewer observations fall back to raw signal.

    Returns
    -------
    DataFrame with original columns plus `{signal_col}_ortho` containing
    the OLS residuals.  Market symbol rows have ortho = 0.0 (perfectly
    explained by itself).
    """
    if signal_col not in features.columns:
        raise ValueError(f"features must contain '{signal_col}' column")

    # Extract market time series
    mkt_df = (
        features.filter(pl.col("symbol") == market_symbol)
        .select(["date", signal_col])
        .rename({signal_col: "_mkt"})
    )

    if mkt_df.is_empty():
        log.warning(
            "orthogonalize_signal: market_symbol=%s not found in features; "
            "returning raw signal unchanged",
            market_symbol,
        )
        return features.with_columns(pl.col(signal_col).alias(f"{signal_col}_ortho"))

    ortho_col = f"{signal_col}_ortho"
    result_frames: list[pl.DataFrame] = []

    for (symbol,), grp in features.group_by("symbol"):
        # Left join so we retain all symbol dates; rows without SPY data get null _mkt
        aligned = (
            grp.select(["date", "symbol"] + [c for c in features.columns
                                              if c not in ("date", "symbol")])
               .join(mkt_df, on="date", how="left")
        )

        y = aligned[signal_col].to_numpy().astype(float)
        x = aligned["_mkt"].to_numpy().astype(float)

        valid = np.isfinite(y) & np.isfinite(x)

        if symbol == market_symbol or valid.sum() < min_obs:
            # Market explains itself perfectly; or insufficient history → raw signal
            residuals = np.where(symbol == market_symbol, 0.0, y)
        else:
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            # Initialize to NaN; compute residuals only where both inputs are finite
            residuals = np.full_like(y, np.nan, dtype=float)
            residuals[valid] = y[valid] - (slope * x[valid] + intercept)

        result_frames.append(
            aligned.drop("_mkt").with_columns(
                pl.Series(name=ortho_col, values=residuals)
            )
        )

    if not result_frames:
        return features.with_columns(pl.lit(None).cast(pl.Float64).alias(ortho_col))

    return pl.concat(result_frames).sort(["date", "symbol"])


def orthogonal_cross_sectional_momentum(
    features: pl.DataFrame,
    rebalance_dates: list[date],
    top_n: int = 5,
    market_symbol: str = "SPY",
    min_universe_size: int = 10,
    min_obs: int = 12,
) -> pl.DataFrame:
    """Rank symbols by market-orthogonalized momentum on each rebalance date.

    Parameters
    ----------
    features          : DataFrame with [date, symbol, momentum_12m_1m, ...].
    rebalance_dates   : Dates on which to evaluate the signal.
    top_n             : Number of top-ranked symbols for the long book.
    market_symbol     : Benchmark to orthogonalize against.
    min_universe_size : Skip rebalance dates with fewer valid symbols.
    min_obs           : Minimum time-series obs for OLS fit.

    Returns
    -------
    DataFrame with columns: [date, symbol, momentum_12m_1m_ortho, rank, selected,
                              rebalance_date].
    """
    ortho_col = "momentum_12m_1m_ortho"

    # Compute orthogonalized signal across the full features history
    ortho_features = orthogonalize_signal(
        features, signal_col="momentum_12m_1m",
        market_symbol=market_symbol, min_obs=min_obs,
    )

    rows: list[pl.DataFrame] = []
    for d in rebalance_dates:
        snap = ortho_features.filter(pl.col("date") <= d)
        if snap.is_empty():
            continue
        latest = snap["date"].max()
        day_df = snap.filter(pl.col("date") == latest)

        # Filter both Polars nulls and NumPy NaN floats, then exclude market benchmark
        valid = day_df.filter(
            pl.col(ortho_col).is_not_null() & pl.col(ortho_col).is_finite() &
            (pl.col("symbol") != market_symbol)
        )

        if len(valid) < min_universe_size:
            continue

        ranked = (
            valid.with_columns(
                pl.col(ortho_col)
                .rank(method="ordinal", descending=True)
                .alias("rank")
            )
            .with_columns((pl.col("rank") <= top_n).alias("selected"))
            .select(["date", "symbol", ortho_col, "rank", "selected"])
            .with_columns(pl.lit(d).cast(pl.Date).alias("rebalance_date"))
        )
        rows.append(ranked)

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.Date, "symbol": pl.Utf8,
            ortho_col: pl.Float64, "rank": pl.UInt32,
            "selected": pl.Boolean, "rebalance_date": pl.Date,
        })

    return pl.concat(rows).sort(["rebalance_date", "rank"])
