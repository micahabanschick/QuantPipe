"""Kalman Pairs Reversion — long-only mean reversion via dynamic hedge ratios.

Mechanism:
  For each pair of co-moving ETFs, fits a Kalman time-varying hedge ratio β
  and computes the spread e = price_Y - β × price_X.  When the spread is
  wide (z-score ≫ 0, meaning Y is expensive relative to X), we tilt toward X.
  When the spread is narrow (z-score ≪ 0, meaning Y is cheap), we tilt toward Y.

  Because this is a long-only portfolio, we cannot short either leg.
  Instead we express the mean-reversion view as a tilt: within each pair the
  allocation shifts between the two legs proportionally to the z-score.

Signal construction:
  1. For each pair (Y, X), extract daily closing prices.
  2. Compute daily log-returns and fit a Kalman dynamic hedge ratio β.
  3. Reconstruct the price-level spread: e_t = price_Y - β_t × price_X.
  4. Compute a rolling z-score of e over a lookback window.
  5. At each rebalance date clamp the z-score to [-2, 2] and compute pair weights:
       w_Y = 0.5 - 0.25 × clamp(z, -2, 2)   # cheap when z << 0
       w_X = 1 - w_Y
  6. Portfolio weight = pair_weight × per-pair allocation (equal across pairs).

Pairs (selected for co-integration in the equity ETF universe):
  (IWO, IWN) — Russell 2000 Growth vs Value       (size/style spread)
  (IWB, IWM) — Russell 1000 vs Russell 2000       (cap-size spread)
  (XLK, XLU) — Technology vs Utilities            (risk-on vs defensive)
  (QQQ, SPY) — Nasdaq 100 vs S&P 500              (growth vs market)
  (GLD, TLT) — Gold vs Long Bonds                 (inflation vs deflation hedge)

Data dependency:
  Requires price data injected via the `prices` kwarg (a Polars DataFrame with
  columns [date, symbol, close] — standard bronze-layer format).
  Call load_bars() and pass the result; get_signal is otherwise pure.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

log = logging.getLogger(__name__)

NAME        = "Kalman Pairs Reversion"
DESCRIPTION = (
    "Long-only mean reversion using Kalman dynamic hedge ratios. "
    "Tilts within co-moving ETF pairs toward the cheap leg when the "
    "spread z-score is extreme, and back to equal-weight when it normalises."
)
DEFAULT_PARAMS = {
    "lookback_years":   6,
    "cost_bps":         8.0,    # pairs strategies have higher turnover
    "zscore_window":    63,     # trading days for spread z-score (~3 months)
    "entry_threshold":  1.0,    # |z| above this → tilt toward cheap leg
    "max_tilt":         0.75,   # maximum allocation to one leg (clamp w_Y)
    "kalman_delta":     1e-4,   # Kalman process noise (lower = slower adaptation)
}

# Pairs: (Y_symbol, X_symbol) — Y is the dependent, X is the hedge
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("IWO", "IWN"),   # Russell 2000 Growth vs Value
    ("IWB", "IWM"),   # Russell 1000 (large) vs Russell 2000 (small)
    ("XLK", "XLU"),   # Technology vs Utilities
    ("QQQ", "SPY"),   # Nasdaq 100 vs S&P 500
    ("GLD", "TLT"),   # Gold vs Long Bonds
]


def _spread_zscore(
    price_y: pd.Series,
    price_x: pd.Series,
    delta: float,
    zscore_window: int,
) -> pd.Series:
    """Return the rolling z-score of the Kalman-filtered spread.

    Pure function — takes price Series and returns a z-score Series.
    """
    from research.kalman_filter import kalman_hedge_ratio

    # Require minimum history
    min_obs = max(zscore_window * 2, 60)
    if len(price_y) < min_obs or len(price_x) < min_obs:
        return pd.Series(0.0, index=price_y.index)

    # Align series
    aligned = pd.concat([price_y.rename("Y"), price_x.rename("X")], axis=1).dropna()
    if len(aligned) < min_obs:
        return pd.Series(0.0, index=price_y.index)

    ret_y = np.log(aligned["Y"]).diff().dropna()
    ret_x = np.log(aligned["X"]).diff().dropna()

    betas, _ = kalman_hedge_ratio(
        pd.Series(ret_y.values, index=ret_y.index),
        pd.Series(ret_x.values, index=ret_x.index),
        delta=delta,
    )

    if betas.size == 0:
        return pd.Series(0.0, index=price_y.index)

    # Reconstruct price-level spread (beta aligned to price dates after dropna)
    beta_s = pd.Series(betas, index=ret_y.index)
    spread = aligned["Y"].loc[beta_s.index] - beta_s.values * aligned["X"].loc[beta_s.index]

    # Rolling z-score
    roll_mu = spread.rolling(zscore_window, min_periods=zscore_window // 2).mean()
    roll_sd = spread.rolling(zscore_window, min_periods=zscore_window // 2).std()
    z = (spread - roll_mu) / roll_sd.replace(0, np.nan)

    return z.reindex(price_y.index).fillna(0.0)


def get_signal(
    _features: pl.DataFrame,
    rebal_dates: list,
    zscore_window: int = DEFAULT_PARAMS["zscore_window"],
    entry_threshold: float = DEFAULT_PARAMS["entry_threshold"],
    max_tilt: float = DEFAULT_PARAMS["max_tilt"],
    kalman_delta: float = DEFAULT_PARAMS["kalman_delta"],
    pairs: list[tuple[str, str]] | None = None,
    prices: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Compute per-pair spread z-scores and derive long-only tilt signals.

    Args:
        _features:       Standard features DataFrame (unused).
        rebal_dates:     Rebalance dates to generate signals for.
        prices:          Bronze-layer OHLCV DataFrame [date, symbol, close].
                         Must be provided by the caller (pure function contract).
        pairs:           List of (Y, X) symbol pairs. Defaults to DEFAULT_PAIRS.

    Returns:
        DataFrame with columns [date, symbol, weight, zscore].
    """
    _EMPTY = pl.DataFrame(schema={
        "date": pl.Date, "symbol": pl.Utf8,
        "weight": pl.Float64, "zscore": pl.Float64,
    })

    if prices is None:
        log.warning(
            "kalman_pairs_reversion: prices not provided — pass load_bars() output "
            "via the prices= kwarg. Returning empty signal."
        )
        return _EMPTY

    active_pairs = pairs or DEFAULT_PAIRS
    min_tilt = 1.0 - max_tilt

    # Build price pivot: date × symbol → close
    if "adj_close" in prices.columns:
        price_col = "adj_close"
    elif "close" in prices.columns:
        price_col = "close"
    else:
        log.warning("kalman_pairs_reversion: prices has neither 'adj_close' nor 'close'")
        return _EMPTY
    price_wide = (
        prices.select(["date", "symbol", price_col])
              .to_pandas()
              .pivot(index="date", columns="symbol", values=price_col)
    )
    price_wide.index = pd.to_datetime(price_wide.index)
    price_wide = price_wide.sort_index()

    # Pre-compute z-score series for every pair
    zscore_cache: dict[tuple[str, str], pd.Series] = {}
    for y_sym, x_sym in active_pairs:
        if y_sym not in price_wide.columns or x_sym not in price_wide.columns:
            log.debug("kalman_pairs_reversion: missing price data for pair (%s, %s)", y_sym, x_sym)
            continue
        zscore_cache[(y_sym, x_sym)] = _spread_zscore(
            price_wide[y_sym].dropna(),
            price_wide[x_sym].dropna(),
            delta=kalman_delta,
            zscore_window=zscore_window,
        )

    if not zscore_cache:
        return _EMPTY

    n_pairs    = len(zscore_cache)
    pair_alloc = 1.0 / n_pairs   # equal allocation across pairs

    rows = []
    for rebal_date in rebal_dates:
        as_of = pd.Timestamp(rebal_date if isinstance(rebal_date, date) else rebal_date.date())

        for (y_sym, x_sym), z_series in zscore_cache.items():
            # Nearest available z-score on or before rebalance date
            available = z_series[z_series.index <= as_of]
            if available.empty:
                z = 0.0
            else:
                z = float(available.iloc[-1])

            # Dead zone: if |z| < entry_threshold, hold equal weight (no conviction)
            if abs(z) < entry_threshold:
                w_y = 0.5
            else:
                # Clamp z to [-2, 2] and convert to weight tilt
                z_clamped = max(-2.0, min(2.0, z))
                raw_w_y   = 0.5 - 0.25 * z_clamped      # ranges 0.0 → 1.0
                w_y       = max(min_tilt, min(max_tilt, raw_w_y))
            w_x = 1.0 - w_y

            rows.extend([
                {"date": rebal_date if isinstance(rebal_date, date) else rebal_date.date(),
                 "symbol": y_sym, "weight": round(w_y * pair_alloc, 6), "zscore": round(z, 4)},
                {"date": rebal_date if isinstance(rebal_date, date) else rebal_date.date(),
                 "symbol": x_sym, "weight": round(w_x * pair_alloc, 6), "zscore": round(-z, 4)},
            ])

    if not rows:
        return _EMPTY

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["date", "symbol"])
    )


def get_weights(
    signal: pl.DataFrame,
    **kwargs,
) -> pl.DataFrame:
    """Pass through weights pre-computed in get_signal.

    The weight tilt is already embedded in the signal; this function simply
    normalises across the full portfolio so weights sum to 1.
    """
    _EMPTY = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    if signal.is_empty() or "weight" not in signal.columns:
        return _EMPTY

    rows = []
    for rebal_date, group in signal.group_by("date"):
        total = group["weight"].sum()
        if total < 1e-10:
            continue
        for row in group.iter_rows(named=True):
            rows.append({
                "date":   row["date"],
                "symbol": row["symbol"],
                "weight": round(row["weight"] / total, 6),
            })

    if not rows:
        return _EMPTY

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["date", "symbol"])
    )
