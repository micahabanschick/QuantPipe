"""Earnings Surprise Drift — Post-Earnings Announcement Drift (PEAD) for sector ETFs.

Mechanism:
  Stocks systematically drift in the direction of their earnings surprise for
  30–60 trading days after the announcement (PEAD effect, documented since 1968).
  This strategy applies that effect at the sector level: sector ETFs whose
  representative holdings reported the largest positive earnings surprises are
  overweighted; those with the weakest surprises are underweighted or excluded.

Signal construction:
  1. For each sector ETF, use pre-loaded earnings data for its 3 proxy stocks.
  2. Compute the Standardised Unexpected Earnings (SUE) score for each proxy:
       SUE = surprise_pct (Alpha Vantage's ((actual - estimate) / |estimate|) × 100)
  3. Average the SUE scores across the sector's proxies → sector-level SUE.
  4. Only use data from the most recent quarter that has already been announced
     before the rebalance date (point-in-time safe).
  5. Rank sector ETFs by sector SUE descending.

Data dependency:
  Requires data/alt/earnings/{symbol}.parquet files populated by
  orchestration/pull_earnings.py (added as a pipeline step).
  Call load_earnings_features() before get_signal() to pre-load the data.

Rebalance frequency: monthly (inherits from the pipeline rebalance schedule).
Holding period: 42 trading days (~2 months) — typical PEAD holding window.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

from config.settings import DATA_DIR
from config.universes import SECTOR_PROXY_MAP

log = logging.getLogger(__name__)

NAME        = "Earnings Surprise Drift"
DESCRIPTION = (
    "Sector ETF rotation based on post-earnings announcement drift (PEAD). "
    "Overweights sectors whose proxy holdings reported the largest recent "
    "positive earnings surprises."
)
DEFAULT_PARAMS = {
    "lookback_years":  6,
    "top_n":           4,     # number of sector ETFs to hold
    "min_sue":         0.0,   # minimum sector SUE score to be eligible
    "cost_bps":        5.0,
    "weight_scheme":   "equal",
    "sue_window_days": 90,    # max days since earnings announcement to use
}

EARNINGS_DIR = DATA_DIR / "alt" / "earnings"


def load_earnings_features(window_days: int = DEFAULT_PARAMS["sue_window_days"]) -> pl.DataFrame:
    """Pre-load all earnings data from the cache directory.

    Returns a combined DataFrame with columns
    [date, symbol, surprise_pct] covering all proxy stocks.
    Call this once before invoking get_signal() to keep get_signal pure.
    """
    frames: list[pl.DataFrame] = []
    all_symbols = {sym for proxies in SECTOR_PROXY_MAP.values() for sym in proxies}

    for symbol in all_symbols:
        path = EARNINGS_DIR / f"{symbol.lower()}.parquet"
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
            if "date" not in df.columns or "surprise_pct" not in df.columns:
                continue
            frames.append(
                df.select(["date", "surprise_pct"])
                  .with_columns(pl.lit(symbol).alias("symbol"))
            )
        except Exception as exc:
            log.debug("load_earnings_features: could not load %s: %s", symbol, exc)

    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "surprise_pct": pl.Float64})

    return (
        pl.concat(frames)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["symbol", "date"])
    )


def _sector_sue(
    sector_etf: str,
    as_of: date,
    window_days: int,
    earnings: pl.DataFrame,
) -> float | None:
    """Return the average SUE score for a sector ETF's proxy stocks.

    Pure — uses only the passed-in earnings DataFrame.
    """
    proxies   = SECTOR_PROXY_MAP.get(sector_etf, [])
    cutoff_lo = as_of - timedelta(days=window_days)
    scores: list[float] = []

    for symbol in proxies:
        eligible = (
            earnings.filter(
                (pl.col("symbol") == symbol) &
                (pl.col("date") <= as_of) &
                (pl.col("date") >= cutoff_lo) &
                pl.col("surprise_pct").is_not_null()
            )
            .sort("date", descending=True)
        )
        if eligible.is_empty():
            continue
        scores.append(float(eligible["surprise_pct"].head(1)[0]))

    return float(sum(scores) / len(scores)) if scores else None


def get_signal(
    _features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    min_sue: float = DEFAULT_PARAMS["min_sue"],
    sue_window_days: int = DEFAULT_PARAMS["sue_window_days"],
    earnings_data: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Rank sector ETFs by their most recent aggregate earnings surprise.

    Args:
        _features:     Standard features DataFrame (unused — earnings signal
                       uses its own data injected via earnings_data).
        rebal_dates:   List of rebalance dates to generate signals for.
        earnings_data: Pre-loaded earnings DataFrame from load_earnings_features().
                       If None, loads from disk (convenience for standalone use).

    Returns:
        DataFrame with columns [date, symbol, signal_rank, sue_score].
    """
    _EMPTY = pl.DataFrame(schema={
        "date": pl.Date, "symbol": pl.Utf8,
        "signal_rank": pl.Int32, "sue_score": pl.Float64,
    })

    # Allow callers to inject pre-loaded data; fall back for standalone use
    earnings = earnings_data if earnings_data is not None else load_earnings_features(sue_window_days)

    if earnings.is_empty():
        log.warning(
            "earnings_surprise_drift: no earnings data available. "
            "Run: uv run python orchestration/pull_earnings.py"
        )
        return _EMPTY

    rows = []
    for rebal_date in rebal_dates:
        as_of = rebal_date if isinstance(rebal_date, date) else rebal_date.date()
        sector_scores: list[tuple[str, float]] = []

        for etf in SECTOR_PROXY_MAP:
            sue = _sector_sue(etf, as_of, sue_window_days, earnings)
            if sue is not None and sue >= min_sue:
                sector_scores.append((etf, sue))

        if not sector_scores:
            continue

        sector_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (etf, sue) in enumerate(sector_scores[:top_n], start=1):
            rows.append({
                "date":        as_of,
                "symbol":      etf,
                "signal_rank": rank,
                "sue_score":   round(sue, 4),
            })

    if not rows:
        return _EMPTY

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Convert earnings surprise rankings to equal portfolio weights."""
    _EMPTY = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    if signal.is_empty():
        return _EMPTY

    rows = []
    for rebal_date, group in signal.group_by("date"):
        n = len(group)
        if n == 0:
            continue
        w = 1.0 / n
        for row in group.iter_rows(named=True):
            rows.append({"date": row["date"], "symbol": row["symbol"], "weight": w})

    if not rows:
        return _EMPTY

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["date", "symbol"])
    )
