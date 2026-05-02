"""Earnings Surprise Drift — Post-Earnings Announcement Drift (PEAD) for sector ETFs.

Mechanism:
  Stocks systematically drift in the direction of their earnings surprise for
  30–60 trading days after the announcement (PEAD effect, documented since 1968).
  This strategy applies that effect at the sector level: sector ETFs whose
  representative holdings reported the largest positive earnings surprises are
  overweighted; those with the weakest surprises are underweighted or excluded.

Signal construction:
  1. For each sector ETF, load cached earnings data for its 3 proxy stocks.
  2. Compute the Standardised Unexpected Earnings (SUE) score for each proxy:
       SUE = surprise_pct (Alpha Vantage's ((actual - estimate) / |estimate|) × 100)
  3. Average the SUE scores across the sector's proxies → sector-level SUE.
  4. Only use data from the most recent quarter that has already been announced
     before the rebalance date (point-in-time safe).
  5. Rank sector ETFs by sector SUE descending.

Data dependency:
  Requires data/alt/earnings/{symbol}.parquet files populated by
  orchestration/pull_earnings.py (added as a pipeline step).

Rebalance frequency: monthly (inherits from the pipeline rebalance schedule).
Holding period: 42 trading days (~2 months) — typical PEAD holding window.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from config.settings import DATA_DIR
from orchestration.pull_earnings import SECTOR_PROXY_MAP

log = logging.getLogger(__name__)

NAME        = "Earnings Surprise Drift"
DESCRIPTION = (
    "Sector ETF rotation based on post-earnings announcement drift (PEAD). "
    "Overweights sectors whose proxy holdings reported the largest recent "
    "positive earnings surprises."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          4,      # number of sector ETFs to hold
    "min_sue":        0.0,    # minimum sector SUE score to be eligible
    "cost_bps":       5.0,
    "weight_scheme":  "equal",
    "sue_window_days": 90,    # max days since earnings announcement to use
}

EARNINGS_DIR = DATA_DIR / "alt" / "earnings"


def _load_sector_sue(sector_etf: str, as_of: date, window_days: int) -> float | None:
    """Return the average SUE score for a sector ETF's proxy stocks.

    Only uses earnings announced on or before as_of and within window_days.
    Returns None if no qualifying data exists.
    """
    proxies = SECTOR_PROXY_MAP.get(sector_etf, [])
    scores: list[float] = []

    cutoff_early = as_of - timedelta(days=window_days)

    for symbol in proxies:
        path = EARNINGS_DIR / f"{symbol.lower()}.parquet"
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
            if "date" not in df.columns or "surprise_pct" not in df.columns:
                continue

            # Point-in-time safe: only use announcements before the rebalance date
            eligible = df.filter(
                (pl.col("date") <= as_of) &
                (pl.col("date") >= cutoff_early) &
                pl.col("surprise_pct").is_not_null()
            ).sort("date", descending=True)

            if eligible.is_empty():
                continue

            # Most recent eligible quarter
            latest_sue = float(eligible["surprise_pct"].head(1)[0])
            scores.append(latest_sue)

        except Exception as exc:
            log.debug("earnings_surprise_drift: could not load %s: %s", symbol, exc)

    return float(sum(scores) / len(scores)) if scores else None


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    min_sue: float = DEFAULT_PARAMS["min_sue"],
    sue_window_days: int = DEFAULT_PARAMS["sue_window_days"],
    **kwargs,
) -> pl.DataFrame:
    """Rank sector ETFs by their most recent aggregate earnings surprise.

    Returns a signal DataFrame with columns [date, symbol, signal_rank, sue_score].
    Falls back to an empty frame if no earnings data is available.
    """
    available_sectors = list(SECTOR_PROXY_MAP.keys())

    # Check that at least some earnings data exists
    any_data = any(
        (EARNINGS_DIR / f"{sym.lower()}.parquet").exists()
        for proxies in SECTOR_PROXY_MAP.values()
        for sym in proxies
    )
    if not any_data:
        log.warning(
            "earnings_surprise_drift: no earnings cache found. "
            "Run: uv run python orchestration/pull_earnings.py"
        )
        return pl.DataFrame(schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signal_rank": pl.Int32, "sue_score": pl.Float64,
        })

    rows = []
    for rebal_date in rebal_dates:
        as_of = rebal_date if isinstance(rebal_date, date) else rebal_date.date()
        sector_scores: list[tuple[str, float]] = []

        for etf in available_sectors:
            sue = _load_sector_sue(etf, as_of, sue_window_days)
            if sue is not None and sue >= min_sue:
                sector_scores.append((etf, sue))

        if not sector_scores:
            log.debug("earnings_surprise_drift: no qualifying sectors on %s", as_of)
            continue

        # Rank descending by SUE — top_n sectors become the portfolio
        sector_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (etf, sue) in enumerate(sector_scores[:top_n], start=1):
            rows.append({
                "date":        as_of,
                "symbol":      etf,
                "signal_rank": rank,
                "sue_score":   round(sue, 4),
            })

    if not rows:
        return pl.DataFrame(schema={
            "date": pl.Date, "symbol": pl.Utf8,
            "signal_rank": pl.Int32, "sue_score": pl.Float64,
        })

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Convert earnings surprise rankings to portfolio weights.

    Currently supports equal-weight only. All selected sectors receive
    an equal allocation; unselected sectors receive zero.
    """
    if signal.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    rows = []
    for (rebal_date,), group in signal.group_by("date"):
        n = len(group)
        if n == 0:
            continue
        w = 1.0 / n
        for row in group.iter_rows(named=True):
            rows.append({
                "date":   row["date"],
                "symbol": row["symbol"],
                "weight": w,
            })

    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["date", "symbol"])
    )
