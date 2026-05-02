"""Pull fixed macro indicator series from FRED for the regime classifier.

Fetches 5 core macro indicators unconditionally whenever FRED_API_KEY is set.
Unlike pull_fred.py (which runs user-scheduled series), this always fetches
the specific series needed by research/regime_classifier.py.

Saved to: data/alt/macro/{series_id.lower()}.parquet

Series fetched:
  INDPRO    — Industrial Production Index (monthly)
  CPIAUCSL  — CPI All Urban Consumers (monthly)
  UNRATE    — Civilian Unemployment Rate (monthly)
  T10Y2Y    — 10Y-2Y Treasury Spread / yield curve (daily → resampled monthly)
  FEDFUNDS  — Federal Funds Effective Rate (monthly)

Cache: 7-day refresh — monthly data but we want it current.

Usage (standalone):
    uv run python orchestration/pull_macro.py

Pipeline integration:
    Called by run_pipeline.py as Step 4 when FRED_API_KEY is set.
    Best-effort — failures do not abort the pipeline.
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import DATA_DIR, FRED_API_KEY, LOGS_DIR

log = logging.getLogger(__name__)

MACRO_DIR  = DATA_DIR / "alt" / "macro"
CACHE_DAYS = 7   # re-fetch when data is older than this

# Fixed series required by the regime classifier
MACRO_SERIES: dict[str, str] = {
    "INDPRO":   "Industrial Production Index (monthly)",
    "CPIAUCSL": "CPI All Urban Consumers (monthly)",
    "UNRATE":   "Unemployment Rate (monthly)",
    "T10Y2Y":   "10Y-2Y Treasury Spread (daily)",
    "FEDFUNDS": "Federal Funds Rate (monthly)",
}

# Start date for initial full backfill
_BACKFILL_FROM = "1990-01-01"


def _is_fresh(series_id: str) -> bool:
    path = MACRO_DIR / f"{series_id.lower()}.parquet"
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    return age < CACHE_DAYS


def _fetch_series(series_id: str, fred) -> bool:
    """Fetch one FRED series, upsert with existing data, and save. Returns success."""
    import polars as pl  # noqa: PLC0415 — local import keeps module-level imports minimal

    path = MACRO_DIR / f"{series_id.lower()}.parquet"

    # Determine start date: extend existing or full backfill
    if path.exists():
        try:
            existing = pl.read_parquet(path)
            start    = str(existing["date"].max())
        except Exception:
            start = _BACKFILL_FROM
    else:
        start = _BACKFILL_FROM

    try:
        df = fred.get_series(series_id, start=start, end=str(date.today()))
        if df.is_empty():
            log.warning("pull_macro: no data returned for %s", series_id)
            return False

        # Monthly resample for daily series (T10Y2Y)
        if series_id == "T10Y2Y":
            import pandas as _pd
            raw_col = series_id if series_id in df.columns else "value"
            pd_df = (
                df.rename({raw_col: "value"} if raw_col != "value" else {})
                  .to_pandas()
                  .assign(date=lambda x: _pd.to_datetime(x["date"]))
                  .set_index("date")
                  .resample("MS")["value"]
                  .last()
                  .reset_index()
            )
            df = pl.from_pandas(pd_df).with_columns(pl.col("date").cast(pl.Date))
        elif series_id in df.columns:
            df = df.rename({series_id: "value"})

        # Upsert with existing data
        if path.exists():
            old = pl.read_parquet(path)
            df  = pl.concat([old, df]).unique(subset=["date"], keep="last").sort("date")

        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        log.info("pull_macro: %s — %d rows saved", series_id, len(df))
        return True

    except Exception as exc:
        log.warning("pull_macro: %s failed — %s", series_id, exc)
        return False


def main() -> int:
    if not FRED_API_KEY:
        log.info("pull_macro: FRED_API_KEY not set — skipping")
        return 0

    from data_adapters.fred_adapter import FREDAdapter
    fred  = FREDAdapter(FRED_API_KEY)
    stale = [sid for sid in MACRO_SERIES if not _is_fresh(sid)]
    fresh = len(MACRO_SERIES) - len(stale)

    log.info(
        "pull_macro: %d series — %d fresh, %d to fetch",
        len(MACRO_SERIES), fresh, len(stale),
    )

    failures = sum(not _fetch_series(sid, fred) for sid in stale)

    status = f"{failures} failure(s)" if failures else "complete"
    log.info("pull_macro: %s (%d fetched, %d fresh)", status, len(stale), fresh)
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "macro.log"),
        ],
    )
    raise SystemExit(main())
