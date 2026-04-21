"""Per-broker NAV snapshot log — written after every rebalance.

Schema: data/gold/equity/trading_history.parquet
  ts_utc          : str   — ISO timestamp of snapshot
  date            : Date  — as-of date
  broker          : str   — "paper" | "ibkr" | "ccxt"
  nav             : f64   — total portfolio value in USD
  cash            : f64   — uninvested cash
  n_positions     : i32   — number of open positions
"""

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

_SCHEMA = {
    "ts_utc":      pl.Utf8,
    "date":        pl.Date,
    "broker":      pl.Utf8,
    "nav":         pl.Float64,
    "cash":        pl.Float64,
    "n_positions": pl.Int32,
}


def append_nav_snapshot(
    path: Path,
    as_of: date,
    broker: str,
    nav: float,
    cash: float,
    n_positions: int,
) -> None:
    """Atomically upsert one NAV snapshot (one row per broker per date)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pl.DataFrame([{
        "ts_utc":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date":        as_of,
        "broker":      broker,
        "nav":         float(nav),
        "cash":        float(cash),
        "n_positions": int(n_positions),
    }]).with_columns(pl.col("date").cast(pl.Date))

    if path.exists():
        existing = pl.read_parquet(path)
        existing = existing.filter(
            ~((pl.col("date") == as_of) & (pl.col("broker") == broker))
        )
        combined = pl.concat([existing, new_row]).sort(["broker", "date"])
    else:
        combined = new_row

    tmp = path.with_suffix(".tmp")
    combined.write_parquet(tmp)
    os.replace(tmp, path)
    log.info(f"NAV snapshot written: broker={broker} date={as_of} nav=${nav:,.0f}")


def load_nav_snapshots(path: Path, broker: str | None = None):
    """Return trading history as a pandas DataFrame, optionally filtered by broker."""
    if not path.exists():
        import pandas as pd
        return pd.DataFrame(columns=list(_SCHEMA.keys()))
    df = pl.read_parquet(path)
    if broker:
        df = df.filter(pl.col("broker") == broker)
    return df.sort("date").to_pandas()
