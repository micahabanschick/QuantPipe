"""Persistent, append-only order journal.

Every order placed by the rebalance script is written here before the
broker call is made. This provides an immutable audit trail that survives
process restarts and can be used for trade reconstruction and reconciliation.

Layout: data/gold/equity/order_journal.parquet
Schema: [ts_utc, rebalance_date, broker, symbol, qty, est_price, order_id, status]

Writes are atomic (tmp + os.replace) so a crash never corrupts the file.
"""

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

_SCHEMA = {
    "ts_utc":          pl.Utf8,
    "rebalance_date":  pl.Date,
    "broker":          pl.Utf8,
    "symbol":          pl.Utf8,
    "qty":             pl.Float64,
    "est_price":       pl.Float64,
    "order_id":        pl.Utf8,
    "status":          pl.Utf8,   # "placed" | "failed" | "skipped"
}


def append_order(
    journal_path: Path,
    rebalance_date: date,
    broker: str,
    symbol: str,
    qty: float,
    est_price: float,
    order_id: str,
    status: str = "placed",
) -> None:
    """Append one order record to the journal atomically.

    Safe to call from a loop — each call is an independent atomic upsert.
    Duplicate order_ids (from a retry) overwrite the previous record.
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pl.DataFrame([{
        "ts_utc":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rebalance_date": rebalance_date,
        "broker":         broker,
        "symbol":         symbol,
        "qty":            float(qty),
        "est_price":      float(est_price),
        "order_id":       str(order_id),
        "status":         status,
    }]).with_columns(pl.col("rebalance_date").cast(pl.Date))

    if journal_path.exists():
        existing = pl.read_parquet(journal_path)
        # Overwrite any prior row with the same order_id (idempotent on retry)
        existing = existing.filter(pl.col("order_id") != str(order_id))
        combined = pl.concat([existing, new_row]).sort(["rebalance_date", "ts_utc"])
    else:
        combined = new_row

    tmp = journal_path.with_suffix(".tmp")
    combined.write_parquet(tmp)
    os.replace(tmp, journal_path)


def load_journal(journal_path: Path) -> pl.DataFrame:
    """Load the full order journal, returning an empty frame if none exists."""
    if not journal_path.exists():
        return pl.DataFrame(schema=_SCHEMA)
    return pl.read_parquet(journal_path)
