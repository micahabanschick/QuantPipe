"""Persistent, append-only order journal.

Every order placed by the rebalance script is written here before the
broker call is made. This provides an immutable audit trail that survives
process restarts and can be used for trade reconstruction and reconciliation.

Layout: data/gold/equity/order_journal.parquet
Schema: [ts_utc, rebalance_date, broker, symbol, qty, est_price, fill_price,
         order_id, status]

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
    "fill_price":      pl.Float64,  # actual broker fill price; None until confirmed
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
        "qty":            float(qty),       # explicit cast: guards against numpy/Decimal
        "est_price":      float(est_price),
        "fill_price":     None,
        "order_id":       str(order_id),    # explicit cast: guards against UUID objects
        "status":         status,
    }]).with_columns(
        pl.col("rebalance_date").cast(pl.Date),
        pl.col("fill_price").cast(pl.Float64),
    )

    if journal_path.exists():
        existing = load_journal(journal_path)
        existing = existing.filter(pl.col("order_id") != str(order_id))
        combined = pl.concat([existing, new_row]).sort(["rebalance_date", "ts_utc"])
    else:
        combined = new_row

    tmp = journal_path.with_suffix(".tmp")
    combined.write_parquet(tmp)
    os.replace(tmp, journal_path)


def batch_update_fill_prices(
    journal_path: Path,
    fills: dict[str, float],
) -> None:
    """Write actual broker fill prices back to the journal in a single atomic write.

    Args:
        fills: mapping of {order_id: avg_fill_price} from broker.get_fills().

    Reads the journal once, patches all matching order_ids in-memory, then
    writes back once — O(1) I/O regardless of how many fills there are.
    """
    if not fills or not journal_path.exists():
        return

    df = load_journal(journal_path)

    for order_id, fill_price in fills.items():
        mask = pl.col("order_id") == order_id
        if df.filter(mask).is_empty():
            log.warning("batch_update_fill_prices: order_id %s not in journal", order_id)
            continue
        df = df.with_columns(
            pl.when(mask)
            .then(pl.lit(fill_price))
            .otherwise(pl.col("fill_price"))
            .alias("fill_price")
        )

    tmp = journal_path.with_suffix(".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, journal_path)


def load_journal(journal_path: Path) -> pl.DataFrame:
    """Load the full order journal, returning an empty frame if none exists.

    Adds fill_price column (all null) for records that pre-date that column.
    """
    if not journal_path.exists():
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.read_parquet(journal_path)
    if "fill_price" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("fill_price"))
    return df
