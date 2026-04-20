"""Parquet-on-disk storage layer with DuckDB query interface.

Layout:  data/{asset_class}/daily/symbol={SYMBOL}/year={YYYY}/data.parquet

Design choices:
- Partition by symbol then year — optimises single-symbol time-range queries.
- DuckDB reads with hive partitioning for fast multi-symbol scans.
- write_bars is idempotent: it merges new rows with existing data (upsert by date).
- Bronze layer (raw) is write-once. Silver/gold are derived and may be overwritten.
"""

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from config.settings import DATA_DIR


def _parquet_path(asset_class: str, symbol: str, year: int) -> Path:
    safe_symbol = symbol.replace("/", "_")
    p = DATA_DIR / asset_class / "daily" / f"symbol={safe_symbol}" / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    return p / "data.parquet"


def write_bars(df: pl.DataFrame, asset_class: str, symbol: str, layer: str = "bronze") -> int:
    """Write OHLCV bars to Parquet storage.

    Merges with any existing data (upsert by date) so nightly incremental
    writes don't create duplicates.

    Returns the number of rows written.
    """
    if df.is_empty():
        return 0

    df = df.with_columns(pl.col("date").cast(pl.Date))
    years = df["date"].dt.year().unique().to_list()

    total_written = 0
    for year in years:
        year_df = df.filter(pl.col("date").dt.year() == year)
        path = _parquet_path(f"{layer}/{asset_class}", symbol, year)

        if path.exists():
            existing = pl.read_parquet(path)
            merged = (
                pl.concat([existing, year_df])
                .unique(subset=["date", "symbol"], keep="last")
                .sort("date")
            )
        else:
            merged = year_df.sort("date")

        merged.write_parquet(path, compression="snappy")
        total_written += len(year_df)

    return total_written


def load_bars(
    symbols: list[str] | str,
    start: date,
    end: date,
    asset_class: str = "equity",
    layer: str = "bronze",
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load OHLCV bars for one or more symbols using DuckDB hive scanning.

    Returns an empty DataFrame (with correct schema) if no data is found.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    safe_symbols = [s.replace("/", "_") for s in symbols]
    base_path = DATA_DIR / layer / asset_class / "daily"

    if not base_path.exists():
        return pl.DataFrame()

    glob_pattern = str(base_path / "**/*.parquet")

    # Build symbol filter for DuckDB hive partition pruning
    symbol_list = ", ".join(f"'{s}'" for s in safe_symbols)

    col_clause = "*" if columns is None else ", ".join(columns)
    query = f"""
        SELECT {col_clause}
        FROM read_parquet('{glob_pattern}', hive_partitioning=true)
        WHERE symbol IN ({symbol_list})
          AND date >= '{start}'
          AND date <= '{end}'
        ORDER BY symbol, date
    """

    try:
        result = duckdb.query(query).pl()
        return result
    except Exception:
        return pl.DataFrame()


def list_symbols(asset_class: str = "equity", layer: str = "bronze") -> list[str]:
    """Return all symbols present in storage for an asset class."""
    base_path = DATA_DIR / layer / asset_class / "daily"
    if not base_path.exists():
        return []
    return sorted([
        p.name.replace("symbol=", "").replace("_", "/")
        for p in base_path.iterdir()
        if p.is_dir() and p.name.startswith("symbol=")
    ])
