"""Feature compute orchestrator — loads from bronze, writes to gold.

This is the script that turns raw OHLCV bars into analysis-ready features
stored in the gold Parquet layer. Run after every ingestion cycle.

Gold layer layout:
    data/gold/{asset_class}/features/symbol={SYMBOL}/year={YYYY}/data.parquet

Usage:
    uv run python features/compute.py
    uv run python features/compute.py --asset-class crypto --start 2022-01-01
"""

import argparse
import logging
import sys
from datetime import date, timedelta

import polars as pl

from config.settings import DATA_DIR, LOGS_DIR
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date
from features.canonical import compute_features

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "features.log"),
    ],
)
log = logging.getLogger(__name__)

# Minimum history required before any feature is meaningful
# momentum_12m_1m needs 252 days + 21 days buffer
MIN_HISTORY_DAYS = 280


def _gold_path(asset_class: str, symbol: str, year: int):
    safe = symbol.replace("/", "_")
    p = DATA_DIR / "gold" / asset_class / "features" / f"symbol={safe}" / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    return p / "data.parquet"


def compute_and_store(
    asset_class: str,
    start: date,
    end: date,
    feature_list: list[str] | None = None,
) -> int:
    """Compute features for the full universe and write to gold layer.

    Returns total rows written.
    """
    symbols = universe_as_of_date(asset_class, end, require_data=True)
    if not symbols:
        log.warning(f"No symbols in universe for {asset_class} as of {end}")
        return 0

    # Load with enough lookback for momentum_12m_1m (252 + 21 days)
    load_start = start - timedelta(days=MIN_HISTORY_DAYS)
    log.info(f"Computing features for {len(symbols)} {asset_class} symbols ({start} to {end})")

    bars = load_bars(symbols, load_start, end, asset_class)
    if bars.is_empty():
        log.warning("No bars loaded — skipping feature compute")
        return 0

    features_df = compute_features(bars, feature_list)
    if features_df.is_empty():
        log.warning("compute_features returned empty DataFrame")
        return 0

    # Trim to the requested date window (extra history was only for warm-up)
    features_df = features_df.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )

    total_written = 0
    for symbol in features_df["symbol"].unique().to_list():
        sym_df = features_df.filter(pl.col("symbol") == symbol)
        years = sym_df["date"].dt.year().unique().to_list()
        for year in years:
            year_df = sym_df.filter(pl.col("date").dt.year() == year)
            path = _gold_path(asset_class, symbol, year)
            if path.exists():
                existing = pl.read_parquet(path)
                feature_cols = [c for c in year_df.columns if c not in ("date", "symbol")]
                merged = (
                    pl.concat([existing, year_df])
                    .unique(subset=["date", "symbol"], keep="last")
                    .sort("date")
                )
            else:
                merged = year_df.sort("date")
            merged.write_parquet(path, compression="snappy")
            total_written += len(year_df)

    log.info(f"Feature compute complete: {total_written} rows written to gold layer")
    return total_written


def load_features(
    symbols: list[str] | str,
    start: date,
    end: date,
    asset_class: str = "equity",
    feature_list: list[str] | None = None,
) -> pl.DataFrame:
    """Load pre-computed features from the gold layer via DuckDB.

    Falls back to computing on-the-fly from bronze if gold is empty
    (useful during development before the full compute pipeline is running).
    """
    import duckdb

    if isinstance(symbols, str):
        symbols = [symbols]

    base_path = DATA_DIR / "gold" / asset_class / "features"
    glob_pattern = str(base_path / "**/*.parquet")

    if not base_path.exists() or not list(base_path.rglob("*.parquet")):
        log.warning("Gold layer empty — computing features on-the-fly from bronze")
        bars = load_bars(symbols, start - timedelta(days=MIN_HISTORY_DAYS), end, asset_class)
        if bars.is_empty():
            return pl.DataFrame()
        df = compute_features(bars, feature_list)
        return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    safe_symbols = [s.replace("/", "_") for s in symbols]
    symbol_list = ", ".join(f"'{s}'" for s in safe_symbols)
    col_clause = "*"
    if feature_list:
        cols = ["date", "symbol"] + feature_list
        col_clause = ", ".join(cols)

    query = f"""
        SELECT {col_clause}
        FROM read_parquet('{glob_pattern}', hive_partitioning=true)
        WHERE symbol IN ({symbol_list})
          AND date >= '{start}'
          AND date <= '{end}'
        ORDER BY symbol, date
    """
    try:
        return duckdb.query(query).pl()
    except Exception as exc:
        log.error(f"Gold layer read failed: {exc}")
        return pl.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute features and write to gold layer")
    parser.add_argument("--asset-class", choices=["equity", "crypto", "all"], default="all")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--features", nargs="+", default=None,
                        help="Subset of features to compute (default: all 5)")
    args = parser.parse_args()

    # Default start: 7 years back so gold layer covers full backtest window
    start = args.start or date(args.end.year - 7, args.end.month, args.end.day)

    asset_classes = ["equity", "crypto"] if args.asset_class == "all" else [args.asset_class]
    for ac in asset_classes:
        compute_and_store(ac, start, args.end, args.features)


if __name__ == "__main__":
    main()
