"""Corporate actions storage and adjusted-price helpers.

Adjustments (splits, dividends) are stored as a separate table and applied
at read time — never baked into the raw (bronze) prices. This preserves the
original vendor data and lets you re-apply or change the adjustment method
without touching stored prices.

Adjustment table schema:
    date       : Date     — ex-date of the action
    symbol     : Utf8
    action     : Utf8     — "split" | "dividend"
    value      : Float64  — split ratio (e.g. 2.0 for 2:1) or cash dividend per share
"""

from datetime import date
from pathlib import Path

import polars as pl
import yfinance as yf

from config.settings import DATA_DIR

ADJUSTMENTS_PATH = DATA_DIR / "adjustments"
ADJUSTMENTS_SCHEMA: dict[str, type] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "action": pl.Utf8,
    "value": pl.Float64,
}


def _adj_file(symbol: str) -> Path:
    ADJUSTMENTS_PATH.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_")
    return ADJUSTMENTS_PATH / f"{safe}.parquet"


def fetch_and_store_adjustments(symbol: str, start: date, end: date) -> int:
    """Pull corporate actions from yfinance and persist to the adjustments store.

    Returns the number of new rows written.
    """
    ticker = yf.Ticker(symbol)
    actions = ticker.actions  # DataFrame with Dividends and Stock Splits indexed by date

    if actions is None or actions.empty:
        return 0

    rows = []
    for ts, row in actions.iterrows():
        action_date = ts.date()
        if not (start <= action_date <= end):
            continue
        if row.get("Stock Splits", 0) != 0:
            rows.append({"date": action_date, "symbol": symbol, "action": "split",
                         "value": float(row["Stock Splits"])})
        if row.get("Dividends", 0) != 0:
            rows.append({"date": action_date, "symbol": symbol, "action": "dividend",
                         "value": float(row["Dividends"])})

    if not rows:
        return 0

    new_df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

    path = _adj_file(symbol)
    if path.exists():
        existing = pl.read_parquet(path)
        merged = (
            pl.concat([existing, new_df])
            .unique(subset=["date", "symbol", "action"], keep="last")
            .sort("date")
        )
    else:
        merged = new_df.sort("date")

    merged.write_parquet(path, compression="snappy")
    return len(rows)


def get_adjustments(symbol: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
    """Load stored adjustments for a symbol, optionally filtered by date range."""
    path = _adj_file(symbol)
    if not path.exists():
        return pl.DataFrame(schema=ADJUSTMENTS_SCHEMA)

    df = pl.read_parquet(path)
    if start:
        df = df.filter(pl.col("date") >= start)
    if end:
        df = df.filter(pl.col("date") <= end)
    return df.sort("date")


def get_adjusted_prices(
    prices: pl.DataFrame,
    symbol: str,
    use_column: str = "adj_close",
) -> pl.Series:
    """Return the adjusted close price series for a symbol.

    For ETFs and symbols where yfinance already provides adj_close, this
    simply returns that column. For custom adjustment logic, override here.

    The adj_close stored in bronze was fetched with auto_adjust=False and
    reflects Yahoo's own split/dividend-adjusted series — sufficient for
    the ETF rotation strategy. This function is the canonical read path
    so that if we upgrade to a custom adjustment method later, there is
    exactly one place to change.
    """
    if use_column not in prices.columns:
        raise ValueError(f"Column '{use_column}' not found. Available: {prices.columns}")
    return prices[use_column]


def list_symbols_with_adjustments() -> list[str]:
    """Return all symbols that have a stored adjustments file."""
    if not ADJUSTMENTS_PATH.exists():
        return []
    return sorted([
        p.stem.replace("_", "/") for p in ADJUSTMENTS_PATH.glob("*.parquet")
    ])
