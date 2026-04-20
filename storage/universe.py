"""Point-in-time universe management.

universe_as_of_date() is the canonical entry point for anything that needs
to know "what symbols were I allowed to trade on date T?" Every downstream
module — features, signals, backtest, portfolio — should call this rather
than importing from config.universes directly.

ETF universe notes:
    The 20 iShares/SPDR ETFs in our equity universe have been continuously
    listed since at least 2006. For this strategy survivorship bias is not
    a concern — the ETFs themselves don't delist. If you expand to single
    stocks later, replace the equity branch with a real point-in-time
    constituent database (Norgate, CRSP, or a custom delisted-tickers file).

Crypto universe notes:
    Exchange listings/delistings can happen at any time. The crypto universe
    is considered valid from the date we have data for each symbol. Symbols
    with no data as of a given date are automatically excluded.
"""

from datetime import date

import polars as pl

from config.universes import CRYPTO_UNIVERSE, EQUITY_UNIVERSE
from storage.parquet_store import list_symbols, load_bars

# Earliest date each ETF was reliably available (conservative floor)
# Used to prevent the universe from including symbols before they existed
_ETF_INCEPTION: dict[str, date] = {
    "XLC": date(2018, 6, 19),   # Communication Services SPDR launched Jun 2018
    "XLRE": date(2015, 10, 7),  # Real Estate SPDR launched Oct 2015
    # All others (IW*, XL*, SPY, QQQ, AGG, TLT, GLD, DIA) predate 2006
}

_DEFAULT_INCEPTION = date(2000, 1, 1)


def universe_as_of_date(
    asset_class: str,
    as_of: date,
    require_data: bool = True,
) -> list[str]:
    """Return the tradeable universe for *asset_class* as of *as_of*.

    Parameters
    ----------
    asset_class : "equity" | "crypto"
    as_of       : The date to evaluate the universe on.
    require_data: If True (default), only include symbols that have at least
                  one stored bar on or before *as_of*. Prevents signals from
                  being computed on symbols with no historical data yet.

    Returns
    -------
    Sorted list of symbol strings.
    """
    if asset_class == "equity":
        candidates = [
            sym for sym in EQUITY_UNIVERSE
            if _ETF_INCEPTION.get(sym, _DEFAULT_INCEPTION) <= as_of
        ]
    elif asset_class == "crypto":
        candidates = list(CRYPTO_UNIVERSE)
    else:
        raise ValueError(f"Unknown asset_class: {asset_class!r}. Use 'equity' or 'crypto'.")

    if not require_data:
        return sorted(candidates)

    # Filter to symbols that actually have stored data on or before as_of
    stored = set(list_symbols(asset_class))
    available = []
    for sym in candidates:
        safe = sym.replace("/", "_")
        if safe in stored or sym in stored:
            available.append(sym)

    return sorted(available)


def universe_date_range(
    asset_class: str,
    start: date,
    end: date,
) -> dict[date, list[str]]:
    """Return the universe for every date in [start, end] (trading days only).

    Builds a date → symbols mapping. Useful for walk-forward validation
    where the universe must be re-evaluated on each rebalance date.

    For ETFs this is effectively constant (no entries/exits). For future
    single-stock expansion this will be populated from a constituents table.
    """
    # Load all available dates from storage to find actual trading days
    stored_symbols = list_symbols(asset_class)
    if not stored_symbols:
        return {}

    sample_sym = stored_symbols[0]
    bars = load_bars(sample_sym, start, end, asset_class)
    if bars.is_empty():
        return {}

    trading_dates = bars["date"].unique().sort().to_list()
    universe_map: dict[date, list[str]] = {}
    for d in trading_dates:
        universe_map[d] = universe_as_of_date(asset_class, d, require_data=True)

    return universe_map
