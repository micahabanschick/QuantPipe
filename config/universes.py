"""Universe definitions for the QuantPipe pipeline.

Equity universe is built around a 9-box Sector × Size × Style framework.
Crypto universe uses CCXT symbol format (BASE/QUOTE).
"""

# Morningstar 9-box: Large/Mid/Small × Growth/Blend/Value (iShares Russell)
STYLE_SIZE_9BOX: list[str] = [
    "IWB", "IWF", "IWD",   # Large: Blend, Growth, Value
    "IWR", "IWP", "IWS",   # Mid:   Blend, Growth, Value
    "IWM", "IWO", "IWN",   # Small: Blend, Growth, Value
]

# GICS sector SPDRs (11 sectors)
SECTOR_SPDRS: list[str] = [
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

# Broad benchmarks and risk-on/risk-off anchors
BENCHMARKS: list[str] = [
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "AGG",  # US Aggregate Bond
    "TLT",  # 20+ Year Treasuries
    "GLD",  # Gold
    "DIA",  # Dow Jones
]

# Full equity universe (deduplicated, sorted for reproducibility)
EQUITY_UNIVERSE: list[str] = sorted(set(STYLE_SIZE_9BOX + SECTOR_SPDRS + BENCHMARKS))

# Crypto rotation universe — liquid, CEX-tradeable
CRYPTO_UNIVERSE: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "DOT/USDT",
    "ATOM/USDT",    # Cosmos (replaces MATIC/POL — not listed on Kraken as USDT)
    "BNB/USDT",
    "XRP/USDT",
]

# Sector ETF → representative proxy stocks for earnings surprise signal.
# 3 liquid, high-analyst-coverage stocks per sector.
# Used by: orchestration/pull_earnings.py and strategies/earnings_surprise_drift/
SECTOR_PROXY_MAP: dict[str, list[str]] = {
    "XLK":  ["AAPL", "MSFT", "NVDA"],   # Technology
    "XLE":  ["XOM",  "CVX",  "COP"],    # Energy
    "XLF":  ["JPM",  "BAC",  "WFC"],    # Financials
    "XLU":  ["NEE",  "DUK",  "SO"],     # Utilities
    "XLI":  ["CAT",  "DE",   "HON"],    # Industrials
    "XLV":  ["JNJ",  "UNH",  "PFE"],    # Health Care
    "XLP":  ["PG",   "KO",   "PEP"],    # Consumer Staples
    "XLC":  ["META", "GOOGL","VZ"],      # Communication Services
    "XLY":  ["AMZN", "TSLA", "HD"],     # Consumer Discretionary
    "XLRE": ["AMT",  "PLD",  "EQIX"],   # Real Estate
}

# Asset class tag → universe mapping (used by orchestration)
UNIVERSES: dict[str, list[str]] = {
    "equity": EQUITY_UNIVERSE,
    "crypto": CRYPTO_UNIVERSE,
}
