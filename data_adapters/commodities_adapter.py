"""Commodities adapter — wraps YFinanceAdapter for futures and commodity ETFs.

Provides a curated symbol catalogue and a convenience method that returns a
single close-price series suitable for the Data Lab tradability pipeline.
No API key required — uses Yahoo Finance via yfinance.

Usage:
    adapter = CommoditiesAdapter()
    df = adapter.get_price_series("GC=F", start=date(2015,1,1))
    print(FUTURES_SYMBOLS)           # curated catalogue
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from .yfinance_adapter import YFinanceAdapter

log = logging.getLogger(__name__)

# ── Curated futures catalogue ─────────────────────────────────────────────────

FUTURES_SYMBOLS: dict[str, str] = {
    # Energy
    "CL=F":  "WTI Crude Oil ($/bbl)",
    "NG=F":  "Natural Gas ($/MMBtu)",
    "HO=F":  "Heating Oil ($/gal)",
    "RB=F":  "RBOB Gasoline ($/gal)",
    "BZ=F":  "Brent Crude Oil ($/bbl)",
    # Metals
    "GC=F":  "Gold ($/troy oz)",
    "SI=F":  "Silver ($/troy oz)",
    "HG=F":  "Copper ($/lb)",
    "PL=F":  "Platinum ($/troy oz)",
    "PA=F":  "Palladium ($/troy oz)",
    # Agriculture — grains
    "ZC=F":  "Corn (cents/bu)",
    "ZS=F":  "Soybeans (cents/bu)",
    "ZW=F":  "Wheat — CBOT (cents/bu)",
    "KE=F":  "Wheat — KC HRW (cents/bu)",
    "ZR=F":  "Rough Rice (cents/cwt)",
    "ZO=F":  "Oats (cents/bu)",
    # Agriculture — softs
    "KC=F":  "Coffee — Arabica (cents/lb)",
    "CC=F":  "Cocoa ($/MT)",
    "SB=F":  "Sugar #11 (cents/lb)",
    "CT=F":  "Cotton #2 (cents/lb)",
    "OJ=F":  "Orange Juice (cents/lb)",
    # Livestock
    "LE=F":  "Live Cattle (cents/lb)",
    "HE=F":  "Lean Hogs (cents/lb)",
    "GF=F":  "Feeder Cattle (cents/lb)",
    # Financial
    "ES=F":  "S&P 500 E-mini",
    "NQ=F":  "NASDAQ 100 E-mini",
    "YM=F":  "Dow Jones E-mini",
    "RTY=F": "Russell 2000 E-mini",
    "ZN=F":  "10-Year Treasury Note",
    "ZB=F":  "30-Year Treasury Bond",
    "ZF=F":  "5-Year Treasury Note",
    "ZT=F":  "2-Year Treasury Note",
    # FX
    "DX=F":  "US Dollar Index",
    "6E=F":  "Euro FX (EUR/USD)",
    "6J=F":  "Japanese Yen (JPY/USD)",
    "6B=F":  "British Pound (GBP/USD)",
    "6C=F":  "Canadian Dollar (CAD/USD)",
    "6A=F":  "Australian Dollar (AUD/USD)",
}

# Commodity ETFs — liquid, tradable proxies when futures aren't needed
COMMODITY_ETFS: dict[str, str] = {
    "GLD":  "SPDR Gold Shares",
    "SLV":  "iShares Silver Trust",
    "USO":  "United States Oil Fund",
    "UNG":  "United States Natural Gas Fund",
    "CORN": "Teucrium Corn Fund",
    "WEAT": "Teucrium Wheat Fund",
    "SOYB": "Teucrium Soybean Fund",
    "PDBC": "Invesco Optimum Yield Diversified Commodity",
    "DJP":  "iPath Bloomberg Commodity Index",
    "COMT": "iShares GSCI Commodity Dynamic Roll Strategy",
}

# Grouped for the UI
COMMODITY_GROUPS: dict[str, list[str]] = {
    "Energy":      ["CL=F", "NG=F", "HO=F", "RB=F", "BZ=F"],
    "Metals":      ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "Agriculture": ["ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "SB=F", "CT=F"],
    "Livestock":   ["LE=F", "HE=F", "GF=F"],
    "Financial":   ["ES=F", "NQ=F", "ZN=F", "ZB=F", "DX=F"],
    "FX":          ["6E=F", "6J=F", "6B=F", "6C=F"],
}


class CommoditiesAdapter:
    """Thin wrapper around YFinanceAdapter for commodity futures and ETFs."""

    def __init__(self) -> None:
        self._yf = YFinanceAdapter()

    def get_price_series(
        self,
        symbol: str,
        start: date,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Return a daily close-price series for a futures or commodity ETF.

        Args:
            symbol: Futures ticker (e.g. "GC=F") or ETF ticker (e.g. "GLD").
            start:  Start date (inclusive).
            end:    End date (inclusive); defaults to today.

        Returns:
            Polars DataFrame with columns [date, close].
        """
        from datetime import date as _date
        end = end or _date.today()
        df  = self._yf.get_bars(symbol, start, end)
        if df.is_empty():
            return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
        return df.select(["date", "close"]).sort("date")

    def get_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Return full OHLCV bars for a futures/ETF symbol."""
        from datetime import date as _date
        end = end or _date.today()
        return self._yf.get_bars(symbol, start, end)

    def catalogue(self) -> dict[str, str]:
        """Return the combined futures + ETF catalogue."""
        return {**FUTURES_SYMBOLS, **COMMODITY_ETFS}
