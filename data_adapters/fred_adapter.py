"""FRED (Federal Reserve Economic Data) adapter.

Pulls macroeconomic and financial time series from the St. Louis Fed API.
API key required — free at https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    adapter = FREDAdapter(api_key="...")
    df = adapter.get_series("UNRATE", start="2020-01-01")
    info = adapter.get_series_info("UNRATE")
    results = adapter.search_series("unemployment")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import polars as pl
import requests

log = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 15

# Curated series — high signal value for equity/macro quant strategies
POPULAR_SERIES: dict[str, str] = {
    # Labour market
    "UNRATE":       "Unemployment Rate (Monthly %)",
    "ICSA":         "Initial Jobless Claims (Weekly, thousands)",
    "JTSJOL":       "Job Openings — JOLTS (Monthly, thousands)",
    "PAYEMS":       "Total Nonfarm Payrolls (Monthly, thousands)",
    # Inflation
    "CPIAUCSL":     "CPI All Urban Consumers (Monthly Index)",
    "PCEPI":        "PCE Price Index (Monthly Index)",
    "T5YIE":        "5-Year Breakeven Inflation Rate (Daily %)",
    # Interest rates & yield curve
    "FEDFUNDS":     "Federal Funds Effective Rate (Monthly %)",
    "DGS10":        "10-Year Treasury Yield (Daily %)",
    "DGS2":         "2-Year Treasury Yield (Daily %)",
    "T10Y2Y":       "10Y–2Y Treasury Spread — Yield Curve (Daily %)",
    # Credit & risk
    "BAMLH0A0HYM2": "High Yield Corporate Bond Spread (Daily %)",
    "TEDRATE":      "TED Spread — bank vs govt (Daily %)",
    # Growth
    "GDPC1":        "Real GDP (Quarterly, billions)",
    "INDPRO":       "Industrial Production Index (Monthly)",
    "RSXFS":        "Retail Sales excl. Food Service (Monthly $B)",
    # Dollar & commodities
    "DTWEXBGS":     "Trade-Weighted USD Index (Daily)",
    "DCOILWTICO":   "WTI Crude Oil Price (Daily $/bbl)",
    # Sentiment & volatility
    "UMCSENT":      "U Michigan Consumer Sentiment (Monthly Index)",
    "VIXCLS":       "CBOE VIX Volatility Index (Daily)",
}


class FREDAdapter:
    """Thin wrapper around the FRED REST API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FRED_API_KEY is required. Get a free key at fred.stlouisfed.org")
        self._key = api_key

    def _get(self, endpoint: str, params: dict) -> dict[str, Any]:
        params = {**params, "api_key": self._key, "file_type": "json"}
        resp = requests.get(f"{_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_series_info(self, series_id: str) -> dict[str, str]:
        """Return metadata for a FRED series (title, units, frequency, etc.)."""
        try:
            data = self._get("series", {"series_id": series_id})
            s = data["seriess"][0]
            return {
                "id":           s["id"],
                "title":        s["title"],
                "units":        s["units_short"],
                "frequency":    s["frequency_short"],
                "last_updated": s["last_updated"][:10],
                "notes":        s.get("notes", "")[:200],
            }
        except Exception as exc:
            log.warning(f"FRED series info failed for {series_id}: {exc}")
            return {"id": series_id, "title": series_id, "units": "", "frequency": "", "last_updated": ""}

    def get_series(
        self,
        series_id: str,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pl.DataFrame:
        """Pull observations for a FRED series.

        Returns a Polars DataFrame with columns [date, {series_id}].
        Missing values (FRED uses '.') are converted to null.
        """
        params: dict = {"series_id": series_id}
        if start:
            params["observation_start"] = str(start)
        if end:
            params["observation_end"] = str(end)

        data = self._get("series/observations", params)
        obs  = data.get("observations", [])
        if not obs:
            return pl.DataFrame(schema={"date": pl.Date, series_id: pl.Float64})

        rows = []
        for o in obs:
            val = o["value"]
            rows.append({
                "date":     o["date"],
                series_id: float(val) if val != "." else None,
            })

        df = pl.DataFrame(rows).with_columns(
            pl.col("date").str.to_date("%Y-%m-%d")
        )
        log.info(f"FRED: pulled {len(df)} rows for {series_id}")
        return df

    def search_series(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        """Search FRED series catalogue by keyword."""
        try:
            data = self._get("series/search", {
                "search_text":  query,
                "limit":        limit,
                "order_by":     "popularity",
                "sort_order":   "desc",
            })
            return [
                {
                    "id":        s["id"],
                    "title":     s["title"],
                    "units":     s["units_short"],
                    "frequency": s["frequency_short"],
                }
                for s in data.get("seriess", [])
            ]
        except Exception as exc:
            log.warning(f"FRED search failed: {exc}")
            return []
