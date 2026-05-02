"""World Bank Open Data adapter.

Pulls macroeconomic indicators by country from the World Bank REST API.
No API key required — completely free and open.

Usage:
    adapter = WorldBankAdapter()
    df = adapter.get_indicator("US", "NY.GDP.MKTP.KD.ZG", start_year=2000)
    info = adapter.get_indicator_info("NY.GDP.MKTP.KD.ZG")
    results = adapter.search_indicators("inflation")
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl
import requests

log = logging.getLogger(__name__)

_BASE    = "https://api.worldbank.org/v2"
_TIMEOUT = 20

# Curated indicators — high signal for macro-aware equity strategies
POPULAR_INDICATORS: dict[str, str] = {
    # Growth
    "NY.GDP.MKTP.KD.ZG":  "GDP Growth Rate (Annual %)",
    "NY.GDP.PCAP.KD.ZG":  "GDP Per Capita Growth (Annual %)",
    "NE.EXP.GNFS.ZS":     "Exports of Goods & Services (% of GDP)",
    "NE.IMP.GNFS.ZS":     "Imports of Goods & Services (% of GDP)",
    "NE.TRD.GNFS.ZS":     "Trade Openness (% of GDP)",
    # Inflation & monetary
    "FP.CPI.TOTL.ZG":     "Inflation, Consumer Prices (Annual %)",
    "FR.INR.LEND":         "Lending Interest Rate (%)",
    "FR.INR.RINR":         "Real Interest Rate (%)",
    # Labour
    "SL.UEM.TOTL.ZS":     "Unemployment Rate (% of Labour Force)",
    "SL.UEM.TOTL.NE.ZS":  "Unemployment Rate, National Estimate (%)",
    "SL.EMP.TOTL.SP.ZS":  "Employment to Population Ratio (%)",
    # Investment & savings
    "NE.GDI.TOTL.ZS":     "Gross Capital Formation (% of GDP)",
    "NY.GNS.ICTR.ZS":     "Gross Savings (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign Direct Investment, Net Inflows (% of GDP)",
    # Debt & fiscal
    "GC.DOD.TOTL.GD.ZS":  "Central Government Debt (% of GDP)",
    "GC.REV.XGRT.GD.ZS":  "Revenue, Excluding Grants (% of GDP)",
    # Financial development
    "FS.AST.DOMS.GD.ZS":  "Domestic Credit to Private Sector (% of GDP)",
    "CM.MKT.LCAP.GD.ZS":  "Market Capitalisation of Listed Companies (% of GDP)",
}

# ISO2 codes for commonly analysed economies
MAJOR_ECONOMIES: dict[str, str] = {
    "US": "United States",
    "CN": "China",
    "DE": "Germany",
    "JP": "Japan",
    "GB": "United Kingdom",
    "FR": "France",
    "IN": "India",
    "BR": "Brazil",
    "CA": "Canada",
    "AU": "Australia",
    "KR": "South Korea",
    "MX": "Mexico",
}


class WorldBankAdapter:
    """Thin wrapper around the World Bank Open Data REST API (v2)."""

    def _get(self, path: str, params: dict | None = None) -> list[Any]:
        """Fetch a paginated World Bank endpoint; return all items across pages."""
        base_params = {"format": "json", "per_page": 1000, **(params or {})}
        resp = requests.get(f"{_BASE}/{path}", params=base_params, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        # World Bank returns [metadata, data] — metadata contains pagination info
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        meta, data = payload[0], payload[1] or []

        # Paginate if there are more pages
        total_pages = int(meta.get("pages", 1))
        for page in range(2, total_pages + 1):
            resp = requests.get(f"{_BASE}/{path}",
                                params={**base_params, "page": page},
                                timeout=_TIMEOUT)
            resp.raise_for_status()
            _, more = resp.json()
            data.extend(more or [])

        return data

    def get_indicator(
        self,
        country:    str,
        indicator:  str,
        start_year: int | None = None,
        end_year:   int | None = None,
    ) -> pl.DataFrame:
        """Pull annual observations for a World Bank indicator.

        Args:
            country:    ISO2 country code (e.g. "US", "CN", "DE").
            indicator:  World Bank indicator code (e.g. "NY.GDP.MKTP.KD.ZG").
            start_year: First year to include (inclusive).
            end_year:   Last year to include (inclusive).

        Returns:
            Polars DataFrame with columns [date, country, indicator, value].
        """
        path   = f"country/{country}/indicator/{indicator}"
        params: dict = {}
        if start_year or end_year:
            s = start_year or 1960
            e = end_year   or 2100
            params["date"] = f"{s}:{e}"

        rows = self._get(path, params)
        if not rows:
            return pl.DataFrame(schema={
                "date": pl.Date, "country": pl.Utf8,
                "indicator": pl.Utf8, "value": pl.Float64,
            })

        records = []
        for r in rows:
            v = r.get("value")
            records.append({
                "date":      f"{r['date']}-01-01",
                "country":   r["countryiso3code"] or country.upper(),
                "indicator": indicator,
                "value":     float(v) if v is not None else None,
            })

        df = (
            pl.DataFrame(records)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
            .filter(pl.col("value").is_not_null())
        )
        log.info("WorldBank: pulled %d rows for %s / %s", len(df), country, indicator)
        return df

    def get_indicator_multi(
        self,
        countries:  list[str],
        indicator:  str,
        start_year: int | None = None,
        end_year:   int | None = None,
    ) -> pl.DataFrame:
        """Pull the same indicator for multiple countries in one call.

        Returns a wide DataFrame with columns [date, <ISO2_code>, ...].
        """
        country_str = ";".join(countries)
        df_long = self.get_indicator(country_str, indicator, start_year, end_year)
        if df_long.is_empty():
            return df_long
        return (
            df_long
            .pivot(values="value", index="date", on="country")
            .sort("date")
        )

    def get_indicator_info(self, indicator: str) -> dict[str, str]:
        """Return metadata for a World Bank indicator."""
        try:
            rows = self._get(f"indicator/{indicator}")
            if not rows:
                return {"id": indicator, "name": indicator}
            r = rows[0]
            return {
                "id":         r.get("id", indicator),
                "name":       r.get("name", indicator),
                "source":     r.get("source", {}).get("value", ""),
                "unit":       r.get("unit", ""),
                "topic":      ", ".join(t["value"] for t in r.get("topics", [])),
                "notes":      (r.get("sourceNote") or "")[:300],
            }
        except Exception as exc:
            log.warning("WorldBank indicator info failed for %s: %s", indicator, exc)
            return {"id": indicator, "name": indicator}

    def search_indicators(self, query: str, limit: int = 20) -> list[dict[str, str]]:
        """Search the World Bank indicator catalogue by keyword."""
        try:
            rows = self._get(f"indicator?source=2", {"mrv": 1})
            results = []
            q = query.lower()
            for r in rows:
                if q in r.get("name", "").lower() or q in r.get("id", "").lower():
                    results.append({
                        "id":   r["id"],
                        "name": r["name"],
                        "unit": r.get("unit", ""),
                    })
                    if len(results) >= limit:
                        break
            return results
        except Exception as exc:
            log.warning("WorldBank search failed: %s", exc)
            return []
