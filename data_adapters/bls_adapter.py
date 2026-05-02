"""BLS (Bureau of Labor Statistics) adapter.

Pulls US economic time series from the BLS Public Data API v2.
Registration key optional but recommended — raises daily limit from 25 to 500 req/day.

Free key at: https://data.bls.gov/registrationEngine/

Set BLS_API_KEY in your .env file (leave blank for unregistered access).

Usage:
    adapter = BLSAdapter(api_key="...")  # or BLSAdapter() for unregistered
    df   = adapter.get_series("LNS14000000", start_year=2000)
    info = adapter.get_series_info("LNS14000000")
"""

from __future__ import annotations

import json
import logging
from typing import Any

import polars as pl
import requests

log = logging.getLogger(__name__)

_BASE    = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_TIMEOUT = 20

# Curated series — high signal for macro-aware equity strategies
POPULAR_SERIES: dict[str, str] = {
    # Labour market
    "CES0000000001":  "Total Nonfarm Payrolls (Monthly, thousands)",
    "LNS14000000":    "Unemployment Rate (Monthly, %)",
    "LNS11300000":    "Labour Force Participation Rate (Monthly, %)",
    "CES0500000003":  "Avg Hourly Earnings — Private (Monthly, $)",
    "CES0500000007":  "Avg Weekly Hours — Private (Monthly, hours)",
    "JTS00000000JOR": "Job Openings Rate — Total (Monthly, %)",
    "JTS00000000QUR": "Quit Rate — Total (Monthly, %)",
    # Inflation
    "CUUR0000SA0":    "CPI-U All Items (Monthly, Index 1982-84=100)",
    "CUUR0000SA0L1E": "CPI-U All Items Less Food & Energy (Core, Monthly)",
    "CUUR0000SAF1":   "CPI-U Food (Monthly)",
    "CUUR0000SAE":    "CPI-U Energy (Monthly)",
    "WPUFD4":         "PPI Final Demand (Monthly, Index 2009=100)",
    # Productivity
    "PRS85006092":    "Nonfarm Business Labour Productivity (Quarterly, %chg)",
    "PRS85006112":    "Nonfarm Business Unit Labour Costs (Quarterly, %chg)",
    # Employment by sector
    "CES1000000001":  "Mining & Logging Employment (Monthly, thousands)",
    "CES2000000001":  "Construction Employment (Monthly, thousands)",
    "CES3000000001":  "Manufacturing Employment (Monthly, thousands)",
    "CES4000000001":  "Trade/Transport/Utilities Employment (Monthly, thousands)",
    "CES6000000001":  "Professional & Business Services Employment (Monthly, thousands)",
    "CES7000000001":  "Leisure & Hospitality Employment (Monthly, thousands)",
}


class BLSAdapter:
    """Thin wrapper around the BLS Public Data API v2."""

    def __init__(self, api_key: str = "") -> None:
        self._key = api_key.strip()
        if not self._key:
            log.info(
                "BLS: running without a registration key (25 req/day). "
                "Register free at https://data.bls.gov/registrationEngine/ "
                "to raise the limit to 500 req/day."
            )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._key:
            payload["registrationkey"] = self._key
        resp = requests.post(_BASE, data=json.dumps(payload),
                             headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "REQUEST_FAILED":
            msgs = data.get("message", ["Unknown BLS API error"])
            raise RuntimeError(f"BLS API error: {'; '.join(msgs)}")
        return data

    def get_series(
        self,
        series_id: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Pull observations for a BLS series.

        Args:
            series_id:  BLS series ID (e.g. "LNS14000000").
            start_year: First year to include (e.g. 2000).
            end_year:   Last year to include (e.g. 2024).

        Returns:
            Polars DataFrame with columns [date, series_id].
        """
        from datetime import date
        payload: dict[str, Any] = {"seriesid": [series_id], "catalog": False}
        if start_year:
            payload["startyear"] = str(start_year)
        if end_year:
            payload["endyear"] = str(end_year)
        if self._key:
            payload["calculations"] = False

        data   = self._post(payload)
        series = data.get("Results", {}).get("series", [])
        if not series:
            return pl.DataFrame(schema={"date": pl.Date, series_id: pl.Float64})

        _MONTH_MAP = {
            "M01": "01", "M02": "02", "M03": "03", "M04": "04",
            "M05": "05", "M06": "06", "M07": "07", "M08": "08",
            "M09": "09", "M10": "10", "M11": "11", "M12": "12",
            "Q01": "01", "Q02": "04", "Q03": "07", "Q04": "10",
            "A01": "01",  # annual → assign to Jan
        }
        rows = []
        for obs in series[0].get("data", []):
            period = obs.get("period", "M01")
            month  = _MONTH_MAP.get(period, "01")
            year   = obs.get("year", "2000")
            val    = obs.get("value", "")
            rows.append({
                "date":     f"{year}-{month}-01",
                series_id:  float(val) if val not in ("", "-") else None,
            })

        df = (
            pl.DataFrame(rows)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
            .filter(pl.col(series_id).is_not_null())
        )
        log.info("BLS: pulled %d rows for %s", len(df), series_id)
        return df

    def get_multiple_series(
        self,
        series_ids: list[str],
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Pull multiple series in a single API call (max 25 with a key, 1 without).

        Returns:
            Dict mapping series_id → DataFrame with columns [date, series_id].
        """
        if not self._key and len(series_ids) > 1:
            log.warning(
                "BLS: unregistered access only supports 1 series per call. "
                "Fetching sequentially — register for a free key to batch up to 25."
            )
            return {sid: self.get_series(sid, start_year, end_year) for sid in series_ids}

        payload: dict[str, Any] = {"seriesid": series_ids[:25], "catalog": False}
        if start_year:
            payload["startyear"] = str(start_year)
        if end_year:
            payload["endyear"] = str(end_year)

        data    = self._post(payload)
        results = {}
        _MONTH_MAP = {
            "M01": "01", "M02": "02", "M03": "03", "M04": "04",
            "M05": "05", "M06": "06", "M07": "07", "M08": "08",
            "M09": "09", "M10": "10", "M11": "11", "M12": "12",
            "Q01": "01", "Q02": "04", "Q03": "07", "Q04": "10",
            "A01": "01",
        }
        for s in data.get("Results", {}).get("series", []):
            sid  = s["seriesID"]
            rows = []
            for obs in s.get("data", []):
                period = obs.get("period", "M01")
                month  = _MONTH_MAP.get(period, "01")
                year   = obs.get("year", "2000")
                val    = obs.get("value", "")
                rows.append({
                    "date": f"{year}-{month}-01",
                    sid:    float(val) if val not in ("", "-") else None,
                })
            if rows:
                results[sid] = (
                    pl.DataFrame(rows)
                    .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
                    .sort("date")
                    .filter(pl.col(sid).is_not_null())
                )
            else:
                results[sid] = pl.DataFrame(schema={"date": pl.Date, sid: pl.Float64})

        return results

    def get_series_info(self, series_id: str) -> dict[str, str]:
        """Return basic metadata for a BLS series (best-effort — BLS v2 is limited)."""
        return {
            "id":          series_id,
            "description": POPULAR_SERIES.get(series_id, series_id),
            "source":      "U.S. Bureau of Labor Statistics",
            "url":         f"https://data.bls.gov/timeseries/{series_id}",
        }
