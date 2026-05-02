"""Alpha Vantage adapter.

Provides company fundamentals (EPS, P/E, revenue) and US macroeconomic
series that complement yfinance price data and FRED macro data.

API key required — free tier at https://www.alphavantage.co/support/#api-key
Free tier: 25 requests/day, 500 requests/month (or 75/min with premium).

Set ALPHA_VANTAGE_API_KEY in your .env file.

Usage:
    adapter = AlphaVantageAdapter(api_key="...")
    df  = adapter.get_macro_series("REAL_GDP")
    df  = adapter.get_earnings("AAPL")
    info = adapter.get_company_overview("MSFT")
    df  = adapter.get_income_statement("GOOGL")
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl
import requests

log = logging.getLogger(__name__)

_BASE    = "https://www.alphavantage.co/query"
_TIMEOUT = 20

# Macro series available via Alpha Vantage (no additional subscription needed)
MACRO_SERIES: dict[str, str] = {
    "REAL_GDP":            "US Real GDP (Quarterly, billions)",
    "REAL_GDP_PER_CAPITA": "US Real GDP Per Capita (Quarterly, $)",
    "TREASURY_YIELD":      "US Treasury Yield (Daily, 10-year %)",
    "FEDERAL_FUNDS_RATE":  "US Federal Funds Rate (Monthly %)",
    "CPI":                 "US Consumer Price Index (Monthly)",
    "INFLATION":           "US Inflation Rate (Annual %)",
    "RETAIL_SALES":        "US Retail Sales (Monthly, $M)",
    "DURABLES":            "US Durable Goods Orders (Monthly, $B)",
    "UNEMPLOYMENT":        "US Unemployment Rate (Monthly %)",
    "NONFARM_PAYROLL":     "US Nonfarm Payroll (Monthly, thousands)",
}

# Company fundamental fields returned by OVERVIEW endpoint
FUNDAMENTAL_FIELDS: list[str] = [
    "Symbol", "Name", "Sector", "Industry", "MarketCapitalization",
    "EBITDA", "PERatio", "PEGRatio", "BookValue", "DividendYield",
    "EPS", "RevenuePerShareTTM", "ProfitMargin", "OperatingMarginTTM",
    "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenueTTM",
    "GrossProfitTTM", "QuarterlyEarningsGrowthYOY",
    "QuarterlyRevenueGrowthYOY", "AnalystTargetPrice",
    "52WeekHigh", "52WeekLow", "Beta",
]


class AlphaVantageAdapter:
    """Wrapper around the Alpha Vantage REST API for fundamentals and macro data."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is required. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
        self._key = api_key

    def _get(self, params: dict) -> dict[str, Any]:
        resp = requests.get(_BASE, params={**params, "apikey": self._key},
                            timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage rate limit / key error: {data['Information']}")
        if "Note" in data:
            log.warning("Alpha Vantage note: %s", data["Note"])
        return data

    # ── Macroeconomic series ────────────────────────────────────────────────────

    def get_macro_series(
        self,
        function:  str,
        interval:  str = "quarterly",
        maturity:  str = "10year",
    ) -> pl.DataFrame:
        """Pull a US macroeconomic time series.

        Args:
            function:  One of the MACRO_SERIES keys (e.g. "REAL_GDP", "CPI").
            interval:  For REAL_GDP and TREASURY_YIELD: "quarterly"/"annual"/"daily"/"weekly"/"monthly".
            maturity:  For TREASURY_YIELD only: "3month" | "2year" | "5year" | "7year" |
                       "10year" | "20year" | "30year".

        Returns:
            Polars DataFrame with columns [date, value].
        """
        params: dict = {"function": function}
        if function in ("REAL_GDP", "REAL_GDP_PER_CAPITA"):
            params["interval"] = interval
        elif function == "TREASURY_YIELD":
            params["interval"] = interval
            params["maturity"] = maturity

        data = self._get(params)
        rows = data.get("data", [])
        if not rows:
            return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})

        records = [
            {"date": r["date"], "value": float(r["value"]) if r["value"] != "." else None}
            for r in rows
        ]
        df = (
            pl.DataFrame(records)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
            .filter(pl.col("value").is_not_null())
        )
        log.info("AlphaVantage: pulled %d rows for %s", len(df), function)
        return df

    # ── Company fundamentals ───────────────────────────────────────────────────

    def get_company_overview(self, symbol: str) -> dict[str, Any]:
        """Return key fundamental metrics for a ticker (P/E, EPS, revenue, etc.).

        Returns a flat dict with fields from FUNDAMENTAL_FIELDS.
        Numeric strings are converted to float where possible.
        """
        data = self._get({"function": "OVERVIEW", "symbol": symbol})
        if not data or "Symbol" not in data:
            log.warning("AlphaVantage: no overview data for %s", symbol)
            return {}
        result: dict[str, Any] = {}
        for field in FUNDAMENTAL_FIELDS:
            raw = data.get(field, "None")
            if raw in ("None", "-", "", None):
                result[field] = None
                continue
            try:
                result[field] = float(raw)
            except ValueError:
                result[field] = raw
        return result

    def get_earnings(self, symbol: str) -> pl.DataFrame:
        """Return quarterly earnings history (EPS actual vs estimate).

        Returns:
            Polars DataFrame with columns
            [date, reported_eps, estimated_eps, surprise, surprise_pct].
        """
        data = self._get({"function": "EARNINGS", "symbol": symbol})
        rows = data.get("quarterlyEarnings", [])
        if not rows:
            return pl.DataFrame(schema={
                "date": pl.Date, "reported_eps": pl.Float64,
                "estimated_eps": pl.Float64, "surprise": pl.Float64,
                "surprise_pct": pl.Float64,
            })

        def _f(v: str) -> float | None:
            return float(v) if v not in ("None", "-", "", None) else None

        records = [
            {
                "date":          r.get("reportedDate") or r.get("fiscalDateEnding"),
                "reported_eps":  _f(r.get("reportedEPS", "None")),
                "estimated_eps": _f(r.get("estimatedEPS", "None")),
                "surprise":      _f(r.get("surprise", "None")),
                "surprise_pct":  _f(r.get("surprisePercentage", "None")),
            }
            for r in rows
        ]
        df = (
            pl.DataFrame(records)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
        )
        log.info("AlphaVantage: pulled %d earnings rows for %s", len(df), symbol)
        return df

    def get_income_statement(self, symbol: str, annual: bool = False) -> pl.DataFrame:
        """Return quarterly (or annual) income statement history.

        Returns:
            Polars DataFrame with columns
            [date, total_revenue, gross_profit, operating_income, net_income,
             ebitda, eps, eps_diluted].
        """
        data = self._get({"function": "INCOME_STATEMENT", "symbol": symbol})
        key  = "annualReports" if annual else "quarterlyReports"
        rows = data.get(key, [])
        if not rows:
            return pl.DataFrame(schema={
                "date": pl.Date, "total_revenue": pl.Float64,
                "gross_profit": pl.Float64, "operating_income": pl.Float64,
                "net_income": pl.Float64, "ebitda": pl.Float64,
            })

        def _f(v: str) -> float | None:
            return float(v) if v not in ("None", "-", "", None) else None

        field_map = {
            "total_revenue":    "totalRevenue",
            "gross_profit":     "grossProfit",
            "operating_income": "operatingIncome",
            "net_income":       "netIncome",
            "ebitda":           "ebitda",
        }
        records = [
            {"date": r.get("fiscalDateEnding"),
             **{k: _f(r.get(v, "None")) for k, v in field_map.items()}}
            for r in rows
        ]
        df = (
            pl.DataFrame(records)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .sort("date")
        )
        log.info("AlphaVantage: pulled %d income statement rows for %s", len(df), symbol)
        return df
