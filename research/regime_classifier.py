"""Macro regime classifier.

Classifies the current economic environment into one of five regimes using
four composite indicators derived from FRED macro data:

  Growth     — Industrial production 3-month rate of change (INDPRO)
  Inflation  — CPI 12-month rate of change (CPIAUCSL)
  Labor      — Unemployment rate change (UNRATE), inverted
  Financial  — Yield curve slope: 10Y-2Y spread (T10Y2Y)

Regime definitions (classic 4-quadrant framework extended to 5 states):

  EXPANSION         Growth ↑   Inflation stable   Yield curve positive
  INFLATIONARY_BOOM Growth ↑   Inflation ↑        Yield curve positive
  STAGFLATION       Growth ↓   Inflation ↑        Any
  CONTRACTION       Any        Any                 Yield curve inverted / labor deteriorating
  RECOVERY          Growth ↑↑  Inflation ↓        Yield curve steepening

Sector rotation implied by each regime (standard macro factor rotation):

  EXPANSION         XLK  XLY  XLF  XLI   (tech, consumer disc, financials, industrials)
  INFLATIONARY_BOOM XLE  XLB  XLI  XLF   (energy, materials, industrials, financials)
  STAGFLATION       XLE  XLP  XLU  XLV   (energy, staples, utilities, healthcare)
  CONTRACTION       XLP  XLU  XLV  TLT   (defensives + long bonds)
  RECOVERY          XLY  XLK  XLF  XLV   (early-cycle: consumer disc, tech, financials)

Usage:
    macro = load_macro_data()
    regime = classify_regime(macro, as_of=date.today())
    sectors = REGIME_SECTORS[regime]
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from enum import Enum

import numpy as np
import polars as pl

from config.settings import DATA_DIR
from config.universes import INVERSE_ETFS as _INVERSE_ETFS

# Named references to inverse ETF tickers — centralised so a ticker change
# only needs to happen in config/universes.py.
_SH = _INVERSE_ETFS[0]   # ProShares Short S&P 500 (1× inverse SPY)

log = logging.getLogger(__name__)

MACRO_DIR = DATA_DIR / "alt" / "macro"


# ── Regime definitions ─────────────────────────────────────────────────────────

class MacroRegime(str, Enum):
    EXPANSION          = "expansion"
    INFLATIONARY_BOOM  = "inflationary_boom"
    STAGFLATION        = "stagflation"
    CONTRACTION        = "contraction"
    RECOVERY           = "recovery"


REGIME_LABELS: dict[MacroRegime, str] = {
    MacroRegime.EXPANSION:          "Expansion — Growth ↑, Inflation stable",
    MacroRegime.INFLATIONARY_BOOM:  "Inflationary Boom — Growth ↑, Inflation ↑",
    MacroRegime.STAGFLATION:        "Stagflation — Growth ↓, Inflation ↑ (SH overlay)",
    MacroRegime.CONTRACTION:        "Contraction — Yield curve inverted, Growth ↓ (SH hedge)",
    MacroRegime.RECOVERY:           "Recovery — Growth recovering, Inflation cooling",
}

# Sector rotation per regime.
# CONTRACTION: SH (inverse SPY) replaces XLU — direct market hedge alongside
#   defensives (XLP, XLV) and long bonds (TLT).  25% allocation each.
# STAGFLATION: SH replaces XLU — utilities underperform when rates are high;
#   SH covers the declining-growth component alongside energy and staples.
# All other regimes: long-only sector ETFs, no inverse exposure.
REGIME_SECTORS: dict[MacroRegime, list[str]] = {
    MacroRegime.EXPANSION:          ["XLK",  "XLY",  "XLF",  "XLI"],
    MacroRegime.INFLATIONARY_BOOM:  ["XLE",  "XLB",  "XLI",  "XLF"],
    MacroRegime.STAGFLATION:        ["XLE",  "XLP",  _SH,    "XLV"],
    MacroRegime.CONTRACTION:        [_SH,    "XLP",  "XLV",  "TLT"],
    MacroRegime.RECOVERY:           ["XLY",  "XLK",  "XLF",  "XLV"],
}


# ── Data loader ────────────────────────────────────────────────────────────────

def load_macro_data() -> dict[str, pl.DataFrame]:
    """Load all available macro indicator files from data/alt/macro/.

    Returns:
        Dict mapping FRED series ID → DataFrame with columns [date, value].
        Empty dict if no macro data has been fetched yet.
    """
    result: dict[str, pl.DataFrame] = {}
    if not MACRO_DIR.exists():
        return result
    for path in MACRO_DIR.glob("*.parquet"):
        sid = path.stem.upper()
        try:
            result[sid] = pl.read_parquet(path).sort("date")
        except Exception as exc:
            log.debug("regime_classifier: could not load %s: %s", path.name, exc)
    return result


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _series_as_of(df: pl.DataFrame, as_of: date, col: str = "value") -> np.ndarray:
    """Return all values on or before as_of as a float array, oldest first."""
    filtered = df.filter(pl.col("date") <= as_of)
    if filtered.is_empty():
        return np.array([])
    return filtered[col].to_numpy().astype(float)


def _rolling_zscore(series: np.ndarray, window: int = 24) -> float:
    """Z-score of the latest value relative to a rolling window."""
    if len(series) < max(window // 2, 3):
        return 0.0
    w     = min(window, len(series))
    samp  = series[-w:]
    mu    = samp[:-1].mean()
    sd    = samp[:-1].std()
    return float((samp[-1] - mu) / sd) if sd > 1e-10 else 0.0


def _pct_change(series: np.ndarray, n: int) -> np.ndarray:
    """n-period percentage change array."""
    if len(series) <= n:
        return np.array([])
    return (series[n:] / series[:-n] - 1) * 100


# ── Regime classifier ──────────────────────────────────────────────────────────

def classify_regime(
    macro: dict[str, pl.DataFrame],
    as_of: date,
    zscore_window: int = 24,
    yield_curve_threshold: float = -0.25,
    growth_threshold: float = 0.3,
    inflation_threshold: float = 0.5,
) -> MacroRegime:
    """Classify the macro regime as of a given date.

    Args:
        macro:                 Output of load_macro_data().
        as_of:                 Date to classify (uses data ≤ as_of).
        zscore_window:         Months of history for z-score normalisation.
        yield_curve_threshold: T10Y2Y spread below which triggers CONTRACTION.
        growth_threshold:      Growth z-score above which signals expansion.
        inflation_threshold:   Inflation z-score above which signals inflation.

    Returns:
        MacroRegime enum value.
    """
    # ── Yield curve (financial conditions) ────────────────────────────────────
    yc_df = macro.get("T10Y2Y")
    yc    = float("nan")
    if yc_df is not None:
        arr = _series_as_of(yc_df, as_of)
        if len(arr) > 0:
            yc = float(arr[-1])

    # ── Growth: industrial production 3-month % change z-score ────────────────
    indpro_df = macro.get("INDPRO")
    growth_z  = 0.0
    if indpro_df is not None:
        arr = _series_as_of(indpro_df, as_of)
        if len(arr) >= 6:
            chg = _pct_change(arr, 3)
            growth_z = _rolling_zscore(chg, zscore_window)

    # ── Inflation: CPI 12-month % change z-score ──────────────────────────────
    cpi_df    = macro.get("CPIAUCSL")
    infl_z    = 0.0
    if cpi_df is not None:
        arr = _series_as_of(cpi_df, as_of)
        if len(arr) >= 15:
            chg   = _pct_change(arr, 12)
            infl_z = _rolling_zscore(chg, zscore_window)

    # ── Labor: unemployment rate change z-score (inverted: ↓ unemployment = ↑ growth) ──
    unrate_df = macro.get("UNRATE")
    labor_z   = 0.0
    if unrate_df is not None:
        arr = _series_as_of(unrate_df, as_of)
        if len(arr) >= 4:
            # Z-score of the unemployment level, inverted: below-average unemployment = positive
            labor_z = -_rolling_zscore(arr, zscore_window)

    # ── Classification rules ───────────────────────────────────────────────────
    # Priority 1: inverted yield curve or severe labor deterioration → CONTRACTION
    if (not np.isnan(yc) and yc < yield_curve_threshold) or labor_z < -1.5:
        return MacroRegime.CONTRACTION

    # Priority 2: quadrant classification on growth × inflation
    expanding = growth_z > growth_threshold
    hot_infl  = infl_z > inflation_threshold

    if expanding and not hot_infl:
        return MacroRegime.EXPANSION
    if expanding and hot_infl:
        return MacroRegime.INFLATIONARY_BOOM
    if not expanding and hot_infl:
        return MacroRegime.STAGFLATION

    # Default: recovery (growth turning up, inflation cooling)
    return MacroRegime.RECOVERY


def classify_regime_history(
    macro: dict[str, pl.DataFrame],
    start: date,
    end: date,
    **kwargs,
) -> pl.DataFrame:
    """Build a full history of monthly regime classifications.

    Returns:
        Polars DataFrame with columns [date, regime, regime_label].
    """
    if not macro:
        return pl.DataFrame(schema={"date": pl.Date, "regime": pl.Utf8, "regime_label": pl.Utf8})

    # Monthly dates from start to end
    rows: list[dict] = []
    current = start.replace(day=1)
    while current <= end:
        r = classify_regime(macro, current, **kwargs)
        rows.append({
            "date":         current,
            "regime":       r.value,
            "regime_label": REGIME_LABELS[r],
        })
        # Advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
