"""Macro Regime Rotation — sector ETF rotation driven by the economic regime.

Mechanism:
  Classifies the current macro environment into one of five regimes using
  industrial production (growth), CPI (inflation), unemployment (labour),
  and the yield curve (financial conditions). Rotates into the sector ETFs
  that have historically outperformed in each regime.

Regime → Sector mapping (standard macro factor rotation framework):

  Expansion          Growth ↑  Inflation stable  → XLK  XLY  XLF  XLI
  Inflationary Boom  Growth ↑  Inflation ↑       → XLE  XLB  XLI  XLF
  Stagflation        Growth ↓  Inflation ↑       → XLE  XLP  XLU  XLV
  Contraction        Yield curve inverted         → XLP  XLU  XLV  TLT
  Recovery           Growth recovering            → XLY  XLK  XLF  XLV

Data dependency:
  Requires data/alt/macro/{series}.parquet files populated by
  orchestration/pull_macro.py (runs as Step 4 in run_pipeline.py).
  Pass pre-loaded data via macro_data kwarg to keep get_signal pure.

Signal: constant within each regime period — all selected sectors receive
equal weight; the portfolio rotates as the regime transitions.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from research.regime_classifier import (
    MacroRegime, REGIME_SECTORS, REGIME_LABELS,
    classify_regime, load_macro_data,
)

log = logging.getLogger(__name__)

NAME        = "Macro Regime Rotation"
DESCRIPTION = (
    "Sector ETF rotation based on the macro economic regime. "
    "Classifies the economy as expansion / inflationary boom / stagflation / "
    "contraction / recovery using growth, inflation, labour, and yield curve "
    "indicators, then holds the 4 sectors that historically outperform."
)
DEFAULT_PARAMS = {
    "lookback_years":         6,
    "top_n":                  4,
    "cost_bps":               5.0,
    "weight_scheme":          "equal",
    "zscore_window":          24,     # months of history for z-score normalisation
    "yield_curve_threshold":  -0.25,  # T10Y2Y below this → CONTRACTION
    "growth_threshold":       0.3,    # growth z-score above this → expanding
    "inflation_threshold":    0.5,    # inflation z-score above this → hot
}


def get_signal(
    _features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    zscore_window: int = DEFAULT_PARAMS["zscore_window"],
    yield_curve_threshold: float = DEFAULT_PARAMS["yield_curve_threshold"],
    growth_threshold: float = DEFAULT_PARAMS["growth_threshold"],
    inflation_threshold: float = DEFAULT_PARAMS["inflation_threshold"],
    macro_data: dict | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Assign sector ETFs to each rebalance date based on the current regime.

    Args:
        _features:   Standard features DataFrame (unused — this strategy uses
                     macro indicators injected via macro_data).
        rebal_dates: List of rebalance dates.
        macro_data:  Pre-loaded macro data from load_macro_data().
                     If None, loads from disk (convenience for standalone use).

    Returns:
        DataFrame with columns [date, symbol, regime, regime_label].
    """
    _EMPTY = pl.DataFrame(schema={
        "date": pl.Date, "symbol": pl.Utf8,
        "regime": pl.Utf8, "regime_label": pl.Utf8,
    })

    if macro_data is None:
        log.warning(
            "macro_regime_rotation: macro_data not provided — pass the result of "
            "load_macro_data() from research.regime_classifier. Returning empty signal."
        )
        return _EMPTY
    macro = macro_data

    classify_kwargs = dict(
        zscore_window=zscore_window,
        yield_curve_threshold=yield_curve_threshold,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
    )

    rows = []
    for rebal_date in rebal_dates:
        as_of = rebal_date if isinstance(rebal_date, date) else rebal_date.date()
        regime = classify_regime(macro, as_of, **classify_kwargs)
        sectors = REGIME_SECTORS[regime][:top_n]

        rows.extend(
            {"date": as_of, "symbol": s, "regime": regime.value, "regime_label": REGIME_LABELS[regime]}
            for s in sectors
        )

    if not rows:
        return _EMPTY

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Equal-weight the sectors selected for each rebalance date.

    Only 'equal' weighting is currently implemented; other schemes raise ValueError.
    """
    if weight_scheme != "equal":
        raise ValueError(f"Unsupported weight_scheme={weight_scheme!r}; only 'equal' is implemented.")
    _EMPTY = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    if signal.is_empty():
        return _EMPTY

    rows = []
    for rebal_date, group in signal.group_by("date"):
        n = len(group)
        if n == 0:
            continue
        w = 1.0 / n
        rows.extend({"date": r["date"], "symbol": r["symbol"], "weight": w}
                     for r in group.iter_rows(named=True))

    if not rows:
        return _EMPTY

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["date", "symbol"])
    )
