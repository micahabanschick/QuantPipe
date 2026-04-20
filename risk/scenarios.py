"""Stress scenario P&L estimation for a portfolio of weights.

Scenarios encode peak-to-trough shocks observed during historical crises.
Each scenario maps ETF symbols to a return shock (e.g. -0.45 = -45%).
Symbols not in the shock table receive a 0% shock.

Public API:
    SCENARIOS            — dict[str, dict[str, float]]
    apply_scenario(weights, scenario_name) -> float
    run_all_scenarios(weights) -> dict[str, float]
"""

# ── Historical shock tables ────────────────────────────────────────────────────
# Shocks represent approximate peak-to-trough returns for each asset class
# during the named crisis period.  Sources: Bloomberg / public indices.
# All figures are approximate and used for illustration only.

SCENARIOS: dict[str, dict[str, float]] = {
    # --- 2008 Global Financial Crisis (Sep 2008 – Mar 2009) ---
    "2008_GFC": {
        # Equity — style/size
        "IWB": -0.48, "IWF": -0.42, "IWD": -0.53,
        "IWR": -0.50, "IWP": -0.46, "IWS": -0.54,
        "IWM": -0.47, "IWO": -0.41, "IWN": -0.53,
        # GICS sectors
        "XLK": -0.45, "XLV": -0.25, "XLF": -0.73,
        "XLY": -0.45, "XLP": -0.18, "XLE": -0.56,
        "XLI": -0.48, "XLB": -0.52, "XLU": -0.28,
        "XLRE": -0.65, "XLC": -0.40,
        # Broad / macro
        "SPY": -0.46, "QQQ": -0.42, "DIA": -0.42,
        "AGG": +0.06, "TLT": +0.25, "GLD": +0.05,
    },

    # --- 2020 COVID Crash (Feb 2020 – Mar 2020) ---
    "2020_COVID": {
        "IWB": -0.34, "IWF": -0.29, "IWD": -0.39,
        "IWR": -0.38, "IWP": -0.30, "IWS": -0.40,
        "IWM": -0.41, "IWO": -0.34, "IWN": -0.45,
        "XLK": -0.27, "XLV": -0.23, "XLF": -0.42,
        "XLY": -0.36, "XLP": -0.15, "XLE": -0.60,
        "XLI": -0.39, "XLB": -0.36, "XLU": -0.25,
        "XLRE": -0.40, "XLC": -0.28,
        "SPY": -0.34, "QQQ": -0.29, "DIA": -0.36,
        "AGG": +0.03, "TLT": +0.20, "GLD": +0.04,
    },

    # --- 2022 Rate Shock (Jan 2022 – Oct 2022) ---
    "2022_RATES": {
        "IWB": -0.25, "IWF": -0.32, "IWD": -0.17,
        "IWR": -0.24, "IWP": -0.31, "IWS": -0.18,
        "IWM": -0.26, "IWO": -0.32, "IWN": -0.19,
        "XLK": -0.35, "XLV": -0.11, "XLF": -0.16,
        "XLY": -0.37, "XLP": -0.05, "XLE": +0.34,
        "XLI": -0.23, "XLB": -0.22, "XLU": -0.08,
        "XLRE": -0.31, "XLC": -0.40,
        "SPY": -0.25, "QQQ": -0.35, "DIA": -0.18,
        "AGG": -0.17, "TLT": -0.33, "GLD": -0.03,
    },

    # --- 2000 Dot-com Bust (Mar 2000 – Oct 2002) ---
    "2000_DOTCOM": {
        "IWB": -0.47, "IWF": -0.55, "IWD": -0.37,
        "IWR": -0.44, "IWP": -0.52, "IWS": -0.36,
        "IWM": -0.40, "IWO": -0.53, "IWN": -0.28,
        "XLK": -0.77, "XLV": -0.23, "XLF": -0.25,
        "XLY": -0.48, "XLP": -0.10, "XLE": -0.08,
        "XLI": -0.35, "XLB": -0.30, "XLU": -0.40,
        "XLRE": -0.25, "XLC": -0.65,
        "SPY": -0.49, "QQQ": -0.83, "DIA": -0.35,
        "AGG": +0.24, "TLT": +0.35, "GLD": +0.15,
    },
}


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_scenario(
    weights: dict[str, float],
    scenario_name: str,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> float:
    """Estimate portfolio P&L (as fraction of NAV) under a named scenario.

    Parameters
    ----------
    weights       : {symbol: weight} — may include short positions
    scenario_name : Key in SCENARIOS (or custom scenarios dict)
    scenarios     : Override SCENARIOS table (tests / custom shocks)

    Returns
    -------
    Estimated P&L as a fraction of NAV (e.g. -0.35 = -35% loss)
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario {scenario_name!r}. Available: {list(scenarios)}")

    shocks = scenarios[scenario_name]
    pnl = sum(w * shocks.get(sym, 0.0) for sym, w in weights.items())
    return round(pnl, 6)


def run_all_scenarios(
    weights: dict[str, float],
    scenarios: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Run all scenarios and return {scenario_name: estimated_pnl}.

    Parameters
    ----------
    weights   : {symbol: weight}
    scenarios : Override SCENARIOS table

    Returns
    -------
    {scenario_name: pnl_fraction}  — sorted worst-first
    """
    if scenarios is None:
        scenarios = SCENARIOS
    results = {name: apply_scenario(weights, name, scenarios) for name in scenarios}
    return dict(sorted(results.items(), key=lambda x: x[1]))
