"""Adaptive Dual Momentum — module-based relative + absolute momentum.

QuantPipe adaptation of Strategy W — Adaptive Dual Momentum
(Antonacci 2014, Jegadeesh & Titman 1993, Moreira & Muir 2017).

Algorithm:
  Four independent modules, each with a base portfolio weight:
    (1) SPY / EFA   — developed equity geography (30%)
    (2) QQQ / IWM   — growth vs small-cap factor   (25%)
    (3) IEF / TLT   — bond duration positioning    (25%)
    (4) GLD / LQD   — real assets / credit         (20%)

  For each module:
    1. Relative momentum: pick the asset with higher 12-1 month return.
    2. Absolute momentum gate: winner must have positive momentum
       (proxy for beating cash; original compares vs BIL T-bill return).
    3. If gate fails: module weight goes to implicit cash (weight = 0).

  Volatility-targeting overlay: scales total equity weights so portfolio
  vol ≈ 10% annualised (Moreira & Muir 2017). Max leverage = 1.0 (long-only).

Target metrics: Sortino > 1.5, Calmar > 0.8, Max DD < 20%
Rebalance: Monthly
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Adaptive Dual Momentum"
DESCRIPTION = (
    "Module-based dual momentum: relative winner in 4 asset-class pairs, "
    "gated by absolute momentum (positive return). Vol-targeting overlay."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          4,   # max simultaneous module positions (up to 4 modules)
    "cost_bps":       5.0,
    "weight_scheme":  "equal",   # "equal" or "vol_scaled"
}

# (candidates, base_weight)
_MODULES = [
    (["SPY", "EFA"], 0.30),
    (["QQQ", "IWM"], 0.25),
    (["IEF", "TLT"], 0.25),
    (["GLD", "LQD"], 0.20),
]

_VOL_TARGET = 0.10   # 10% annualised portfolio vol target
_VOL_N      = 63     # 3-month vol window
_RHO_EST    = 0.30   # low avg correlation across 4 diversified modules
_MIN_VOL    = 0.05


def _portfolio_vol(vols: list[float], weights: list[float], rho: float) -> float:
    sum_w2s2 = sum(w * w * v * v for w, v in zip(weights, vols))
    sum_ws   = sum(w * v         for w, v in zip(weights, vols))
    variance = (1.0 - rho) * sum_w2s2 + rho * sum_ws * sum_ws
    return float(np.sqrt(max(variance, 0.0)))


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Select module winners by dual momentum; compute vol-scaled equity_pct."""
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    has_vol   = "realized_vol_21d" in features.columns
    feat_syms = set(features["symbol"].unique().to_list())
    rows: list[dict] = []

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)

        snap = features.filter(pl.col(date_col) == rd)
        if snap.is_empty():
            avail   = sorted(features[date_col].unique().to_list())
            nearest = min(avail, key=lambda d: abs((pd.Timestamp(d) - rd_ts).days))
            snap    = features.filter(pl.col(date_col) == nearest)
        if snap.is_empty():
            continue

        snap_pd  = snap.to_pandas().set_index("symbol")
        mom_ser  = snap_pd["momentum_12m_1m"].dropna()
        vol_ser  = snap_pd["realized_vol_21d"].dropna() if has_vol else pd.Series(dtype=float)

        # Module evaluation
        module_alloc: list[tuple[str, float]] = []  # (symbol, base_weight)

        for candidates, base_w in _MODULES:
            avail_cands = [(c, float(mom_ser[c])) for c in candidates if c in mom_ser.index]
            if not avail_cands:
                continue  # module goes to cash

            # Relative momentum: pick winner
            winner, winner_mom = max(avail_cands, key=lambda x: x[1])

            # Absolute momentum gate
            if winner_mom > 0:
                module_alloc.append((winner, base_w))
            # else: module allocation becomes implicit cash

        if not module_alloc:
            continue

        # Deduplicate (same asset may win multiple modules)
        combined: dict[str, float] = {}
        for sym, w in module_alloc:
            combined[sym] = combined.get(sym, 0.0) + w

        # Volatility-targeting overlay
        total_w = sum(combined.values())
        if total_w > 0 and len(combined) >= 1:
            syms_list    = list(combined.keys())
            weights_list = [combined[s] / total_w for s in syms_list]
            vols_list    = [
                max(float(vol_ser[s]) if s in vol_ser.index else 0.15, _MIN_VOL)
                for s in syms_list
            ]
            pv = _portfolio_vol(vols_list, weights_list, _RHO_EST)
            if pv > 0.001:
                scale = min(_VOL_TARGET / pv, 1.0)
                equity_pct = min(total_w * scale, 1.0)
            else:
                equity_pct = min(total_w, 1.0)
        else:
            equity_pct = min(total_w, 1.0)

        # Normalize within selected set
        for sym, raw_w in combined.items():
            norm_w = (raw_w / total_w) if total_w > 0 else 0.0
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(float(mom_ser.get(sym, 0.0)), 6),
                "rank":           1,
                "selected":       True,
                "base_weight":    round(norm_w, 6),
                "equity_pct":     round(equity_pct, 4),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    if signal.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in signal.columns else "date"
    rows: list[dict] = []

    for rd in signal[date_col].unique().sort().to_list():
        day      = signal.filter(pl.col(date_col) == rd)
        selected = day.filter(pl.col("selected"))
        if selected.is_empty():
            continue

        equity_pct = float(selected["equity_pct"][0]) if "equity_pct" in selected.columns else 1.0
        syms       = selected["symbol"].to_list()
        n_sel      = len(syms)

        if weight_scheme == "vol_scaled" and "base_weight" in selected.columns:
            raw_ws_raw = [max(float(w), 0.0) for w in selected["base_weight"].to_list()]
            total      = sum(raw_ws_raw) or 1.0
            raw_ws     = [w / total for w in raw_ws_raw]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w * equity_pct, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
