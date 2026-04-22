"""Protective Breadth Momentum — PAA-based breadth-driven allocation.

QuantPipe adaptation of the Protective Asset Allocation strategy
(Keller & Keuning 2016) with inverse-vol weighting (Maillard et al. 2010)
and a volatility-targeting overlay (Moreira & Muir 2017).

Algorithm:
  1. Breadth count: N_positive = # risky assets with momentum_12m_1m > 0.
  2. Bond fraction (PAA): BF = clip((N - N_pos) / (N - PF·N/4), 0, 1).
     PF=2 → aggressive protection; BF=0 = fully invested, BF=1 = all cash.
  3. Equity allocation: top-K assets by positive momentum, inverse-vol weighted.
  4. Vol-targeting: scale equity weights so portfolio vol ≈ 8% annualised.
  5. Remaining weight stays as implicit cash (backtest engine treats it as 0% return).

Target metrics: Sharpe 0.8-1.3, Max DD 5-12%, Sortino > 1.2
Rebalance: Monthly
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Protective Breadth Momentum"
DESCRIPTION = (
    "PAA breadth count scales equity exposure; top-K assets by inverse-vol "
    "weighting; volatility-targeting overlay targets 8% annualized vol."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          6,   # TOP_K: max risky assets to hold
    "cost_bps":       5.0,
    "weight_scheme":  "equal",  # "equal" or "vol_scaled" (inverse vol)
}

_PROTECTION_FACTOR = 2     # PF in PAA formula (0=low, 1=mid, 2=high protection)
_VOL_TARGET        = 0.08  # 8% annualised portfolio volatility target
_VOL_N             = 63    # 3-month realized vol window
_RHO_EST           = 0.50  # assumed average pairwise correlation
_MIN_VOL           = 0.05  # vol floor for inverse-vol weighting


def _portfolio_vol(vols: list[float], weights: list[float], rho: float) -> float:
    """Estimate portfolio vol: σ²_p = (1-ρ)·Σw²σ² + ρ·(Σwσ)²."""
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
    """Breadth-filtered momentum signal with PAA equity fraction."""
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    has_vol = "realized_vol_21d" in features.columns
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

        snap_pd = snap.to_pandas().set_index("symbol")

        # Extract momentum (required) and vol (optional)
        mom_ser = snap_pd["momentum_12m_1m"].dropna()
        vol_ser = snap_pd["realized_vol_21d"].dropna() if has_vol else pd.Series(dtype=float)

        n_universe = len(mom_ser)
        if n_universe < 4:
            continue

        # PAA breadth formula
        n_positive = int((mom_ser > 0).sum())
        pf_floor   = _PROTECTION_FACTOR * n_universe / 4.0
        denom      = max(n_universe - pf_floor, 1.0)
        bond_frac  = float(np.clip((n_universe - n_positive) / denom, 0.0, 1.0))
        equity_pct = 1.0 - bond_frac

        # Select top-K positive-momentum assets
        positives = mom_ser[mom_ser > 0].sort_values(ascending=False)
        selected_syms = list(positives.index[:top_n])
        selected_set  = set(selected_syms)

        # Compute per-asset vol for inverse-vol weighting
        inv_vols: dict[str, float] = {}
        for sym in selected_syms:
            v = float(vol_ser[sym]) if sym in vol_ser.index else 0.15
            inv_vols[sym] = 1.0 / max(v, _MIN_VOL)

        total_iv  = sum(inv_vols.values()) or 1.0
        raw_ws    = {sym: iv / total_iv for sym, iv in inv_vols.items()}

        # Volatility-targeting scale
        if selected_syms and equity_pct > 0:
            vols_list = [
                max(float(vol_ser[s]) if s in vol_ser.index else 0.15, _MIN_VOL)
                for s in selected_syms
            ]
            w_list    = [raw_ws[s] for s in selected_syms]
            pv        = _portfolio_vol(vols_list, w_list, _RHO_EST)
            if pv > 0.001:
                scale      = min(_VOL_TARGET / pv, 1.0)
                equity_pct = min(equity_pct * scale, 1.0)

        for sym in mom_ser.index:
            mom = float(mom_ser[sym])
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(mom, 6),
                "rank":           int(mom_ser.rank(ascending=False)[sym]),
                "selected":       sym in selected_set,
                "equity_pct":     round(equity_pct, 4),
                "inv_vol_w":      round(raw_ws.get(sym, 0.0), 6),
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

        if weight_scheme == "vol_scaled" and "inv_vol_w" in selected.columns:
            raw_ws_raw = [max(float(w), 0.0) for w in selected["inv_vol_w"].to_list()]
            total      = sum(raw_ws_raw) or 1.0
            raw_ws     = [w / total for w in raw_ws_raw]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w * equity_pct, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
