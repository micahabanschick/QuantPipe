"""Volatility-Scaled Sector Momentum (VAMS) — momentum/vol scoring.

QuantPipe adaptation of Strategy X — Volatility-Scaled Sector Momentum
(Barroso & Santa-Clara 2015, Daniel & Moskowitz 2016, Moreira & Muir 2017,
Jegadeesh & Titman 1993, Faber 2007, Grossman & Zhou 1993).

Algorithm:
  1. VAMS score per sector:  score_i = mom_12_1(i) / max(vol_i, 5%)
     [Barroso & Santa-Clara 2015 — normalises momentum by realized vol]
  2. Three-gate eligibility: VAMS > 0, price > SMA-200, short momentum > 0.
     Short momentum (3-month) proxies the MACD confirmation from the original.
  3. Regime-dependent allocation:
       Drawdown ≥ −10%  (bull): top-N sectors by VAMS, vol-targeting overlay
       Drawdown < −10%  (warn): 1 sector + 75% defensive
       Drawdown < −15%  (crit): 100% defensive (weights sum to < 1 = cash)
  4. Volatility-targeting: scale equity weights to 15% annualised vol target.
     Max leverage cap = 1.0 (long-only; no leverage in QuantPipe).

Target metrics: Sortino > 1.2, CAGR > 15%, Max DD < 25%
Rebalance: Monthly (bi-weekly in original; weekly not practical in QuantPipe)
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Volatility-Scaled Sector Momentum"
DESCRIPTION = (
    "VAMS = momentum / vol ranks sectors; regime filter (SPY drawdown + SMA-200) "
    "shifts between concentrated sector bets and defensive allocation."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          3,
    "cost_bps":       7.0,
    "weight_scheme":  "equal",
}

_SECTORS       = ["XLK", "XLV", "XLE", "XLF", "XLI", "XLU", "XLP", "XLB", "XLRE"]
_SMA_PERIOD    = 200
_VOL_N         = 63
_MOM_3M        = 63
_MIN_VOL       = 0.05
_VOL_TARGET    = 0.15
_RHO_EST       = 0.55   # sectors share common equity beta
_DD_WARN       = -0.10
_DD_CRIT       = -0.15
_CHURN_THRESH  = 0.03


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
    """VAMS-scored sector momentum with regime-dependent allocation."""
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    has_vol   = "realized_vol_21d" in features.columns
    feat_syms = set(features["symbol"].unique().to_list())
    candidates = [s for s in _SECTORS if s in feat_syms]
    if not candidates:
        candidates = list(feat_syms)

    prices_pd = pd.DataFrame()
    if prices_df is not None and not prices_df.is_empty():
        pc = "adj_close" if "adj_close" in prices_df.columns else "close"
        prices_pd = (
            prices_df.select(["date", "symbol", pc])
            .to_pandas()
            .pivot(index="date", columns="symbol", values=pc)
            .sort_index()
        )
        prices_pd.index = pd.to_datetime(prices_pd.index)

    rows: list[dict] = []

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)

        # SPY drawdown from 252-day high
        spy_dd = 0.0
        if not prices_pd.empty and "SPY" in prices_pd.columns:
            spy = prices_pd["SPY"].dropna()
            spy = spy[spy.index <= rd_ts]
            if len(spy) >= 2:
                lookback = spy.iloc[-252:] if len(spy) >= 252 else spy
                peak     = float(lookback.max())
                if peak > 0:
                    spy_dd = (float(spy.iloc[-1]) - peak) / peak

        # Regime classification
        if spy_dd < _DD_CRIT:
            regime = "critical"
        elif spy_dd < _DD_WARN:
            regime = "warning"
        else:
            regime = "bull"

        # Feature snapshot
        snap = features.filter(pl.col(date_col) == rd)
        if snap.is_empty():
            avail   = sorted(features[date_col].unique().to_list())
            nearest = min(avail, key=lambda d: abs((pd.Timestamp(d) - rd_ts).days))
            snap    = features.filter(pl.col(date_col) == nearest)
        if snap.is_empty():
            continue

        snap_pd = snap.to_pandas().set_index("symbol")
        mom_ser = snap_pd["momentum_12m_1m"].dropna()
        vol_ser = snap_pd["realized_vol_21d"].dropna() if has_vol else pd.Series(dtype=float)

        if regime == "critical":
            # Emergency: no equity (implicit cash only)
            # Emit a placeholder row so the date appears in output
            rows.append({
                "rebalance_date": rd,
                "symbol":         candidates[0] if candidates else "SPY",
                "score":          0.0,
                "vams":           0.0,
                "rank":           1,
                "selected":       False,
                "regime":         regime,
                "equity_pct":     0.0,
            })
            continue

        # VAMS scoring
        sector_vams: list[tuple[str, float, float]] = []  # (sym, vams, vol)
        for sym in candidates:
            if sym not in mom_ser.index:
                continue
            mom = float(mom_ser[sym])
            v   = float(vol_ser[sym]) if sym in vol_ser.index else 0.18
            v   = max(v, _MIN_VOL)
            vams = mom / v

            # Gate 1: VAMS > 0 (positive momentum)
            if vams <= 0:
                continue

            # Gate 2: price > SMA-200 (from prices_df)
            if not prices_pd.empty and sym in prices_pd.columns:
                p_hist = prices_pd[sym].dropna()
                p_hist = p_hist[p_hist.index <= rd_ts]
                if len(p_hist) >= _SMA_PERIOD:
                    if float(p_hist.iloc[-1]) <= float(p_hist.iloc[-_SMA_PERIOD:].mean()):
                        continue
                # Gate 3: 3-month momentum > 0 (MACD proxy)
                if len(p_hist) >= _MOM_3M + 1:
                    mom_3m = float(p_hist.iloc[-1]) / float(p_hist.iloc[-_MOM_3M]) - 1.0
                    if mom_3m <= 0:
                        continue

            sector_vams.append((sym, vams, v))

        if regime == "warning":
            # 1 sector + 75% to implicit cash
            if sector_vams:
                best = max(sector_vams, key=lambda x: x[1])
                effective_top_n = 1
                equity_pct      = 0.25
            else:
                rows.append({
                    "rebalance_date": rd,
                    "symbol":         candidates[0] if candidates else "SPY",
                    "score":          0.0, "vams": 0.0, "rank": 1,
                    "selected":       False, "regime": regime, "equity_pct": 0.0,
                })
                continue
        else:
            effective_top_n = top_n
            equity_pct      = 1.0

        # Sort by VAMS, take top effective_top_n
        sector_vams.sort(key=lambda x: -x[1])
        ranked = sector_vams[:effective_top_n]
        top_set = {s for s, _, _ in ranked}

        # Vol-targeting overlay
        if ranked:
            vols_list    = [v for _, _, v in ranked]
            total_s      = sum(s for _, s, _ in ranked) or 1.0
            weights_list = [s / total_s for _, s, _ in ranked]
            pv = _portfolio_vol(vols_list, weights_list, _RHO_EST)
            if pv > 0.001:
                scale      = min(_VOL_TARGET / pv, 1.0)
                equity_pct = min(equity_pct * scale, 1.0)

        for rank, (sym, vams, v) in enumerate(sector_vams, 1):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(vams, 6),
                "vams":           round(vams, 6),
                "rank":           rank,
                "selected":       sym in top_set,
                "regime":         regime,
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

        if weight_scheme == "vol_scaled" and "score" in selected.columns:
            raw_scores = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total      = sum(raw_scores)
            raw_ws     = [s / total for s in raw_scores]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w * equity_pct, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
