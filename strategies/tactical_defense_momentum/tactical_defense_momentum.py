"""Tactical Defense Momentum — hybrid offense/defense momentum strategy.

QuantPipe adaptation of Strategy C — Momentum with Tactical Defense
(Faber 2007 regime filter, Antonacci 2014 dual momentum,
Moreira & Muir 2017 vol targeting).

Algorithm:
  1. Compute 4-component regime score (trend · breadth · vol · momentum).
     Same academic priors as Regime-Adaptive Dynamic Allocation (RADA).
  2. Map regime score to equity_pct:
       ≥ 0.65 (bull)       : 100% equity
       0.40 – 0.65 (neutral): linear 60% – 100%
       ≤ 0.40 (bear)       : floor at 40% (never fully exits equities)
  3. Score candidates by composite dual momentum:
       50% × 12-1m momentum  +  30% × 3-month momentum
       + 20% golden-cross bonus (SMA-50 > SMA-200)
  4. SMA-50 trend filter (loosed in bear regime: SMA-200 filter dropped).
  5. Select top_n candidates; score-proportional weighting.

Key distinction vs RADA: equity floor = 40% (never goes fully defensive).
Key distinction vs Aggressive Momentum: has a safe-cash allocation.

Target metrics: Sharpe > 0.7, CAGR > 12%, Max DD < 25%
Rebalance: Every ~10 trading days (approximated by monthly in QuantPipe)
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Tactical Defense Momentum"
DESCRIPTION = (
    "Regime-adaptive dual momentum with a 40% equity floor. "
    "Composite score blends 12-month, 3-month momentum and golden-cross bonus."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          6,
    "cost_bps":       8.0,
    "weight_scheme":  "equal",
}

_REGIME_REF    = "SPY"
_SECTORS       = ["XLK","XLV","XLF","XLE","XLI","XLU","XLP","XLY","XLC","XLB","XLRE"]
_SMA_FAST      = 50
_SMA_SLOW      = 200
_VOL_LOOKBACK  = 63
_VOL_HIGH      = 0.25
_VOL_LOW       = 0.12
_MOM_REG_LB    = 126
_W_TREND       = 0.40
_W_BREADTH     = 0.25
_W_VOL         = 0.20
_W_MOM         = 0.15
_REGIME_SMOOTH = 0.70
_MIN_EQ_W      = 0.10
_MOM_3M        = 63


def _compute_regime_score(prices_pd: pd.DataFrame, rd_ts: pd.Timestamp) -> float | None:
    hist = prices_pd[prices_pd.index <= rd_ts]
    if _REGIME_REF not in hist.columns:
        return None
    spy = hist[_REGIME_REF].dropna()
    if len(spy) < _SMA_SLOW:
        return None

    cur      = float(spy.iloc[-1])
    sma_fast = float(spy.iloc[-_SMA_FAST:].mean())
    sma_slow = float(spy.iloc[-_SMA_SLOW:].mean())
    trend = 0.0
    if cur      > sma_slow:  trend += 0.50
    if cur      > sma_fast:  trend += 0.25
    if sma_fast > sma_slow:  trend += 0.25

    above = total = 0
    for s in _SECTORS:
        if s not in hist.columns:
            continue
        col = hist[s].dropna()
        if len(col) < _SMA_FAST:
            continue
        total += 1
        if float(col.iloc[-1]) > float(col.iloc[-_SMA_FAST:].mean()):
            above += 1
    breadth = above / total if total > 0 else 0.5

    spy_arr  = spy.values
    log_rets = np.log(spy_arr[1:] / spy_arr[:-1])
    if len(log_rets) >= _VOL_LOOKBACK:
        ann_vol = float(np.std(log_rets[-_VOL_LOOKBACK:], ddof=1) * np.sqrt(252))
        vol_s   = 1.0 if ann_vol <= _VOL_LOW else (
                  0.0 if ann_vol >= _VOL_HIGH else
                  1.0 - (ann_vol - _VOL_LOW) / (_VOL_HIGH - _VOL_LOW))
    else:
        vol_s = 0.5

    if len(spy) >= _MOM_REG_LB + 1:
        mom_6m  = float(spy.iloc[-1]) / float(spy.iloc[-(_MOM_REG_LB + 1)]) - 1.0
        mom_s   = float(np.clip(0.5 + mom_6m * 2.5, 0.0, 1.0))
    else:
        mom_s = 0.5

    score = _W_TREND * trend + _W_BREADTH * breadth + _W_VOL * vol_s + _W_MOM * mom_s
    return float(np.clip(score, 0.0, 1.0))


def _regime_to_equity_pct(score: float) -> float:
    """Map regime score to equity allocation, with 40% floor."""
    if score >= 0.65:
        return 1.00
    elif score <= 0.40:
        return 0.40 + 0.20 * (score / 0.40)
    else:
        return 0.60 + 0.40 * (score - 0.40) / (0.65 - 0.40)


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    feat_syms = set(features["symbol"].unique().to_list())

    # Build price pivot for regime + trend filter + 3-month momentum
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
    smoothed_regime: float | None = None

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)

        # Regime score
        regime_score = 0.6
        if not prices_pd.empty:
            raw = _compute_regime_score(prices_pd, rd_ts)
            if raw is not None:
                smoothed_regime = (
                    raw if smoothed_regime is None
                    else _REGIME_SMOOTH * raw + (1.0 - _REGIME_SMOOTH) * smoothed_regime
                )
        if smoothed_regime is not None:
            regime_score = smoothed_regime

        equity_pct  = float(np.clip(_regime_to_equity_pct(regime_score), 0.40, 1.0))
        bear_regime = regime_score < 0.40

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

        # Score each candidate
        scores: list[tuple[str, float]] = []
        for sym in feat_syms:
            if sym not in mom_ser.index:
                continue
            mom_long = float(mom_ser[sym])
            if mom_long <= 0:
                continue

            # 3-month momentum from prices_df (optional)
            mom_3m = 0.0
            if not prices_pd.empty and sym in prices_pd.columns:
                p_hist = prices_pd[sym].dropna()
                p_hist = p_hist[p_hist.index <= rd_ts]
                if len(p_hist) >= _MOM_3M + 1:
                    mom_3m = float(p_hist.iloc[-1]) / float(p_hist.iloc[-_MOM_3M]) - 1.0

                # Trend filter
                if len(p_hist) >= _SMA_FAST:
                    sma_fast = float(p_hist.iloc[-_SMA_FAST:].mean())
                    if float(p_hist.iloc[-1]) <= sma_fast:
                        continue
                if not bear_regime and len(p_hist) >= _SMA_SLOW:
                    sma_slow = float(p_hist.iloc[-_SMA_SLOW:].mean())
                    if float(p_hist.iloc[-1]) <= sma_slow:
                        continue

                # Golden cross bonus
                golden = 0.0
                if len(p_hist) >= _SMA_SLOW:
                    sf = float(p_hist.iloc[-_SMA_FAST:].mean())
                    ss = float(p_hist.iloc[-_SMA_SLOW:].mean())
                    golden = 0.20 if sf > ss else 0.0
            else:
                golden = 0.0

            composite = 0.50 * mom_long + 0.30 * mom_3m + golden
            scores.append((sym, composite))

        if not scores:
            continue

        scores.sort(key=lambda x: -x[1])
        top_set = {s for s, _ in scores[:top_n]}

        for rank, (sym, comp) in enumerate(scores, 1):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(comp, 6),
                "rank":           rank,
                "selected":       sym in top_set,
                "regime_score":   round(regime_score, 4),
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
            floor      = _MIN_EQ_W / n_sel
            raw_ws     = [max(s / total, floor) for s in raw_scores]
            t2         = sum(raw_ws)
            raw_ws     = [w / t2 for w in raw_ws]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w * equity_pct, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
