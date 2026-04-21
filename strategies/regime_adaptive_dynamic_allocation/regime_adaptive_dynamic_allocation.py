"""Regime-Adaptive Dynamic Allocation (RADA)

Full QuantPipe adaptation of the RADA v3 algorithm (originally for QuantConnect).

Regime score (4 components, frozen academic priors):
  Trend   (40%) : SPY price vs SMA-50 and SMA-200 (Faber 2007)
  Breadth (25%) : fraction of independent breadth ETFs above SMA-50
  Volatility(20%): SPY annualized vol vs low/high thresholds (Moreira & Muir 2017)
  Momentum(15%) : SPY 6-month return as a macro trend confirmer

Signal construction:
  Skip-month momentum (12-1m) ranks investable ETFs.
  Trend filter applied: only symbols with price > SMA-50 qualify.
  Regime score → equity_pct; remainder is implicit cash (weights sum < 1).

Bias fixes vs naive momentum (see README for details):
  1. ETF-only universe — no individual stock survivorship bias.
  2. Parameters frozen to academic priors; no in-sample tuning.
  3. Independent breadth universe (sub-industry ETFs never held).
  4. Skip-month momentum (Jegadeesh-Titman 1993).
  5. EWMA regime smoother seeded from first valid score.
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Regime-Adaptive Dynamic Allocation"
DESCRIPTION = (
    "4-component regime score (trend · breadth · vol · momentum) sets equity/cash "
    "split; top-N skip-month momentum selects ETF positions within that allocation."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          5,
    "cost_bps":       5.0,
    "weight_scheme":  "momentum",   # "equal" or "momentum"
}

# ── Investable universe ────────────────────────────────────────────────────────
SECTORS    = ["XLK","XLV","XLF","XLE","XLI","XLU","XLP","XLY","XLC","XLB","XLRE"]
BROAD      = ["SPY","QQQ","IWM","EFA","VWO"]
ALT_ASSETS = ["TLT","GLD"]
REGIME_REF = "SPY"

# NOT held — used only for breadth measurement (independence fix)
_SUB_IND       = ["KRE","XBI","SMH","ITB","XRT","IBB"]
_BREADTH_UNIV  = SECTORS + _SUB_IND

_INVESTABLE    = SECTORS + BROAD + ALT_ASSETS

# ── Regime parameters (FROZEN — academic priors, do not tune on in-sample data) ─
_SMA_FAST      = 50
_SMA_SLOW      = 200
_VOL_LOOKBACK  = 63
_VOL_HIGH      = 0.25
_VOL_LOW       = 0.12
_MOM_REGIME    = 126   # 6-month SPY momentum for regime component (no skip)
_MOM_WINDOW    = 126   # selection window (12m-1m already in features, kept for docs)
_W_TREND       = 0.40
_W_BREADTH     = 0.25
_W_VOL         = 0.20
_W_MOM         = 0.15
_RISK_ON       = 0.65
_RISK_OFF      = 0.35
_REGIME_SMOOTH = 0.70
_MIN_EQ_W      = 0.10  # minimum weight floor per equity position before scaling


# ── Regime helpers ─────────────────────────────────────────────────────────────

def _compute_regime_score(price_history: pd.DataFrame) -> float | None:
    """4-component regime score from price history up to current date.

    price_history : pd.DataFrame (date × symbol), most-recent row = today.
    Returns float in [0, 1], or None if insufficient data for SPY.
    """
    if REGIME_REF not in price_history.columns:
        return None
    spy = price_history[REGIME_REF].dropna()
    if len(spy) < _SMA_SLOW:
        return None

    # Component 1: Trend (40%)
    cur      = float(spy.iloc[-1])
    sma_fast = float(spy.iloc[-_SMA_FAST:].mean())
    sma_slow = float(spy.iloc[-_SMA_SLOW:].mean())
    trend = 0.0
    if cur      > sma_slow:  trend += 0.50
    if cur      > sma_fast:  trend += 0.25
    if sma_fast > sma_slow:  trend += 0.25

    # Component 2: Breadth (25%) — on independent universe (not investable)
    above = 0; total = 0
    for sym in _BREADTH_UNIV:
        if sym not in price_history.columns:
            continue
        s = price_history[sym].dropna()
        if len(s) < _SMA_FAST:
            continue
        total += 1
        if float(s.iloc[-1]) > float(s.iloc[-_SMA_FAST:].mean()):
            above += 1
    breadth = above / total if total > 0 else 0.5

    # Component 3: Volatility (20%)
    spy_arr  = spy.values
    log_rets = np.log(spy_arr[1:] / spy_arr[:-1])
    if len(log_rets) >= _VOL_LOOKBACK:
        ann_vol = float(np.std(log_rets[-_VOL_LOOKBACK:], ddof=1) * np.sqrt(252))
        if ann_vol <= _VOL_LOW:    vol_s = 1.0
        elif ann_vol >= _VOL_HIGH: vol_s = 0.0
        else:                      vol_s = 1.0 - (ann_vol - _VOL_LOW) / (_VOL_HIGH - _VOL_LOW)
    else:
        vol_s = 0.5

    # Component 4: Macro momentum (15%) — SPY 6-month, no skip
    if len(spy) >= _MOM_REGIME + 1:
        mom_6m  = float(spy.iloc[-1]) / float(spy.iloc[-(_MOM_REGIME + 1)]) - 1.0
        mom_s   = float(np.clip(0.5 + mom_6m * 2.5, 0.0, 1.0))
    else:
        mom_s = 0.5

    score = _W_TREND * trend + _W_BREADTH * breadth + _W_VOL * vol_s + _W_MOM * mom_s
    return float(np.clip(score, 0.0, 1.0))


def _regime_to_equity_pct(score: float) -> float:
    """Map regime score → equity allocation percentage."""
    if score >= _RISK_ON:
        return 0.85 + 0.15 * (score - _RISK_ON) / (1.0 - _RISK_ON)
    elif score <= _RISK_OFF:
        return 0.30 * (score / _RISK_OFF if _RISK_OFF > 0 else 0.0)
    else:
        return 0.30 + 0.55 * (score - _RISK_OFF) / (_RISK_ON - _RISK_OFF)


# ── Public interface ───────────────────────────────────────────────────────────

def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    **kwargs,
) -> pl.DataFrame:
    """Regime-filtered, skip-month momentum signal.

    For each rebalance date:
      1. Compute 4-component regime score from SPY + sector ETF price history.
      2. Apply trend filter: drop symbols with price ≤ 50-day SMA.
      3. Rank remaining candidates by momentum_12m_1m (skip-month, already computed).
      4. Select top ``top_n``.

    Returns
    -------
    Polars DataFrame:
      [rebalance_date, symbol, score, rank, selected, regime_score, equity_pct]
    """
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    # ── Load prices for regime score + trend filter ────────────────────────────
    need_syms = list(set(_INVESTABLE + _BREADTH_UNIV + [REGIME_REF]))
    prices_pd = pd.DataFrame()
    try:
        from storage.parquet_store import load_bars
        history_start = (
            pd.Timestamp(min(rebal_dates)) - pd.Timedelta(days=420)
        ).date()
        bars_pl = load_bars(need_syms, history_start, pd.Timestamp(max(rebal_dates)).date(), "equity")
        if not bars_pl.is_empty():
            pc = "adj_close" if "adj_close" in bars_pl.columns else "close"
            prices_pd = (
                bars_pl.select(["date", "symbol", pc])
                .to_pandas()
                .pivot(index="date", columns="symbol", values=pc)
                .sort_index()
            )
            prices_pd.index = pd.to_datetime(prices_pd.index)
    except Exception:
        pass

    # ── Available investable symbols (must be in features) ─────────────────────
    feat_syms  = set(features["symbol"].unique().to_list())
    candidates = [s for s in _INVESTABLE if s in feat_syms]

    rows: list[dict] = []
    smoothed_regime: float | None = None

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)

        # ── Regime score (with EWMA smoother seeded from first valid value) ────
        regime_score = 0.5  # neutral default
        if not prices_pd.empty:
            hist = prices_pd[prices_pd.index <= rd_ts]
            raw = _compute_regime_score(hist)
            if raw is not None:
                smoothed_regime = (
                    raw if smoothed_regime is None
                    else _REGIME_SMOOTH * raw + (1.0 - _REGIME_SMOOTH) * smoothed_regime
                )
        if smoothed_regime is not None:
            regime_score = smoothed_regime
        equity_pct = float(np.clip(_regime_to_equity_pct(regime_score), 0.0, 1.0))

        # ── Feature snapshot at this rebalance date ────────────────────────────
        snap = features.filter(pl.col(date_col) == rd)
        if snap.is_empty():
            avail   = sorted(features[date_col].unique().to_list())
            nearest = min(avail, key=lambda d: abs((pd.Timestamp(d) - rd_ts).days))
            snap    = features.filter(pl.col(date_col) == nearest)
        if snap.is_empty():
            continue

        snap_pd = snap.to_pandas().set_index("symbol")["momentum_12m_1m"].dropna()

        # ── Trend filter: price > SMA-50 ───────────────────────────────────────
        passing: list[tuple[str, float]] = []
        for sym in candidates:
            if sym not in snap_pd.index:
                continue
            mom = float(snap_pd[sym])
            if mom <= 0:
                continue   # positive momentum required

            # Trend filter (skip if price data unavailable)
            if not prices_pd.empty and sym in prices_pd.columns:
                sym_hist = prices_pd[sym].dropna()
                sym_hist = sym_hist[sym_hist.index <= rd_ts]
                if len(sym_hist) >= _SMA_FAST:
                    if float(sym_hist.iloc[-1]) <= float(sym_hist.iloc[-_SMA_FAST:].mean()):
                        continue

            passing.append((sym, mom))

        # Fallback: use all positives if trend filter leaves fewer than top_n
        if len(passing) < min(top_n, 2):
            passing = [
                (s, float(snap_pd[s]))
                for s in candidates
                if s in snap_pd.index and not np.isnan(snap_pd[s])
            ]

        # ── Rank passing symbols by skip-month momentum ────────────────────────
        passing_sorted = sorted(passing, key=lambda x: -x[1])
        top_set = {s for s, _ in passing_sorted[:top_n]}

        # ── Build full output (all candidates, ranked among passing) ──────────
        rank_map = {s: i + 1 for i, (s, _) in enumerate(passing_sorted)}

        for sym in candidates:
            if sym not in snap_pd.index:
                continue
            mom = float(snap_pd[sym]) if sym in snap_pd.index else 0.0
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(mom, 6),
                "rank":           rank_map.get(sym, 999),
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
    """Convert signal to target weights.

    Equity weights sum to ``equity_pct`` from the regime score.
    The remaining (1 − equity_pct) is implicit cash; the backtest
    engine treats uninvested capital as a cash position.

    weight_scheme : "equal"    — equal weight among selected symbols
                    "momentum" — proportional to momentum score, floored at MIN_EQ_W/n
    """
    if signal.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in signal.columns else "date"
    rows: list[dict] = []

    for rd in signal[date_col].unique().sort().to_list():
        day      = signal.filter(pl.col(date_col) == rd)
        selected = day.filter(pl.col("selected"))

        if selected.is_empty():
            continue

        equity_pct = float(selected["equity_pct"][0])
        n_sel      = len(selected)
        syms       = selected["symbol"].to_list()
        scores     = selected["score"].to_list()

        if weight_scheme == "momentum":
            total_pos = sum(max(s, 0.0) for s in scores)
            if total_pos > 1e-12:
                raw_ws = [max(s, 0.0) / total_pos for s in scores]
            else:
                raw_ws = [1.0 / n_sel] * n_sel
            # Apply minimum weight floor so no position is trivially tiny
            floor  = _MIN_EQ_W / n_sel
            raw_ws = [max(w, floor) for w in raw_ws]
            total  = sum(raw_ws)
            final_ws = [w / total * equity_pct for w in raw_ws]
        else:
            final_ws = [equity_pct / n_sel] * n_sel

        for sym, w in zip(syms, final_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
