"""Shium Optimal Momentum — multi-horizon time-series momentum with dynamic sizing.

QuantPipe adaptation of the HQG ShiumOptimalMomentum strategy.

Signal pipeline (per rebalance date):
  1. Asymmetric weighted ensemble: horizons [21, 63, 126, 252] carry weights
     [0.10, 0.20, 0.30, 0.40] — longer horizons trusted more.
  2. weighted_score_i = sum(w_h * [price_t > price_{t-h}]) in [0, 1].
  3. Cross-sectional z-score of weighted_scores blended with time-series score:
     final_score = 0.6 * ts_score + 0.4 * z_score (normalised to [0,1]).
  4. Vol-scaling: final_score divided by asset's 63-day realised vol so that
     equal-scoring high-vol assets receive smaller allocations.
  5. SPY drawdown filter: if SPY is >10% below its 252-day peak, equity_pct
     is halved — reacts faster than waiting for SPY's own score to drop.
  6. equity_pct = mean(weighted_scores across universe) * spy_multiplier.
  7. Select top_n assets by final_score with score > 0.
  8. Transaction-cost dampening: a per-asset score is only updated when it
     changes by >= MIN_SCORE_CHANGE (0.10) vs the previous rebalance; otherwise
     last period's score is carried forward, cutting unnecessary turnover.
  9. If no asset passes filters -> __CASH__ sentinel (go fully to cash).

HQG -> QuantPipe changes:
  - Removed hqg_algorithms class framework -> functional get_signal / get_weights.
  - VUG (Vanguard Growth ETF) not in QuantPipe price store; universe extended to
    QuantPipe growth / size ETFs: SPY QQQ IWF IWB IWP IWS IWM IWO IWN.
    IWF is the closest single-asset proxy for VUG (Russell 1000 Growth).
  - Single-asset fixed allocation replaced with cross-sectional score-weighted
    multi-asset allocation.
  - Five improvements over the original: asymmetric horizon weights, vol scaling,
    SPY drawdown regime filter, transaction-cost dampening, cross-sectional blend.
  - Added __CASH__ sentinel so the backtest engine correctly resets to cash
    rather than forward-filling old weights.
"""

import numpy as np
import pandas as pd
import polars as pl

NAME        = "Shium Optimal Momentum"
DESCRIPTION = (
    "Multi-horizon momentum ensemble with asymmetric horizon weights, vol-scaling, "
    "SPY drawdown regime filter, cross-sectional blending, and turnover dampening. "
    "equity_pct = universe breadth score; positions ∝ vol-adjusted final score."
)
DEFAULT_PARAMS = {
    "lookback_years": 2,
    "top_n":          9,
    "cost_bps":       5.0,
    "weight_scheme":  "score_weighted",
}

_HORIZONS      = [21, 63, 126, 252]
_HOR_WEIGHTS   = [0.10, 0.20, 0.30, 0.40]   # longer horizons carry more weight
_MAX_LB        = max(_HORIZONS)
_VOL_WINDOW    = 63                           # days for realised vol estimate
_SPY_DD_THRESH = 0.10                         # SPY drawdown threshold
_CS_BLEND      = 0.40                         # cross-sectional share in final score
_MIN_SCORE_CHG = 0.10                         # min score delta to trigger rebalance

_UNIVERSE = ["SPY", "QQQ", "IWF", "IWB", "IWP", "IWS", "IWM", "IWO", "IWN"]


def _weighted_ts_score(prices: np.ndarray) -> float:
    """Asymmetric-weighted time-series momentum score in [0, 1]."""
    current = prices[-1]
    score = 0.0
    for h, w in zip(_HORIZONS, _HOR_WEIGHTS):
        if len(prices) > h and current > prices[-(h + 1)]:
            score += w
    return score


def _realised_vol(prices: np.ndarray, window: int) -> float:
    """Annualised realised vol from daily returns over last `window` days."""
    if len(prices) < window + 1:
        return 1.0
    rets = np.diff(np.log(prices[-window - 1:]))
    return float(rets.std() * np.sqrt(252)) or 1.0


def _spy_multiplier(spy_prices: np.ndarray) -> float:
    """1.0 normally; 0.5 if SPY is >10% below its 252-day peak."""
    if len(spy_prices) < 2:
        return 1.0
    peak = spy_prices[-min(252, len(spy_prices)):].max()
    drawdown = (spy_prices[-1] - peak) / peak
    return 0.5 if drawdown < -_SPY_DD_THRESH else 1.0


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    if not rebal_dates or prices_df is None or prices_df.is_empty():
        return pl.DataFrame()

    pc = "adj_close" if "adj_close" in prices_df.columns else "close"
    price_wide = (
        prices_df.select(["date", "symbol", pc])
        .to_pandas()
        .pivot(index="date", columns="symbol", values=pc)
        .sort_index()
    )
    price_wide.index = pd.to_datetime(price_wide.index)

    avail = [s for s in _UNIVERSE if s in price_wide.columns]
    if not avail:
        return pl.DataFrame()

    def _cash_row(rd):
        return {"rebalance_date": rd, "symbol": "__CASH__", "score": 0.0,
                "rank": 0, "selected": False, "equity_pct": 0.0}

    rows: list[dict] = []
    prev_scores: dict[str, float] = {}   # dampening: last committed score per asset

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)
        hist  = price_wide.loc[price_wide.index <= rd_ts, avail].ffill()

        # --- 1. Time-series weighted scores ---
        ts_scores: dict[str, float] = {}
        vols: dict[str, float] = {}
        for sym in avail:
            col = hist[sym].dropna().values
            if len(col) >= _MAX_LB + 1:
                ts_scores[sym] = _weighted_ts_score(col)
                vols[sym]      = _realised_vol(col, _VOL_WINDOW)

        if not ts_scores:
            rows.append(_cash_row(rd))
            continue

        # --- 2. Cross-sectional z-score of ts_scores, normalised to [0,1] ---
        sym_list = list(ts_scores.keys())
        ts_arr   = np.array([ts_scores[s] for s in sym_list])
        z        = (ts_arr - ts_arr.mean()) / (ts_arr.std() + 1e-8)
        z_norm   = (z - z.min()) / (z.max() - z.min() + 1e-8)

        # --- 3. Blend ts + cross-sectional ---
        blended: dict[str, float] = {}
        for i, sym in enumerate(sym_list):
            blended[sym] = (1.0 - _CS_BLEND) * ts_scores[sym] + _CS_BLEND * float(z_norm[i])

        # --- 4. Vol-adjust: score / vol (then re-normalise within universe) ---
        vol_adj: dict[str, float] = {}
        for sym in sym_list:
            vol_adj[sym] = blended[sym] / vols.get(sym, 1.0)
        va_arr  = np.array([vol_adj[sym] for sym in sym_list])
        va_min, va_max = va_arr.min(), va_arr.max()
        va_norm = (va_arr - va_min) / (va_max - va_min + 1e-8)
        final_scores: dict[str, float] = {sym_list[i]: float(va_norm[i]) for i in range(len(sym_list))}

        # --- 5. Turnover dampening: carry forward if change < MIN_SCORE_CHG ---
        dampened: dict[str, float] = {}
        for sym in sym_list:
            new  = final_scores[sym]
            prev = prev_scores.get(sym, -999.0)
            dampened[sym] = new if abs(new - prev) >= _MIN_SCORE_CHG else prev
        prev_scores = dampened.copy()

        # --- 6. SPY drawdown regime filter -> equity_pct multiplier ---
        spy_mult = 1.0
        if "SPY" in hist.columns:
            spy_col = hist["SPY"].dropna().values
            spy_mult = _spy_multiplier(spy_col)

        equity_pct = float(np.mean(list(ts_scores.values()))) * spy_mult

        # --- 7. Select top_n by dampened score, require score > 0 ---
        positive = {s: v for s, v in dampened.items() if v > 1e-8}
        if not positive:
            rows.append(_cash_row(rd))
            prev_scores = {}
            continue

        sorted_syms   = sorted(positive, key=lambda s: -positive[s])
        selected_syms = set(sorted_syms[:top_n])

        score_arr  = np.array([dampened.get(s, 0.0) for s in avail])
        rank_order = np.argsort(-score_arr)
        rank_of    = {avail[i]: int(np.where(rank_order == i)[0][0]) + 1
                      for i in range(len(avail))}

        for sym in avail:
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(dampened.get(sym, 0.0), 6),
                "rank":           rank_of[sym],
                "selected":       sym in selected_syms,
                "equity_pct":     round(float(np.clip(equity_pct, 0.0, 1.0)), 4),
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
        selected = day.filter(pl.col("selected")).filter(pl.col("symbol") != "__CASH__")

        if selected.is_empty():
            rows.append({"rebalance_date": rd, "symbol": "__CASH__", "weight": 0.0})
            continue

        equity_pct = float(np.clip(
            selected["equity_pct"][0] if "equity_pct" in selected.columns else 1.0,
            0.0, 1.0,
        ))
        syms = selected["symbol"].to_list()

        if weight_scheme == "score_weighted" and "score" in selected.columns:
            raw   = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total = sum(raw)
            props = [r / total for r in raw]
        else:
            props = [1.0 / len(syms)] * len(syms)

        for sym, prop in zip(syms, props):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "weight":         round(prop * equity_pct, 6),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()
