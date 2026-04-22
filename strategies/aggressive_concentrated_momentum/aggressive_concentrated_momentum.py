"""Aggressive Concentrated Momentum — fully-invested dual-momentum top-N.

QuantPipe adaptation of the Aggressive Concentrated Momentum strategy
(Jegadeesh & Titman 1993, Faber 2007 SMA filter).

Algorithm:
  1. Score every ETF by composite dual momentum:
       50% × 12-1 month return  (skip-month momentum)
       30% × 3-month return      (short-term confirmation)
       20% × golden-cross bonus  (SMA-50 > SMA-200, from prices_df)
  2. Qualification filter: positive 12-month momentum AND price > SMA-50.
     In the absence of price data, fall back to momentum_12m_1m > 0 only.
  3. Select top_n by composite score; momentum-proportional weighting
     (floored at 10%/n_sel to avoid trivially small positions).
  4. Always fully invested (equity_pct = 1.0). Falls back to all assets
     with positive momentum if nothing passes the full filter.

Note: No cash or bond allocation. Drawdowns can be severe.

Target metrics: Sharpe > 0.8, Total Return > SPY, Max DD 25-35%
Rebalance: Bi-weekly in original; monthly in QuantPipe
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Aggressive Concentrated Momentum"
DESCRIPTION = (
    "Fully-invested top-N by composite dual-momentum score "
    "(12m + 3m + golden-cross). SMA-50 trend filter. No cash allocation."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          6,
    "cost_bps":       8.0,
    "weight_scheme":  "equal",
}

_SMA_FAST   = 50
_SMA_SLOW   = 200
_MOM_3M     = 63
_MIN_W_FRAC = 0.10  # minimum fraction of equal weight per position


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Composite dual-momentum scoring with SMA-50 qualification filter."""
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    feat_syms = set(features["symbol"].unique().to_list())

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

        snap = features.filter(pl.col(date_col) == rd)
        if snap.is_empty():
            avail   = sorted(features[date_col].unique().to_list())
            nearest = min(avail, key=lambda d: abs((pd.Timestamp(d) - rd_ts).days))
            snap    = features.filter(pl.col(date_col) == nearest)
        if snap.is_empty():
            continue

        snap_pd = snap.to_pandas().set_index("symbol")
        mom_ser = snap_pd["momentum_12m_1m"].dropna()

        scores: list[tuple[str, float]] = []
        for sym in feat_syms:
            if sym not in mom_ser.index:
                continue
            mom_long = float(mom_ser[sym])
            if mom_long <= 0:
                continue

            mom_3m = 0.0
            golden = 0.0

            if not prices_pd.empty and sym in prices_pd.columns:
                p_hist = prices_pd[sym].dropna()
                p_hist = p_hist[p_hist.index <= rd_ts]

                # SMA-50 trend filter
                if len(p_hist) >= _SMA_FAST:
                    sma_fast = float(p_hist.iloc[-_SMA_FAST:].mean())
                    if float(p_hist.iloc[-1]) <= sma_fast:
                        continue

                # 3-month momentum
                if len(p_hist) >= _MOM_3M + 1:
                    mom_3m = float(p_hist.iloc[-1]) / float(p_hist.iloc[-_MOM_3M]) - 1.0

                # Golden cross bonus
                if len(p_hist) >= _SMA_SLOW:
                    sf = float(p_hist.iloc[-_SMA_FAST:].mean())
                    ss = float(p_hist.iloc[-_SMA_SLOW:].mean())
                    golden = 0.20 if sf > ss else 0.0

            composite = 0.50 * mom_long + 0.30 * mom_3m + golden
            scores.append((sym, composite))

        # Fallback: use all positives if SMA filter leaves too few
        if len(scores) < min(top_n, 2):
            scores = [
                (sym, float(mom_ser[sym]))
                for sym in feat_syms
                if sym in mom_ser.index and float(mom_ser[sym]) > 0
            ]

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

        syms   = selected["symbol"].to_list()
        n_sel  = len(syms)

        if weight_scheme == "vol_scaled" and "score" in selected.columns:
            raw_scores = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total      = sum(raw_scores)
            floor      = _MIN_W_FRAC / n_sel
            raw_ws     = [max(s / total, floor) for s in raw_scores]
            t2         = sum(raw_ws)
            raw_ws     = [w / t2 for w in raw_ws]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
