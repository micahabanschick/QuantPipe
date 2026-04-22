"""Shium Optimal Momentum — multi-horizon time-series momentum with dynamic sizing.

QuantPipe adaptation of the HQG ShiumOptimalMomentum strategy.

Signal pipeline (per rebalance date):
  1. For each asset, count how many of 4 horizons [21, 63, 126, 252] show
     price_t > price_{t-h}  (positive trend at that lookback).
  2. ensemble_score = count / 4  -> values in {0, 0.25, 0.50, 0.75, 1.00}.
  3. equity_pct = mean(ensemble_scores across all universe assets)
     reflecting overall market-momentum "temperature".
  4. Select assets with score > 0; weight_i ∝ score_i (high-conviction
     assets get larger allocations within the invested sleeve).
  5. If no asset scores > 0 (all horizons negative) -> go fully to cash.

HQG -> QuantPipe changes:
  - Removed hqg_algorithms class framework -> functional get_signal / get_weights.
  - VUG (Vanguard Growth ETF) not in QuantPipe price store; universe extended to
    QuantPipe growth / size ETFs: SPY QQQ IWF IWB IWP IWS IWM IWO IWN.
    IWF is the closest single-asset proxy for VUG (Russell 1000 Growth).
  - Single-asset fixed allocation replaced with cross-sectional score-weighted
    multi-asset allocation; equity_pct derived from universe-average score so
    position sizing degrades gracefully when momentum breadth is weak.
  - Added __CASH__ sentinel so the backtest engine correctly resets to cash
    rather than forward-filling old weights.
"""

import numpy as np
import pandas as pd
import polars as pl

NAME        = "Shium Optimal Momentum"
DESCRIPTION = (
    "Multi-horizon time-series momentum ensemble: 4 lookbacks [21,63,126,252] "
    "score each asset {0,0.25,0.5,0.75,1.0}; weights ∝ score, equity_pct = "
    "universe mean score. Goes to cash when all scores are 0."
)
DEFAULT_PARAMS = {
    "lookback_years": 2,
    "top_n":          9,
    "cost_bps":       5.0,
    "weight_scheme":  "score_weighted",
}

_HORIZONS  = [21, 63, 126, 252]
_MAX_LB    = max(_HORIZONS)

_UNIVERSE  = ["SPY", "QQQ", "IWF", "IWB", "IWP", "IWS", "IWM", "IWO", "IWN"]


def _ensemble_score(prices: np.ndarray) -> float:
    """Return count of horizons where current price > h-days-ago price, / 4."""
    current = prices[-1]
    count = sum(1 for h in _HORIZONS if len(prices) > h and current > prices[-(h + 1)])
    return count / len(_HORIZONS)


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

    for rd in rebal_dates:
        rd_ts = pd.Timestamp(rd)
        hist  = price_wide.loc[price_wide.index <= rd_ts, avail].ffill()

        scores: dict[str, float] = {}
        for sym in avail:
            col = hist[sym].dropna().values
            if len(col) >= _MAX_LB + 1:
                scores[sym] = _ensemble_score(col)

        if not scores:
            rows.append(_cash_row(rd))
            continue

        equity_pct = float(np.mean(list(scores.values())))

        positive = {s: v for s, v in scores.items() if v > 0.0}

        if not positive:
            rows.append(_cash_row(rd))
            continue

        sorted_syms = sorted(positive, key=lambda s: -positive[s])
        selected_syms = set(sorted_syms[:top_n])

        score_arr = np.array([scores[s] for s in avail])
        rank_order = np.argsort(-score_arr)
        rank_of = {avail[i]: int(np.where(rank_order == i)[0][0]) + 1
                   for i in range(len(avail))}

        for sym in avail:
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(scores.get(sym, 0.0), 4),
                "rank":           rank_of[sym],
                "selected":       sym in selected_syms,
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
