"""Sector Rotation Momentum — long-only SPDR sector ETF rotation.

QuantPipe adaptation of the Dollar-Neutral Sector Rotation strategy
(Moskowitz & Grinblatt 1999, Faber 2010).

Original strategy: long top-3 sectors, short bottom-3 sectors (dollar-neutral).
QuantPipe adaptation: long-only top-N sectors by 12-1 month momentum.
The short leg is dropped; remaining capital stays as implicit cash.

Regime filter: when SPY is below its 200-day SMA, effective top_n is halved
and equity allocation drops to 50%, reducing drawdowns in bear markets.

Target metrics (long-only): Sharpe 0.6-1.0, Max DD 15-25%, Beta 0.5-0.8
Rebalance: Monthly
"""

import numpy as np
import pandas as pd
import polars as pl

NAME = "Sector Rotation Momentum"
DESCRIPTION = (
    "Long-only rotation across 11 SPDR sector ETFs by 12-1 month momentum. "
    "Regime filter (SPY vs SMA-200) halves equity exposure in bear markets."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          3,
    "cost_bps":       5.0,
    "weight_scheme":  "equal",
}

_SECTORS  = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLP", "XLY", "XLC", "XLB", "XLRE"]
_SMA_SLOW = 200


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Rank sector ETFs by 12-1 month momentum; apply SPY regime filter."""
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

    feat_syms  = set(features["symbol"].unique().to_list())
    candidates = [s for s in _SECTORS if s in feat_syms]
    if not candidates:
        candidates = list(feat_syms)

    # Build price pivot for SPY regime filter
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

        # Regime filter: SPY vs 200-day SMA
        regime_ok = True
        if not prices_pd.empty and "SPY" in prices_pd.columns:
            spy = prices_pd["SPY"].dropna()
            spy = spy[spy.index <= rd_ts]
            if len(spy) >= _SMA_SLOW:
                sma200 = float(spy.iloc[-_SMA_SLOW:].mean())
                if float(spy.iloc[-1]) < sma200:
                    regime_ok = False

        effective_top_n = top_n if regime_ok else max(1, top_n // 2)
        equity_pct      = 1.0   if regime_ok else 0.50

        # Feature snapshot
        snap = features.filter(pl.col(date_col) == rd)
        if snap.is_empty():
            avail   = sorted(features[date_col].unique().to_list())
            nearest = min(avail, key=lambda d: abs((pd.Timestamp(d) - rd_ts).days))
            snap    = features.filter(pl.col(date_col) == nearest)
        if snap.is_empty():
            continue

        snap_pd = snap.to_pandas().set_index("symbol")["momentum_12m_1m"].dropna()

        scored = [
            (sym, float(snap_pd[sym]))
            for sym in candidates
            if sym in snap_pd.index
        ]
        if not scored:
            continue

        scored.sort(key=lambda x: -x[1])
        top_set = {s for s, _ in scored[:effective_top_n]}

        for rank, (sym, mom) in enumerate(scored, 1):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(mom, 6),
                "rank":           rank,
                "selected":       sym in top_set,
                "regime_ok":      regime_ok,
                "equity_pct":     equity_pct,
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
        n_sel      = len(selected)
        syms       = selected["symbol"].to_list()

        if weight_scheme == "vol_scaled" and "score" in selected.columns:
            scores = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total  = sum(scores)
            raw_ws = [s / total for s in scores]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w * equity_pct, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
