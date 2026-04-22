"""ETF Pairs Relative Momentum — pair-wise momentum rotation.

QuantPipe adaptation of the OU Mean Reversion / Pairs Trading strategy
(Chan 2008, Avellaneda & Lee 2010).

Original strategy: Kalman-filter hedge ratios, OU process fitting, Z-score
entries, long/short dollar-neutral pairs trading.

QuantPipe adaptation: Long-only simplification. Within each defined pair,
go long the asset with higher 12-1 month momentum (relative momentum within
pairs). Only include a pair when the selected asset has positive absolute
momentum (beats cash proxy). This preserves the cross-sectional pair
structure while eliminating the short leg.

Note: This is MOMENTUM-based (trend following), not mean-reverting. The
OU/cointegration framework is incompatible with the long-only constraint.
For true mean reversion, a platform with short-selling is required.

Target metrics: Sharpe 0.5-0.9, Max DD 10-20%, Beta 0.3-0.6
Rebalance: Monthly
"""

import pandas as pd
import polars as pl

NAME = "ETF Pairs Relative Momentum"
DESCRIPTION = (
    "Within 8 related ETF pairs, selects the stronger asset by 12-1 month "
    "momentum. Only includes a pair when the winner has positive momentum."
)
DEFAULT_PARAMS = {
    "lookback_years": 6,
    "top_n":          5,   # max pairs held simultaneously
    "cost_bps":       7.0,
    "weight_scheme":  "equal",
}

# Same pairs as the original OU strategy
_PAIRS = [
    ("EWA", "EWC"),   # Australia / Canada
    ("GLD", "GDX"),   # Gold / Gold Miners
    ("XLF", "KBE"),   # Financials / Banks
    ("EWG", "EWQ"),   # Germany / France
    ("XLU", "XLP"),   # Utilities / Staples
    ("TLT", "IEF"),   # Long vs intermediate treasuries
    ("USO", "XLE"),   # Oil ETF / Energy sector
    ("EEM", "EFA"),   # Emerging vs developed
]


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    prices_df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """For each pair, select the stronger asset by 12-1 momentum.
    Only include the pair if the winner has positive absolute momentum.
    Rank included pairs by their winner's momentum score; keep top_n.
    """
    if not rebal_dates or features.is_empty():
        return pl.DataFrame()

    date_col = "rebalance_date" if "rebalance_date" in features.columns else "date"
    if "momentum_12m_1m" not in features.columns:
        return pl.DataFrame()

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

        snap_pd = snap.to_pandas().set_index("symbol")["momentum_12m_1m"].dropna()

        # Evaluate each pair
        eligible: list[tuple[str, float]] = []
        for a, b in _PAIRS:
            mom_a = float(snap_pd[a]) if a in snap_pd.index else None
            mom_b = float(snap_pd[b]) if b in snap_pd.index else None

            if mom_a is None and mom_b is None:
                continue
            if mom_a is None:
                winner, mom = b, mom_b
            elif mom_b is None:
                winner, mom = a, mom_a
            else:
                winner, mom = (a, mom_a) if mom_a >= mom_b else (b, mom_b)

            # Absolute momentum gate: winner must have positive return
            if mom > 0:
                eligible.append((winner, mom))

        # Rank by winner's momentum; take top_n
        eligible.sort(key=lambda x: -x[1])
        top_set = {s for s, _ in eligible[:top_n]}

        # All candidate symbols for output
        seen: set[str] = set()
        for rank, (sym, mom) in enumerate(eligible, 1):
            if sym in seen:
                continue
            seen.add(sym)
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(mom, 6),
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

        n_sel = len(selected)
        syms  = selected["symbol"].to_list()

        if weight_scheme == "vol_scaled" and "score" in selected.columns:
            scores = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total  = sum(scores)
            raw_ws = [s / total for s in scores]
        else:
            raw_ws = [1.0 / n_sel] * n_sel

        for sym, w in zip(syms, raw_ws):
            rows.append({"rebalance_date": rd, "symbol": sym, "weight": round(w, 6)})

    return pl.DataFrame(rows) if rows else pl.DataFrame()
