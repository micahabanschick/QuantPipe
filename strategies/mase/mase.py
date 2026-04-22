"""MASE — Mean-variance with Adaptive Signal Estimation.

QuantPipe adaptation of the HQG DynamicRegimeMVO strategy.

Signal pipeline (per rebalance date):
  1. Building-block expected returns blended with historical CAGR (67% hist).
  2. Stein shrinkage toward cross-sectional mean.
  3. Z-score normalise: mu_t = base_excess + tilt_strength * z.
  4. Ledoit-Wolf shrinkage covariance on monthly returns (annualised).
  5. Long-only max-Sharpe weights: w proportional to inv(Sigma) @ mu.
  6. Equity group cap: total equity weight <= 40%.
  7. Vol-target to 6% annual: equity_pct = min(scale, 1.0) holds cash residual.
     Do NOT re-normalise after scaling (original mase.py bug).

HQG -> QuantPipe changes:
  - Removed hqg_algorithms class framework -> functional get_signal/get_weights
  - Removed HMM regime governor (optional dep, unstable on monthly cadence)
  - Removed SHY cash sleeve (not in QuantPipe); uninvested portion = implicit cash
  - Universe adapted to QuantPipe price store (EFA/EEM/VNQ/IEF/SHY/TIP/DBC absent):
      SPY QQQ IWM XLRE XLE  -> equity / real-estate / energy
      AGG TLT               -> investment-grade + duration
      GLD                   -> commodity / inflation hedge
  - Added __CASH__ sentinel when covariance estimation fails
  - Fixed vol-targeting bug: scale but do not renormalise to 1.0
  - equity_pct = clip(0, 1) guard in get_weights
"""

import numpy as np
import pandas as pd
import polars as pl

NAME        = "MASE"
DESCRIPTION = (
    "Dynamic Regime-Aware MVO: building-block expected returns + "
    "Ledoit-Wolf covariance -> max-Sharpe, equity-capped, vol-targeted to 6%."
)
DEFAULT_PARAMS = {
    "lookback_years": 5,
    "top_n":          8,
    "cost_bps":       5.0,
    "weight_scheme":  "vol_scaled",
}

_UNIVERSE = ["SPY", "QQQ", "IWM", "XLRE", "XLE", "AGG", "TLT", "GLD"]

_ASSET_CLASSES: dict[str, str] = {
    "SPY":  "Equity",
    "QQQ":  "Equity",
    "IWM":  "Equity",
    "XLRE": "Equity",
    "XLE":  "Equity",
    "AGG":  "Fixed Income",
    "TLT":  "Fixed Income",
    "GLD":  "Commodity",
}

_ASSUMPTIONS: dict[str, dict[str, float]] = {
    "SPY":  {"dividend_yield": 1.5,  "buyback_yield": 1.0, "earnings_growth": 4.0,  "valuation_change": -1.5},
    "QQQ":  {"dividend_yield": 0.5,  "buyback_yield": 1.5, "earnings_growth": 7.0,  "valuation_change": -2.0},
    "IWM":  {"dividend_yield": 1.2,  "buyback_yield": 0.5, "earnings_growth": 5.0,  "valuation_change": -1.0},
    "XLRE": {"dividend_yield": 3.5,  "buyback_yield": 0.0, "earnings_growth": 2.0,  "valuation_change": -1.0},
    "XLE":  {"dividend_yield": 3.0,  "buyback_yield": 1.5, "earnings_growth": 2.0,  "valuation_change": -1.5},
    "AGG":  {"current_yield": 3.5,   "roll_return": 0.2,   "yield_change": -0.3},
    "TLT":  {"current_yield": 3.7,   "roll_return": 0.2,   "yield_change": -0.4},
    "GLD":  {"real_return": 0.0,     "inflation_assumption": 2.0},
}

_HIST_WEIGHT   = 0.67
_SHRINK_FACTOR = 0.50
_BASE_EXCESS   = 0.05
_TILT_STRENGTH = 0.03
_MAX_EQUITY    = 0.40
_VOL_TARGET    = 0.06
_MIN_MONTHS    = 24


def _building_block(ticker: str) -> float:
    a   = _ASSUMPTIONS[ticker]
    cls = _ASSET_CLASSES[ticker]
    if cls == "Equity":
        return (a["dividend_yield"] + a["buyback_yield"]
                + a["earnings_growth"] + a["valuation_change"])
    if cls == "Fixed Income":
        return a["current_yield"] + a["roll_return"] + a["yield_change"]
    return a["real_return"] + a["inflation_assumption"]


def _ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    X = returns.values.copy()
    T, N = X.shape
    X -= X.mean(axis=0)
    S  = np.cov(X, rowvar=False, bias=True)
    F  = (np.trace(S) / N) * np.eye(N)

    phi_hat = 0.0
    for t in range(T):
        xt = X[t, :, None]
        d  = xt @ xt.T - S
        phi_hat += float((d * d).sum())
    phi_hat /= max(T, 1)

    gamma_hat = float(((S - F) * (S - F)).sum())
    if gamma_hat <= 0:
        return S
    shrink = float(np.clip(phi_hat / gamma_hat / max(T, 1), 0.0, 1.0))
    return shrink * F + (1.0 - shrink) * S


def _mvo_weights(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    reg = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    w   = np.linalg.inv(reg) @ mu
    w   = np.clip(w, 0.0, None)
    s   = w.sum()
    return w / s if s > 0 else np.ones(len(mu)) / len(mu)


def _apply_equity_cap(w: np.ndarray, tickers: list[str]) -> np.ndarray:
    out     = w.copy()
    eq_mask = np.array([_ASSET_CLASSES.get(t) == "Equity" for t in tickers])
    eq_sum  = float(out[eq_mask].sum())
    if eq_sum <= _MAX_EQUITY or eq_sum <= 0:
        return out
    out[eq_mask] *= _MAX_EQUITY / eq_sum
    non_sum = float(out[~eq_mask].sum())
    excess  = eq_sum - _MAX_EQUITY
    if non_sum > 0:
        out[~eq_mask] += excess * out[~eq_mask] / non_sum
    s = out.sum()
    return out / s if s > 0 else out


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
    if len(avail) < 2:
        return pl.DataFrame()

    lookback_months = int(kwargs.get("lookback_years", DEFAULT_PARAMS["lookback_years"])) * 12

    def _cash_row(rd):
        return {"rebalance_date": rd, "symbol": "__CASH__", "score": 0.0,
                "rank": 0, "selected": False, "equity_pct": 0.0}

    rows: list[dict] = []

    for rd in rebal_dates:
        rd_ts   = pd.Timestamp(rd)
        hist    = price_wide.loc[price_wide.index <= rd_ts, avail].dropna(how="all")
        monthly = hist.resample("ME").last()

        active = [s for s in avail if monthly[s].notna().sum() >= _MIN_MONTHS]
        if len(active) < 2:
            rows.append(_cash_row(rd))
            continue

        window     = monthly[active].dropna().iloc[-(lookback_months + 1):]
        window_ret = window.pct_change(fill_method=None).dropna()

        if len(window_ret) < _MIN_MONTHS:
            rows.append(_cash_row(rd))
            continue

        mu_bb   = np.array([_building_block(s) / 100.0 for s in active])
        mu_hist = np.expm1(np.log1p(window_ret).mean().values * 12.0)
        mu_raw  = (1.0 - _HIST_WEIGHT) * mu_bb + _HIST_WEIGHT * mu_hist
        mu_raw  = (1.0 - _SHRINK_FACTOR) * mu_raw + _SHRINK_FACTOR * mu_raw.mean()
        z       = (mu_raw - mu_raw.mean()) / (mu_raw.std() + 1e-8)
        mu_t    = _BASE_EXCESS + _TILT_STRENGTH * z

        Sigma = _ledoit_wolf_cov(window_ret) * 12.0

        w_mvo = _mvo_weights(mu_t, Sigma)
        w_mvo = _apply_equity_cap(w_mvo, active)

        if top_n < len(active):
            order = np.argsort(-w_mvo)
            mask  = np.zeros(len(active), dtype=bool)
            mask[order[:top_n]] = True
            w_mvo = w_mvo * mask
            s     = w_mvo.sum()
            w_mvo = w_mvo / s if s > 0 else mask.astype(float) / mask.sum()

        port_vol   = float(np.sqrt(w_mvo @ Sigma @ w_mvo))
        scale      = min(_VOL_TARGET / port_vol, 1.0) if port_vol > 1e-6 else 1.0
        w_scaled   = w_mvo * scale
        equity_pct = float(np.clip(w_scaled.sum(), 0.0, 1.0))
        w_norm     = w_scaled / w_scaled.sum() if w_scaled.sum() > 1e-8 else w_scaled

        rank_order = np.argsort(-mu_t)
        rank_of    = {active[i]: int(np.where(rank_order == i)[0][0]) + 1
                      for i in range(len(active))}

        for j, sym in enumerate(active):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "score":          round(float(w_norm[j]), 6),
                "rank":           rank_of[sym],
                "selected":       bool(w_scaled[j] > 1e-6),
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
        syms  = selected["symbol"].to_list()
        n_sel = len(syms)

        if weight_scheme == "vol_scaled" and "score" in selected.columns:
            raw   = [max(float(s), 1e-8) for s in selected["score"].to_list()]
            total = sum(raw)
            props = [r / total for r in raw]
        else:
            props = [1.0 / n_sel] * n_sel

        for sym, prop in zip(syms, props):
            rows.append({
                "rebalance_date": rd,
                "symbol":         sym,
                "weight":         round(prop * equity_pct, 6),
            })

    return pl.DataFrame(rows) if rows else pl.DataFrame()
