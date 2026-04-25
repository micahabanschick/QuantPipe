"""Performance dashboard — Streamlit app (Dashboard #2).

Tabs: Overview · Portfolio · Risk · Analytics

Run with: streamlit run reports/performance_dashboard.py
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

from config.settings import DATA_DIR
from risk.engine import EQUITY_SECTOR_MAP, compute_exposures
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, apply_subplot_theme, range_selector,
    kpi_card, section_label, badge, page_header, status_banner,
)
from reports.pdf_export import build_performance_pdf
from risk.scenarios import run_all_scenarios
from storage.parquet_store import load_bars

# st.set_page_config() is called once in app.py; do not call it here.
# To run standalone: uncomment and add st.set_page_config(...) above st.markdown().
st.markdown(CSS, unsafe_allow_html=True)

PORTFOLIO_LOG  = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"
TARGET_WEIGHTS = DATA_DIR / "gold" / "equity" / "target_weights.parquet"


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _compute_stats(eq_values: np.ndarray) -> dict:
    r = np.diff(eq_values) / eq_values[:-1]
    n_years = len(eq_values) / 252
    total = eq_values[-1] / eq_values[0] - 1
    cagr  = (eq_values[-1] / eq_values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol   = r.std() * np.sqrt(252)
    ann   = r.mean() * 252
    sharpe = ann / vol if vol > 1e-10 else 0
    down = r[r < 0]
    sortino_denom = down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 1e-10 else 1e-10
    sortino = ann / sortino_denom
    peak = np.maximum.accumulate(eq_values)
    dd   = (eq_values - peak) / peak
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < -1e-10 else 0
    win_rate = float((r > 0).sum() / len(r))
    var95 = float(np.percentile(r, 5))
    var99 = float(np.percentile(r, 1))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    return dict(
        total=total, cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
        calmar=calmar, max_dd=max_dd, win_rate=win_rate,
        best_day=float(r.max()), worst_day=float(r.min()),
        var95=var95, var99=var99, cvar95=cvar95,
        daily_rets=r,
    )


def _trailing_returns(eq_pd: pd.DataFrame) -> dict[str, float | None]:
    v = eq_pd["portfolio_value"]
    last = v.iloc[-1]
    idx  = eq_pd.index

    def back(n):
        return last / v.iloc[-n] - 1 if len(v) > n else None

    def since(year, month, day=1):
        try:
            cut = pd.Timestamp(year, month, day)
            sub = v[idx >= cut]
            return last / sub.iloc[0] - 1 if len(sub) > 1 else None
        except Exception:
            return None

    last_ts = idx[-1]
    qm = ((last_ts.month - 1) // 3) * 3 + 1
    return {
        "MTD": since(last_ts.year, last_ts.month),
        "QTD": since(last_ts.year, qm),
        "YTD": since(last_ts.year, 1),
        "1Y":  back(252),
        "3Y":  back(756),
        "5Y":  back(1260),
    }


def _top_drawdowns(eq_values: np.ndarray, idx, top_n: int = 5) -> list[dict]:
    peak = np.maximum.accumulate(eq_values)
    dd   = (eq_values - peak) / peak
    records, i, n = [], 0, len(dd)

    def _d(x):
        return x.date() if hasattr(x, "date") else x

    while i < n:
        if dd[i] >= -5e-4:
            i += 1
            continue
        s, t = i, i
        while i < n and dd[i] < -5e-4:
            if dd[i] < dd[t]:
                t = i
            i += 1
        e = i if i < n else None
        records.append({
            "Start":       str(_d(idx[s])),
            "Trough":      str(_d(idx[t])),
            "Recovery":    str(_d(idx[e])) if e is not None else "Ongoing",
            "Drawdown":    f"{dd[t]:.2%}",
            "Depth (d)":   str(t - s),
            "Recov (d)":   str(e - t) if e is not None else "—",
            "_dd_val":     dd[t],
        })

    records.sort(key=lambda x: x["_dd_val"])
    for r in records:
        del r["_dd_val"]
    return records[:top_n]


def _rolling_metric(ret_series: pd.Series, window: int, metric: str) -> pd.Series:
    def _sharpe(x):
        s = x.std()
        return x.mean() / s * np.sqrt(252) if s > 1e-10 else np.nan

    def _sortino(x):
        d = x[x < 0]
        return x.mean() / d.std() * np.sqrt(252) if len(d) > 0 and d.std() > 1e-10 else np.nan

    fn = _sharpe if metric == "sharpe" else _sortino
    return ret_series.rolling(window).apply(fn, raw=False)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def _load_portfolio_log() -> pl.DataFrame | None:
    return pl.read_parquet(PORTFOLIO_LOG).sort("date") if PORTFOLIO_LOG.exists() else None


@st.cache_data(ttl=1800)
def _load_target_weights() -> pl.DataFrame | None:
    return pl.read_parquet(TARGET_WEIGHTS).sort(["date", "symbol"]) if TARGET_WEIGHTS.exists() else None


@st.cache_data(ttl=1800)
def _load_equity_curve(lookback_years: int) -> tuple[pl.DataFrame, pd.DataFrame] | None:
    """Returns (equity_pl, trades_pd) or None on failure."""
    try:
        from backtest.engine import run_backtest
        from features.compute import load_features
        from signals.momentum import (
            cross_sectional_momentum, get_monthly_rebalance_dates, momentum_weights,
        )
        from storage.universe import universe_as_of_date

        end   = date.today()
        start = date(end.year - lookback_years, end.month, end.day)
        syms  = universe_as_of_date("equity", end, require_data=True)
        if not syms:
            return None
        prices = load_bars(syms, start, end, "equity")
        if prices.is_empty():
            return None
        feats = load_features(syms, start, end, "equity",
                              feature_list=["momentum_12m_1m", "realized_vol_21d"])
        if feats.is_empty():
            return None
        td    = sorted(prices["date"].unique().to_list())
        rd    = get_monthly_rebalance_dates(start, end, td)
        sig   = cross_sectional_momentum(feats, rd, top_n=5)
        wts   = momentum_weights(sig, weight_scheme="equal")
        result = run_backtest(prices, wts, cost_bps=5.0)
        df = pl.from_pandas(result.equity_curve.reset_index())
        df.columns = ["date", "portfolio_value"]
        eq_pl = df.with_columns(pl.col("date").cast(pl.Date))

        # Enrich trades with human-readable dates from the equity curve index
        trades_pd = result.trades.copy() if not result.trades.empty else pd.DataFrame()
        return eq_pl, trades_pd
    except Exception:
        return None


@st.cache_data(ttl=1800)
def _load_benchmark(sym: str, start_str: str, end_str: str) -> pd.Series | None:
    try:
        prices = load_bars([sym], date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        if prices.is_empty():
            return None
        return prices.sort("date").to_pandas().set_index("date")["close"]
    except Exception:
        return None


_FACTOR_PROXY_SYMBOLS = ("SPY", "IWM", "IWB", "IWD", "IWF", "IWS", "IWP")


@st.cache_data(ttl=300)
def _load_factor_proxy_prices(start_str: str, end_str: str) -> pl.DataFrame | None:
    """Load OHLCV bars for all factor proxy ETFs."""
    try:
        prices = load_bars(
            list(_FACTOR_PROXY_SYMBOLS),
            date.fromisoformat(start_str),
            date.fromisoformat(end_str),
            "equity",
        )
        return None if prices.is_empty() else prices
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_correlation(symbols: list[str], lookback: int = 252) -> pd.DataFrame | None:
    try:
        end   = date.today()
        start = end - timedelta(days=lookback + 60)
        prices = load_bars(list(symbols), start, end, "equity")
        if prices.is_empty():
            return None
        pivot = prices.sort("date").to_pandas().pivot(index="date", columns="symbol", values="close")
        return pivot.pct_change().dropna().tail(lookback).corr()
    except Exception:
        return None


@st.cache_data(ttl=1800)
def _load_contribution_data(
    symbols: tuple,
    start_str: str,
    end_str: str,
    weights: tuple,
) -> pd.DataFrame | None:
    """Per-symbol daily contribution = weight × daily return over the backtest window."""
    try:
        prices = load_bars(list(symbols), date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        if prices.is_empty():
            return None
        pivot = prices.sort("date").to_pandas().pivot(index="date", columns="symbol", values="close")
        rets  = pivot.pct_change().dropna()
        wt    = dict(weights)
        contrib = pd.DataFrame({s: rets[s] * wt.get(s, 0.0) for s in symbols if s in rets.columns})
        return contrib
    except Exception:
        return None


# ── Page-level controls (horizontal strip above all tabs) ─────────────────────

_c1, _c2, _c3, _c4, _c5 = st.columns([2.5, 1.5, 1.5, 1, 1])
with _c1:
    lookback_years = st.select_slider(
        "Lookback",
        options=[1, 2, 3, 4, 5, 6, 7],
        value=6,
        format_func=lambda x: f"{x}yr",
        key="perf_lookback",
    )
with _c2:
    benchmark_sym = st.selectbox(
        "Benchmark",
        ["None", "SPY", "QQQ", "IWM", "AGG"],
        index=1,
        key="perf_benchmark",
    )
with _c3:
    rolling_window = st.selectbox(
        "Rolling window",
        [21, 63, 126, 252],
        index=1,
        format_func=lambda x: f"{x}d",
        key="perf_rolling",
    )
with _c4:
    show_rebal_lines = st.toggle(
        "Rebalances",
        value=True,
        key="perf_rebal",
    )
with _c5:
    pass   # spacer — download buttons follow below

st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

# ── Load & pre-process data ───────────────────────────────────────────────────

portfolio_log    = _load_portfolio_log()
target_weights_df = _load_target_weights()

with st.spinner("Running backtest…"):
    _bt_result = _load_equity_curve(lookback_years)

equity_df        = None
backtest_trades  = pd.DataFrame()

if _bt_result is not None:
    equity_df, backtest_trades = _bt_result

eq_pd        = None
stats        = None
trailing     = None
drawdowns    = None
daily_rets   = None

if equity_df is not None and not equity_df.is_empty():
    eq_pd = equity_df.to_pandas()
    eq_pd["date"] = pd.to_datetime(eq_pd["date"])
    eq_pd = eq_pd.set_index("date")
    eq_vals    = eq_pd["portfolio_value"].values
    stats      = _compute_stats(eq_vals)
    trailing   = _trailing_returns(eq_pd)
    drawdowns  = _top_drawdowns(eq_vals, eq_pd.index, top_n=5)
    daily_rets = pd.Series(stats["daily_rets"], index=eq_pd.index[1:])

# ── Factor model pre-computations ─────────────────────────────────────────────
_factor_returns = None
_factor_betas   = None
_factor_attribution = None

if daily_rets is not None and eq_pd is not None:
    try:
        from risk.factor_model import estimate_factor_returns, estimate_factor_betas
        from risk.attribution import factor_return_attribution

        _fp_start = eq_pd.index[0].date().isoformat()
        _fp_end   = eq_pd.index[-1].date().isoformat()
        _fp_prices_pl = _load_factor_proxy_prices(_fp_start, _fp_end)

        if _fp_prices_pl is not None:
            _factor_returns = estimate_factor_returns(_fp_prices_pl)
            _factor_betas   = estimate_factor_betas(daily_rets, _factor_returns)
            _factor_attribution = factor_return_attribution(
                daily_rets, _factor_returns, _factor_betas
            )
    except Exception:
        pass

# Current portfolio
current_weights: dict[str, float] = {}
exposures = None
if target_weights_df is not None and not target_weights_df.is_empty():
    latest_date = target_weights_df["date"].max()
    lw_df       = target_weights_df.filter(pl.col("date") == latest_date)
    current_weights = dict(zip(lw_df["symbol"].to_list(), lw_df["weight"].to_list()))
    exposures   = compute_exposures(current_weights)

# Page-level pre-computations (needed by PDF export and individual tabs)
stress: dict = run_all_scenarios(current_weights) if current_weights else {}

monthly_pivot: pd.DataFrame | None = None
if eq_pd is not None:
    try:
        _monthly = eq_pd["portfolio_value"].resample("ME").last().pct_change().dropna()
        _mdf = _monthly.to_frame()
        _mdf["Year"]  = _mdf.index.year
        _mdf["Month"] = _mdf.index.strftime("%b")
        monthly_pivot = _mdf.pivot_table(
            values="portfolio_value", index="Year", columns="Month"
        )
    except Exception:
        monthly_pivot = None

portfolio_log_pd: pd.DataFrame | None = (
    portfolio_log.to_pandas() if portfolio_log is not None and not portfolio_log.is_empty()
    else None
)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Performance",
        "Evaluate backtest results, risk metrics, factor exposure, and return analytics.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# ── Download bar ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def _build_pdf(
    stats_frozen,
    trailing_frozen,
    drawdowns_frozen,
    weights_frozen,
    stress_frozen,
    monthly_csv: str | None,
    lookback_years: int,
    benchmark_sym: str,
    as_of_str: str,
    eq_json: str | None,
    bench_json: str | None,
    daily_rets_json: str | None,
) -> bytes:
    import json as _json
    import io as _io
    from datetime import date as _date

    _stats    = dict(stats_frozen)                    if stats_frozen    else {}
    _trailing = dict(trailing_frozen)                 if trailing_frozen else {}
    _draws    = [dict(d) for d in drawdowns_frozen]   if drawdowns_frozen else []
    _weights  = dict(weights_frozen)                  if weights_frozen  else {}
    _stress   = dict(stress_frozen)                   if stress_frozen   else {}
    _monthly  = (pd.read_csv(_io.StringIO(monthly_csv)).set_index("Year")
                 if monthly_csv else None)

    def _unpack(js):
        if not js:
            return None, None
        d = _json.loads(js)
        return d["dates"], d["values"]

    eq_dates, eq_vals           = _unpack(eq_json)
    bench_dates, bench_vals     = _unpack(bench_json)
    dr_dates, dr_vals           = _unpack(daily_rets_json)

    from risk.engine import EQUITY_SECTOR_MAP as _SECTOR_MAP
    return build_performance_pdf(
        stats=_stats,
        trailing=_trailing,
        drawdowns=_draws,
        current_weights=_weights,
        exposures=None,
        sector_map=_SECTOR_MAP,
        stress=_stress,
        monthly_pivot=_monthly,
        lookback_years=lookback_years,
        benchmark_sym=benchmark_sym,
        as_of=_date.fromisoformat(as_of_str),
        eq_dates=eq_dates or [],
        eq_vals=eq_vals or [],
        bench_dates=bench_dates,
        bench_vals=bench_vals,
        daily_rets_dates=dr_dates,
        daily_rets_vals=dr_vals,
    )


# Prepare hashable/serialisable inputs for cache key
# Exclude daily_rets (numpy array) — not needed by the PDF generator
_stats_frozen    = tuple(sorted(
    (k, float(v)) for k, v in stats.items() if k != "daily_rets"
)) if stats else ()
_trailing_frozen = tuple(sorted(
    (k, float(v) if v is not None else 0.0) for k, v in trailing.items()
)) if trailing else ()
_draws_frozen    = tuple(tuple(sorted(d.items())) for d in (drawdowns or []))
_weights_frozen  = tuple(sorted(current_weights.items()))
_stress_frozen   = tuple(sorted(stress.items()))

_monthly_csv: str | None = None
if monthly_pivot is not None:
    try:
        _monthly_csv = monthly_pivot.reset_index().to_csv(index=False)
    except Exception:
        pass

# Trade history CSV — backtest transactions from result.trades
_trades_csv: str | None = None
if not backtest_trades.empty:
    try:
        _trades_csv = backtest_trades.to_csv(index=False)
    except Exception:
        pass

import json as _json_mod

_eq_json: str | None = None
if eq_pd is not None:
    try:
        _eq_json = _json_mod.dumps({
            "dates":  [str(d.date()) for d in eq_pd.index],
            "values": eq_pd["portfolio_value"].tolist(),
        })
    except Exception:
        pass

_bench_json: str | None = None
if benchmark_sym != "None" and eq_pd is not None:
    try:
        _bench_raw = _load_benchmark(
            benchmark_sym,
            eq_pd.index[0].date().isoformat(),
            eq_pd.index[-1].date().isoformat(),
        )
        if _bench_raw is not None:
            _bench_raw.index = pd.to_datetime(_bench_raw.index)
            _bench_aligned = _bench_raw.reindex(eq_pd.index, method="ffill").dropna()
            if not _bench_aligned.empty:
                _bench_norm = _bench_aligned / _bench_aligned.iloc[0] * eq_vals[0]
                _bench_json = _json_mod.dumps({
                    "dates":  [str(d.date()) for d in _bench_norm.index],
                    "values": _bench_norm.tolist(),
                })
    except Exception:
        pass

_daily_rets_json: str | None = None
if daily_rets is not None:
    try:
        _daily_rets_json = _json_mod.dumps({
            "dates":  [str(d.date()) for d in daily_rets.index],
            "values": daily_rets.tolist(),
        })
    except Exception:
        pass

# ── Information Ratio & Tracking Error (vs benchmark) ────────────────────────
_ir: float | None = None
_te: float | None = None
if daily_rets is not None and benchmark_sym != "None" and eq_pd is not None:
    try:
        _bench_for_ir = _load_benchmark(
            benchmark_sym,
            eq_pd.index[0].date().isoformat(),
            eq_pd.index[-1].date().isoformat(),
        )
        if _bench_for_ir is not None:
            _bench_for_ir.index = pd.to_datetime(_bench_for_ir.index)
            _bench_for_ir = _bench_for_ir.reindex(eq_pd.index, method="ffill").dropna()
            _bench_rets   = _bench_for_ir.pct_change().dropna()
            _active       = daily_rets.reindex(_bench_rets.index).dropna()
            _bench_rets   = _bench_rets.reindex(_active.index).dropna()
            _diff         = _active - _bench_rets
            _te           = float(_diff.std() * np.sqrt(252))
            _ir           = float(_diff.mean() / _diff.std() * np.sqrt(252)) if _diff.std() > 1e-10 else 0.0
    except Exception:
        pass

_dl1, _dl2, _dl3, _dl4 = st.columns([1, 1, 1, 5])

with _dl1:
    try:
        _pdf_bytes = _build_pdf(
            stats_frozen     = _stats_frozen,
            trailing_frozen  = _trailing_frozen,
            drawdowns_frozen = _draws_frozen,
            weights_frozen   = _weights_frozen,
            stress_frozen    = _stress_frozen,
            monthly_csv      = _monthly_csv,
            lookback_years   = lookback_years,
            benchmark_sym    = benchmark_sym,
            as_of_str        = date.today().isoformat(),
            eq_json          = _eq_json,
            bench_json       = _bench_json,
            daily_rets_json  = _daily_rets_json,
        )
        st.download_button(
            label="📄 PDF Report",
            data=_pdf_bytes,
            file_name=f"quantpipe_performance_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            help="Download full performance report (all tabs) as PDF",
        )
    except Exception as _pdf_err:
        st.button("📄 PDF Report", disabled=True, help=f"PDF unavailable: {_pdf_err}")

with _dl2:
    if _trades_csv:
        st.download_button(
            label="📊 Trade History",
            data=_trades_csv.encode(),
            file_name=f"quantpipe_trades_{date.today().isoformat()}.csv",
            mime="text/csv",
            help="Every backtest transaction — entry, exit, size, P&L",
        )
    else:
        st.button("📊 Trade History", disabled=True,
                  help="No backtest trades available — run the pipeline first")

with _dl3:
    if eq_pd is not None:
        _eq_csv = eq_pd.reset_index().rename(columns={"index": "date"}).to_csv(index=False)
        st.download_button(
            label="📈 Equity Curve",
            data=_eq_csv.encode(),
            file_name=f"quantpipe_equity_{date.today().isoformat()}.csv",
            mime="text/csv",
            help="Daily portfolio value as CSV",
        )
    else:
        st.button("📈 Equity Curve", disabled=True, help="No equity curve available")

st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_ov, tab_port, tab_risk, tab_analytics = st.tabs(
    ["  Overview  ", "  Portfolio  ", "  Risk  ", "  Analytics  "]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ov:
    if stats is None:
        st.warning("No equity curve available. Ensure data and features are populated.")
    else:
        # ── KPI row 1 ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, delta, dup, accent in [
            (c1, "Total Return",  f"{stats['total']:.1%}",   None, None,        COLORS["positive"]),
            (c2, "CAGR",          f"{stats['cagr']:.1%}",    None, None,        COLORS["teal"]),
            (c3, "Sharpe Ratio",  f"{stats['sharpe']:.2f}",  None, None,        COLORS["blue"]),
            (c4, "Sortino Ratio", f"{stats['sortino']:.2f}", None, None,        COLORS["purple"]),
        ]:
            col.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)

        # ── KPI row 2 ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, accent in [
            (c1, "Max Drawdown",    f"{stats['max_dd']:.1%}",   COLORS["negative"]),
            (c2, "Calmar Ratio",    f"{stats['calmar']:.2f}",   COLORS["orange"]),
            (c3, "Ann. Volatility", f"{stats['vol']:.1%}",      COLORS["warning"]),
            (c4, "Win Rate",        f"{stats['win_rate']:.1%}", COLORS["neutral"]),
        ]:
            col.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)

        # ── KPI row 3: benchmark-relative metrics (only when benchmark selected) ─
        if benchmark_sym != "None" and (_ir is not None or _te is not None):
            c1, c2, c3, c4 = st.columns(4)
            ir_val = f"{_ir:.2f}" if _ir is not None else "N/A"
            te_val = f"{_te:.1%}" if _te is not None else "N/A"
            ir_accent = COLORS["positive"] if (_ir or 0) >= 0 else COLORS["negative"]
            c1.markdown(kpi_card("Information Ratio", ir_val, accent=ir_accent), unsafe_allow_html=True)
            c2.markdown(kpi_card("Tracking Error", te_val, accent=COLORS["blue"]), unsafe_allow_html=True)
            st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)

        # ── Equity curve ──────────────────────────────────────────────────────
        st.markdown(section_label("Equity Curve"), unsafe_allow_html=True)

        fig_eq = go.Figure()

        # Optional benchmark
        bench_pd = None
        if benchmark_sym != "None":
            bench_raw = _load_benchmark(
                benchmark_sym,
                eq_pd.index[0].date().isoformat(),
                eq_pd.index[-1].date().isoformat(),
            )
            if bench_raw is not None:
                bench_idx = pd.to_datetime(bench_raw.index)
                bench_raw.index = bench_idx
                bench_pd = bench_raw.reindex(eq_pd.index, method="ffill").dropna()
                if not bench_pd.empty:
                    base = eq_vals[0]
                    bench_norm = bench_pd / bench_pd.iloc[0] * base
                    fig_eq.add_trace(go.Scatter(
                        x=bench_norm.index, y=bench_norm.values,
                        name=benchmark_sym,
                        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + benchmark_sym + ": $%{y:,.0f}<extra></extra>",
                    ))

        fig_eq.add_trace(go.Scatter(
            x=eq_pd.index, y=eq_vals,
            name="Portfolio",
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.07)",
            line=dict(color=COLORS["positive"], width=2.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Portfolio: $%{y:,.0f}<extra></extra>",
        ))

        # Rebalance date markers (add_shape handles date strings; add_vline annotation
        # breaks in Plotly 6.x when x is a date string due to mean-computation bug)
        if show_rebal_lines and target_weights_df is not None:
            rebal_col = "rebalance_date" if "rebalance_date" in target_weights_df.columns else "date"
            rdates = target_weights_df[rebal_col].unique().to_list()
            for rd in rdates:
                fig_eq.add_shape(
                    type="line",
                    x0=str(rd), x1=str(rd),
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=COLORS["border"], width=1, dash="dot"),
                )

        apply_theme(fig_eq, legend_inside=True)
        fig_eq.update_layout(
            height=360,
            yaxis=dict(tickformat="$,.0f"),
            xaxis=dict(rangeselector=range_selector()),
        )
        st.plotly_chart(fig_eq, width="stretch", config=PLOTLY_CONFIG)

        # ── Trailing returns bar ──────────────────────────────────────────────
        st.markdown(section_label("Trailing Returns"), unsafe_allow_html=True)

        tr_labels = [k for k, v in trailing.items() if v is not None]
        tr_values = [v for v in trailing.values() if v is not None]
        tr_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in tr_values]

        fig_tr = go.Figure(go.Bar(
            x=tr_labels,
            y=tr_values,
            marker=dict(color=tr_colors, line=dict(width=0)),
            text=[f"{v:.1%}" for v in tr_values],
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b>: %{y:.2%}<extra></extra>",
        ))
        fig_tr.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        apply_theme(fig_tr)
        fig_tr.update_layout(
            height=220,
            yaxis=dict(tickformat=".1%", showgrid=False),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_tr, width="stretch", config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_port:
    if not current_weights:
        st.info("No target weights found. Run: `uv run python orchestration/generate_signals.py`")
    else:
        # Header row
        col_h, col_b = st.columns([3, 1])
        with col_h:
            st.markdown(
                section_label(f"Latest Rebalance · {latest_date}"),
                unsafe_allow_html=True,
            )
        with col_b:
            if portfolio_log is not None and not portfolio_log.is_empty():
                passed = portfolio_log.tail(1).to_dicts()[0]["pre_trade_passed"]
                st.markdown(
                    "<br>" + badge("PRE-TRADE PASS", "positive") if passed
                    else "<br>" + badge("PRE-TRADE FAIL", "negative"),
                    unsafe_allow_html=True,
                )

        # ── Exposure KPIs ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions", exposures.n_positions)
        c2.metric("Gross Exposure", f"{exposures.gross_exposure:.1%}")
        c3.metric("Top-5 Concentration", f"{exposures.top_5_concentration:.1%}")
        c4.metric("Largest Name",
                  f"{exposures.largest_position[0]}",
                  delta=f"{exposures.largest_position[1]:.1%}")

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

        col_tbl, col_pie = st.columns([1, 1])

        # ── Holdings table ─────────────────────────────────────────────────────
        with col_tbl:
            st.markdown(section_label("Holdings"), unsafe_allow_html=True)
            syms   = list(current_weights.keys())
            wts    = list(current_weights.values())
            tbl_pd = pd.DataFrame({
                "Symbol": syms,
                "Weight": [f"{w:.1%}" for w in wts],
                "Sector": [EQUITY_SECTOR_MAP.get(s, "Other") for s in syms],
            })
            st.dataframe(tbl_pd, width="stretch", hide_index=True, height=220)

        # ── Sector donut ───────────────────────────────────────────────────────
        with col_pie:
            st.markdown(section_label("Sector Allocation"), unsafe_allow_html=True)
            if exposures.sector_exposures:
                sec_labels = list(exposures.sector_exposures.keys())
                sec_vals   = list(exposures.sector_exposures.values())
                fig_pie = go.Figure(go.Pie(
                    labels=sec_labels,
                    values=sec_vals,
                    hole=0.58,
                    marker=dict(
                        colors=COLORS["series"][:len(sec_labels)],
                        line=dict(color=COLORS["bg"], width=3),
                    ),
                    textinfo="label+percent",
                    textfont=dict(color=COLORS["text"], size=11),
                    hovertemplate="<b>%{label}</b><br>%{value:.1%}<extra></extra>",
                    direction="clockwise",
                    sort=True,
                ))
                apply_theme(fig_pie)
                fig_pie.update_layout(
                    height=240,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_pie, width="stretch", config=PLOTLY_CONFIG)

        # ── Position weight bars ───────────────────────────────────────────────
        st.markdown(section_label("Position Weights"), unsafe_allow_html=True)
        sorted_idx = sorted(range(len(wts)), key=lambda i: wts[i])
        s_syms = [syms[i] for i in sorted_idx]
        s_wts  = [wts[i]  for i in sorted_idx]

        fig_bars = go.Figure(go.Bar(
            x=s_wts, y=s_syms,
            orientation="h",
            marker=dict(
                color=[COLORS["positive"] if w >= max(s_wts) * 0.9 else COLORS["blue"]
                       for w in s_wts],
                line=dict(width=0),
            ),
            text=[f"{w:.1%}" for w in s_wts],
            textposition="outside",
            textfont=dict(color=COLORS["neutral"], size=11),
            hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
        ))
        apply_theme(fig_bars)
        fig_bars.update_layout(
            height=max(160, 38 * len(s_syms)),
            xaxis=dict(tickformat=".0%", showgrid=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_bars, width="stretch", config=PLOTLY_CONFIG)

        # ── Weight history (if multiple rebalance dates) ───────────────────────
        if target_weights_df is not None:
            all_dates = target_weights_df["date"].unique().sort().to_list()
            if len(all_dates) >= 2:
                st.markdown(section_label("Weight History"), unsafe_allow_html=True)
                history_pd = target_weights_df.to_pandas()
                history_pd["date"] = pd.to_datetime(history_pd["date"])
                fig_hist = go.Figure()
                for i, sym in enumerate(sorted(current_weights.keys())):
                    sym_data = history_pd[history_pd["symbol"] == sym].sort_values("date")
                    fig_hist.add_trace(go.Scatter(
                        x=sym_data["date"], y=sym_data["weight"],
                        name=sym,
                        mode="lines+markers",
                        line=dict(color=COLORS["series"][i % len(COLORS["series"])], width=2),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}: %{{y:.1%}}<extra></extra>",
                    ))
                apply_theme(fig_hist, legend_inside=True)
                fig_hist.update_layout(
                    height=240,
                    yaxis=dict(tickformat=".0%"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_hist, width="stretch", config=PLOTLY_CONFIG)

        # ── Contribution Analysis ──────────────────────────────────────────────
        if eq_pd is not None:
            st.markdown(section_label("Contribution Analysis"), unsafe_allow_html=True)
            _contrib = _load_contribution_data(
                tuple(sorted(current_weights.keys())),
                eq_pd.index[0].date().isoformat(),
                eq_pd.index[-1].date().isoformat(),
                tuple(sorted(current_weights.items())),
            )
            if _contrib is not None and not _contrib.empty:
                _total  = _contrib.sum().sort_values(ascending=False)
                _avgd   = _contrib.mean()
                _best5  = _contrib.sum(axis=1).nlargest(5).index
                _worst5 = _contrib.sum(axis=1).nsmallest(5).index

                col_ct, col_cb = st.columns([1, 1])
                with col_ct:
                    _ctbl = pd.DataFrame({
                        "Symbol":      _total.index.tolist(),
                        "Total Contrib": [f"{v:.2%}" for v in _total.values],
                        "Avg Daily":    [f"{_avgd.get(s, 0):.4%}" for s in _total.index],
                    })
                    st.dataframe(_ctbl, hide_index=True, width="stretch", height=220)
                with col_cb:
                    _ccolors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in _total.values]
                    fig_cb = go.Figure(go.Bar(
                        x=_total.values, y=_total.index,
                        orientation="h",
                        marker=dict(color=_ccolors, line=dict(width=0)),
                        text=[f"{v:.2%}" for v in _total.values],
                        textposition="outside",
                        textfont=dict(size=11, color=COLORS["text"]),
                        hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                    ))
                    fig_cb.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
                    apply_theme(fig_cb)
                    fig_cb.update_layout(
                        height=max(160, 38 * len(_total)),
                        xaxis=dict(tickformat=".1%", showgrid=False),
                        yaxis=dict(showgrid=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cb, width="stretch", config=PLOTLY_CONFIG)

                with st.expander("Best / Worst days breakdown"):
                    _bd = _contrib.loc[_best5].T
                    _bd.columns = [str(d.date()) if hasattr(d, "date") else str(d) for d in _bd.columns]
                    st.caption("Top-5 Best Days — per-position contribution")
                    st.dataframe(_bd.style.format("{:.4%}"), width="stretch")
                    _wd = _contrib.loc[_worst5].T
                    _wd.columns = [str(d.date()) if hasattr(d, "date") else str(d) for d in _wd.columns]
                    st.caption("Top-5 Worst Days — per-position contribution")
                    st.dataframe(_wd.style.format("{:.4%}"), width="stretch")
            else:
                st.info("Price data unavailable for contribution analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_risk:
    if stats is None:
        st.info("Equity curve required for risk analytics.")
    else:
        # ── Risk KPI row ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, accent in [
            (c1, "1-Day VaR 95%",    f"{stats['var95']:.2%}",   COLORS["negative"]),
            (c2, "1-Day VaR 99%",    f"{stats['var99']:.2%}",   COLORS["negative"]),
            (c3, "CVaR 95% (ES)",    f"{stats['cvar95']:.2%}",  COLORS["orange"]),
            (c4, "Max Drawdown",     f"{stats['max_dd']:.1%}",  COLORS["negative"]),
        ]:
            col.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, accent in [
            (c1, "Best Day",   f"{stats['best_day']:.2%}",   COLORS["positive"]),
            (c2, "Worst Day",  f"{stats['worst_day']:.2%}",  COLORS["negative"]),
            (c3, "Ann. Vol",   f"{stats['vol']:.1%}",        COLORS["warning"]),
            (c4, "Calmar",     f"{stats['calmar']:.2f}",     COLORS["blue"]),
        ]:
            col.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)

        col_stress, col_dd = st.columns([1, 1])

        # ── Stress scenario waterfall ──────────────────────────────────────────
        with col_stress:
            st.markdown(section_label("Stress Scenarios"), unsafe_allow_html=True)
            if stress:
                s_names = list(stress.keys())
                s_vals  = [v * 100 for v in stress.values()]
                s_colors = [COLORS["negative"] if v < 0 else COLORS["positive"] for v in s_vals]
                fig_wf = go.Figure(go.Bar(
                    x=s_names, y=s_vals,
                    marker=dict(color=s_colors, line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in s_vals],
                    textposition="outside",
                    textfont=dict(color=COLORS["text"], size=11),
                    hovertemplate="<b>%{x}</b><br>P&L: %{y:.1f}%<extra></extra>",
                ))
                fig_wf.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_wf)
                fig_wf.update_layout(
                    height=300,
                    yaxis=dict(title="Estimated P&L (%)", showgrid=False),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_wf, width="stretch", config=PLOTLY_CONFIG)
            else:
                st.info("No portfolio loaded for stress scenarios.")

        # ── Top drawdowns table ────────────────────────────────────────────────
        with col_dd:
            st.markdown(section_label("Top Drawdown Periods"), unsafe_allow_html=True)
            if drawdowns:
                dd_df = pd.DataFrame(drawdowns)
                st.dataframe(
                    dd_df.style.map(
                        lambda v: f"color:{COLORS['negative']}" if isinstance(v, str) and "%" in v and "-" in v else "",
                    ),
                    width="stretch",
                    hide_index=True,
                    height=290,
                )

        # ── Rolling volatility chart ───────────────────────────────────────────
        st.markdown(section_label("Rolling Risk Metrics"), unsafe_allow_html=True)

        roll_vol = daily_rets.rolling(rolling_window).std() * np.sqrt(252)

        fig_rv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.06,
                               subplot_titles=["", ""])
        fig_rv.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol.values,
            fill="tozeroy", fillcolor="rgba(246,173,85,0.1)",
            line=dict(color=COLORS["warning"], width=1.8),
            name=f"Rolling {rolling_window}d Vol",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>Vol</extra>",
        ), row=1, col=1)

        # Add VaR from portfolio log if available
        if portfolio_log is not None and len(portfolio_log) > 3:
            pl_pd = portfolio_log.select(["date", "var_1d_95"]).to_pandas()
            pl_pd["date"] = pd.to_datetime(pl_pd["date"])
            fig_rv.add_trace(go.Scatter(
                x=pl_pd["date"], y=pl_pd["var_1d_95"],
                line=dict(color=COLORS["negative"], width=1.8),
                name="VaR 95% (snapshot)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>VaR</extra>",
            ), row=1, col=1)

        # Drawdown series
        eq_vals_full = eq_pd["portfolio_value"].values
        peak   = np.maximum.accumulate(eq_vals_full)
        dd_arr = (eq_vals_full - peak) / peak
        fig_rv.add_trace(go.Scatter(
            x=eq_pd.index, y=dd_arr,
            fill="tozeroy", fillcolor="rgba(255,75,75,0.1)",
            line=dict(color=COLORS["negative"], width=1.5),
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>DD</extra>",
        ), row=2, col=1)

        apply_subplot_theme(fig_rv, height=400)
        fig_rv.update_yaxes(tickformat=".1%", row=1, col=1)
        fig_rv.update_yaxes(tickformat=".1%", row=2, col=1)
        fig_rv.update_layout(hovermode="x unified")
        st.plotly_chart(fig_rv, width="stretch", config=PLOTLY_CONFIG)

        # ── Section D: Factor Exposure ─────────────────────────────────────────
        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        st.markdown(section_label("Factor Exposure"), unsafe_allow_html=True)

        if _factor_betas is None or not _factor_betas.betas:
            st.info("Factor model unavailable — ensure factor proxy ETF prices are loaded.")
        else:
            _fb = _factor_betas
            _factor_names_list = list(_fb.betas.keys())
            _beta_vals = list(_fb.betas.values())

            # KPI row: one card per factor + R² + alpha
            _kpi_cols = st.columns(len(_factor_names_list) + 2)
            for _ci, (_fn, _bv) in enumerate(zip(_factor_names_list, _beta_vals)):
                _accent = COLORS["teal"] if _bv >= 0 else COLORS["negative"]
                _kpi_cols[_ci].markdown(
                    kpi_card(f"β {_fn}", f"{_bv:.3f}", accent=_accent),
                    unsafe_allow_html=True,
                )
            _kpi_cols[-2].markdown(
                kpi_card("R²", f"{_fb.r_squared:.3f}", accent=COLORS["blue"]),
                unsafe_allow_html=True,
            )
            _alpha_pct = _fb.alpha * 252  # annualise daily alpha
            _alpha_accent = COLORS["positive"] if _alpha_pct >= 0 else COLORS["negative"]
            _kpi_cols[-1].markdown(
                kpi_card("Alpha (ann.)", f"{_alpha_pct:.2%}", accent=_alpha_accent),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

            # Horizontal bar chart of betas
            _beta_colors = [
                COLORS["positive"] if v >= 0 else COLORS["negative"]
                for v in _beta_vals
            ]
            fig_betas = go.Figure(go.Bar(
                x=_beta_vals,
                y=_factor_names_list,
                orientation="h",
                marker=dict(color=_beta_colors, line=dict(width=0)),
                text=[f"{v:.3f}" for v in _beta_vals],
                textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{y}</b>: β=%{x:.4f}<extra></extra>",
            ))
            fig_betas.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_betas)
            fig_betas.update_layout(
                height=max(180, 45 * len(_factor_names_list)),
                xaxis=dict(showgrid=False, title="Beta"),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_betas, width="stretch", config=PLOTLY_CONFIG)

        # ── Section E: Factor Return Attribution ───────────────────────────────
        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        st.markdown(section_label("Factor Return Attribution"), unsafe_allow_html=True)

        if _factor_attribution is None or _factor_attribution.cumulative.empty:
            st.info("Factor attribution unavailable — ensure factor model computed successfully.")
        else:
            _fa = _factor_attribution
            _cum = _fa.cumulative.copy()

            fig_attr = go.Figure()
            _attr_colors = COLORS["series"] + [COLORS["neutral"]]
            for _i, _col in enumerate(_cum.columns):
                _color = _attr_colors[_i % len(_attr_colors)]
                _is_residual = _col == "Residual"
                fig_attr.add_trace(go.Scatter(
                    x=_cum.index,
                    y=_cum[_col].values,
                    name=_col,
                    mode="lines",
                    stackgroup="one",
                    line=dict(color=_color, width=0 if not _is_residual else 1),
                    fillcolor=_color.replace(")", ", 0.6)").replace("rgb(", "rgba(")
                               if _color.startswith("rgb") else _color,
                    hovertemplate=f"<b>{_col}</b><br>%{{x|%Y-%m-%d}}: %{{y:.4f}}<extra></extra>",
                ))
            apply_theme(fig_attr, legend_inside=False)
            fig_attr.update_layout(
                height=320,
                yaxis=dict(tickformat=".2%", title="Cumulative Contribution"),
                xaxis=dict(rangeselector=range_selector()),
                hovermode="x unified",
            )
            st.plotly_chart(fig_attr, width="stretch", config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    if stats is None or daily_rets is None:
        st.info("Equity curve required.")
    else:
        # ── Monthly returns heatmap ────────────────────────────────────────────
        st.markdown(section_label("Monthly Returns"), unsafe_allow_html=True)
        mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = (monthly_pivot.reindex(columns=[m for m in mo if m in monthly_pivot.columns])
                 if monthly_pivot is not None else None)

        if pivot is not None and not pivot.empty:
            z    = pivot.values
            text = [[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in z]

            fig_hm = go.Figure(go.Heatmap(
                z=z,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorscale="RdYlGn",
                zmid=0,
                showscale=True,
                colorbar=dict(tickformat=".0%", len=0.8, thickness=14),
                hovertemplate="<b>%{y} %{x}</b>: %{text}<extra></extra>",
            ))
            apply_theme(fig_hm)
            fig_hm.update_layout(
                height=max(180, 38 * len(pivot)),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_hm, width="stretch", config=PLOTLY_CONFIG)

        col_dist, col_corr = st.columns([1, 1])

        # ── Return distribution ────────────────────────────────────────────────
        with col_dist:
            st.markdown(section_label("Daily Return Distribution"), unsafe_allow_html=True)
            r = stats["daily_rets"]
            x_grid = np.linspace(r.min(), r.max(), 300)
            pdf_n  = norm.pdf(x_grid, r.mean(), r.std())

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=r,
                histnorm="probability density",
                nbinsx=80,
                marker=dict(color=COLORS["blue"], line=dict(width=0)),
                opacity=0.65,
                name="Daily returns",
                hovertemplate="%{x:.2%}: %{y:.4f}<extra></extra>",
            ))
            fig_dist.add_trace(go.Scatter(
                x=x_grid, y=pdf_n,
                line=dict(color=COLORS["warning"], width=2),
                name="Normal fit",
                hovertemplate="%{x:.2%}<extra>Normal</extra>",
            ))
            # VaR reference lines
            for v, label, color in [
                (stats["var95"], "VaR 95%", COLORS["negative"]),
                (stats["var99"], "VaR 99%", COLORS["purple"]),
            ]:
                fig_dist.add_vline(
                    x=v, line=dict(color=color, dash="dot", width=1.5),
                    annotation_text=label,
                    annotation_font=dict(color=color, size=10),
                    annotation_position="top",
                )
            apply_theme(fig_dist, legend_inside=True)
            fig_dist.update_layout(
                height=300,
                xaxis=dict(tickformat=".1%"),
                yaxis=dict(title="Density"),
                bargap=0.05,
            )
            st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)

        # ── Correlation matrix ─────────────────────────────────────────────────
        with col_corr:
            st.markdown(section_label("Holdings Correlation (252d)"), unsafe_allow_html=True)
            if len(current_weights) >= 2:
                corr_df = _load_correlation(tuple(sorted(current_weights.keys())))
                if corr_df is not None:
                    z_c  = corr_df.values
                    lbls = corr_df.columns.tolist()
                    txt  = [[f"{v:.2f}" for v in row] for row in z_c]
                    fig_corr = go.Figure(go.Heatmap(
                        z=z_c, x=lbls, y=lbls,
                        text=txt,
                        texttemplate="%{text}",
                        textfont=dict(size=11),
                        colorscale=[
                            [0.0, COLORS["negative"]],
                            [0.5, COLORS["card_bg"]],
                            [1.0, COLORS["positive"]],
                        ],
                        zmin=-1, zmax=1,
                        showscale=True,
                        colorbar=dict(
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                            thickness=14, len=0.9,
                        ),
                        hovertemplate="<b>%{y} / %{x}</b>: %{text}<extra></extra>",
                    ))
                    apply_theme(fig_corr)
                    fig_corr.update_layout(
                        height=300,
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_corr, width="stretch", config=PLOTLY_CONFIG)
                else:
                    st.info("Price data unavailable for correlation.")
            else:
                st.info("Need 2+ positions for correlation matrix.")

        # ── Rolling Sharpe & Sortino ───────────────────────────────────────────
        st.markdown(section_label(f"Rolling {rolling_window}d Risk-Adjusted Returns"), unsafe_allow_html=True)

        rs = _rolling_metric(daily_rets, rolling_window, "sharpe")
        so = _rolling_metric(daily_rets, rolling_window, "sortino")

        fig_ra = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.5, 0.5],
        )
        fig_ra.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            line=dict(color=COLORS["positive"], width=1.8),
            name="Sharpe",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Sharpe</extra>",
        ), row=1, col=1)
        fig_ra.add_hline(y=0, line=dict(color=COLORS["border"], width=1), row=1, col=1)
        fig_ra.add_hline(y=1, line=dict(color=COLORS["positive"], dash="dot", width=1), row=1, col=1)

        fig_ra.add_trace(go.Scatter(
            x=so.index, y=so.values,
            line=dict(color=COLORS["purple"], width=1.8),
            name="Sortino",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Sortino</extra>",
        ), row=2, col=1)
        fig_ra.add_hline(y=0, line=dict(color=COLORS["border"], width=1), row=2, col=1)
        fig_ra.add_hline(y=1, line=dict(color=COLORS["purple"], dash="dot", width=1), row=2, col=1)

        apply_subplot_theme(fig_ra, height=380)
        fig_ra.update_layout(hovermode="x unified", showlegend=True)
        st.plotly_chart(fig_ra, width="stretch", config=PLOTLY_CONFIG)

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
