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

@st.cache_data(ttl=300)
def _load_portfolio_log() -> pl.DataFrame | None:
    return pl.read_parquet(PORTFOLIO_LOG).sort("date") if PORTFOLIO_LOG.exists() else None


@st.cache_data(ttl=300)
def _load_target_weights() -> pl.DataFrame | None:
    return pl.read_parquet(TARGET_WEIGHTS).sort(["date", "symbol"]) if TARGET_WEIGHTS.exists() else None


@st.cache_data(ttl=300)
def _load_equity_curve(lookback_years: int) -> pl.DataFrame | None:
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
        return df.with_columns(pl.col("date").cast(pl.Date))
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_benchmark(sym: str, start_str: str, end_str: str) -> pd.Series | None:
    try:
        prices = load_bars([sym], date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        if prices.is_empty():
            return None
        return prices.sort("date").to_pandas().set_index("date")["close"]
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    lookback_years  = st.slider("Backtest lookback (years)", 1, 7, 6)
    rolling_window  = st.selectbox("Rolling window", [21, 63, 126, 252], index=1,
                                   format_func=lambda x: f"{x}d")
    benchmark_sym   = st.selectbox("Benchmark", ["None", "SPY", "QQQ", "IWM", "AGG"], index=1)
    show_rebal_lines = st.checkbox("Show rebalance dates", value=True)

# ── Load & pre-process data ───────────────────────────────────────────────────

portfolio_log    = _load_portfolio_log()
target_weights_df = _load_target_weights()

with st.spinner("Running backtest…"):
    equity_df = _load_equity_curve(lookback_years)

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

# Current portfolio
current_weights: dict[str, float] = {}
exposures = None
if target_weights_df is not None and not target_weights_df.is_empty():
    latest_date = target_weights_df["date"].max()
    lw_df       = target_weights_df.filter(pl.col("date") == latest_date)
    current_weights = dict(zip(lw_df["symbol"].to_list(), lw_df["weight"].to_list()))
    exposures   = compute_exposures(current_weights)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "QuantPipe — Performance",
        "Cross-sectional momentum · Top-5 ETFs · Equal weight · Monthly rebalance",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

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
            stress = run_all_scenarios(current_weights) if current_weights else {}
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    if stats is None or daily_rets is None:
        st.info("Equity curve required.")
    else:
        # ── Monthly returns heatmap ────────────────────────────────────────────
        st.markdown(section_label("Monthly Returns"), unsafe_allow_html=True)
        monthly = eq_pd["portfolio_value"].resample("ME").last().pct_change().dropna()
        m_df = monthly.to_frame()
        m_df["Year"]  = m_df.index.year
        m_df["Month"] = m_df.index.strftime("%b")
        pivot = m_df.pivot_table(values="portfolio_value", index="Year", columns="Month")
        mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex(columns=[m for m in mo if m in pivot.columns])
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
