"""Pipeline health dashboard — Streamlit app (Dashboard #1).

Tabs: Status · Data Quality · Portfolio State · Logs

Run with: streamlit run reports/health_dashboard.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, apply_subplot_theme,
    kpi_card, section_label, badge, page_header, status_banner,
)
from storage.parquet_store import list_symbols, load_bars

st.set_page_config(
    page_title="QuantPipe — Pipeline Health",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

LOOKBACK          = 30
TARGET_WEIGHTS    = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
PORTFOLIO_LOG     = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"
PIPELINE_LOG      = LOGS_DIR / "pipeline.log"
INGEST_LOG        = LOGS_DIR / "ingest.log"
SIGNALS_LOG       = LOGS_DIR / "signals.log"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _last_modified(asset_class: str) -> tuple[str, float]:
    base  = DATA_DIR / "bronze" / asset_class / "daily"
    if not base.exists():
        return "No data", float("inf")
    files = list(base.rglob("*.parquet"))
    if not files:
        return "No files", float("inf")
    latest   = max(files, key=lambda p: p.stat().st_mtime)
    ts       = datetime.fromtimestamp(latest.stat().st_mtime)
    age_h    = (datetime.now() - ts).total_seconds() / 3600
    return ts.strftime("%Y-%m-%d %H:%M"), age_h


def _signals_freshness() -> tuple[str, float, bool]:
    if not TARGET_WEIGHTS.exists():
        return "No signals file", float("inf"), False
    mtime    = datetime.fromtimestamp(TARGET_WEIGHTS.stat().st_mtime)
    age_h    = (datetime.now() - mtime).total_seconds() / 3600
    is_today = mtime.date() == date.today()
    return mtime.strftime("%Y-%m-%d %H:%M"), age_h, is_today


def _last_log_run(log_path: Path) -> tuple[str, float, bool]:
    if not log_path.exists():
        return "No log file", float("inf"), False
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        for line in reversed(lines):
            if "Pipeline complete" in line or "pipeline complete" in line.lower():
                parts = line.split()
                ts    = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "unknown"
                ok    = "0 failure" in line or "0 ingestion failure" in line
                try:
                    dt    = datetime.fromisoformat(ts)
                    age_h = (datetime.now() - dt).total_seconds() / 3600
                except Exception:
                    age_h = float("inf")
                return ts, age_h, ok
        return "No complete marker", float("inf"), False
    except Exception:
        return "Log read error", float("inf"), False


def _tail_log(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return ["Log file not found"]
    lines = path.read_text(errors="replace").splitlines()
    return lines[-n:]


def _count_errors(lines: list[str]) -> int:
    return sum(1 for l in lines if " ERROR " in l or "FAILED" in l or "ALERT" in l)


# ── Compute status ────────────────────────────────────────────────────────────

eq_ts,   eq_age   = _last_modified("equity")
cr_ts,   cr_age   = _last_modified("crypto")
sig_ts,  sig_age, sig_fresh  = _signals_freshness()
pipe_ts, pipe_age, pipe_ok   = _last_log_run(PIPELINE_LOG)

any_missing  = eq_age == float("inf") or cr_age == float("inf")
any_stale    = eq_age > 30 or cr_age > 30 or not sig_fresh or not pipe_ok
all_healthy  = not any_missing and not any_stale

if any_missing:
    system_color  = COLORS["negative"]
    system_text   = "CRITICAL — Missing data source"
    animate       = True
elif any_stale:
    system_color  = COLORS["warning"]
    system_text   = "WARNING — Some data is stale or pipeline incomplete"
    animate       = False
else:
    system_color  = COLORS["positive"]
    system_text   = "HEALTHY — All systems nominal"
    animate       = False

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "QuantPipe — Pipeline Health",
        "Data ingestion · Feature computation · Signal generation",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_status, tab_data, tab_port, tab_logs = st.tabs(
    ["  Status  ", "  Data Quality  ", "  Portfolio State  ", "  Logs  "]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATUS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_status:
    # System banner
    st.markdown(
        status_banner(system_text, system_color, animate=animate),
        unsafe_allow_html=True,
    )

    # ── Pipeline component KPI cards ──────────────────────────────────────────
    st.markdown(section_label("Pipeline Components"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, ts, age, healthy, threshold in [
        (c1, "Equity Ingest",   eq_ts,   eq_age,   eq_age  < 30,   30),
        (c2, "Crypto Ingest",   cr_ts,   cr_age,   cr_age  < 30,   30),
        (c3, "Signal Generate", sig_ts,  sig_age,  sig_fresh,       24),
        (c4, "Last Pipeline",   pipe_ts, pipe_age, pipe_ok,         48),
    ]:
        accent = (
            COLORS["positive"] if healthy else
            COLORS["warning"]  if age < threshold * 2 else
            COLORS["negative"]
        )
        age_str = f"{age:.0f}h ago" if age < float("inf") else "Unknown"
        col.markdown(
            kpi_card(label, ts, delta=age_str, delta_up=healthy, accent=accent),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

    # ── Data freshness visual ──────────────────────────────────────────────────
    st.markdown(section_label("Data Freshness (hours since last update)"), unsafe_allow_html=True)

    MAX_H = 72
    sources   = ["Equity Prices", "Crypto Prices", "Signals", "Pipeline Run"]
    ages_raw  = [eq_age, cr_age, sig_age, pipe_age]
    ages_disp = [min(a, MAX_H) for a in ages_raw]
    bar_colors = [
        COLORS["positive"] if a < 24 else
        COLORS["warning"]  if a < 48 else
        COLORS["negative"]
        for a in ages_raw
    ]
    age_text = [
        f"{a:.1f}h" if a < float("inf") else "N/A"
        for a in ages_raw
    ]

    fig_fresh = go.Figure(go.Bar(
        x=ages_disp,
        y=sources,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=age_text,
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=12),
        hovertemplate="<b>%{y}</b>: %{text}<extra></extra>",
    ))
    fig_fresh.add_vline(x=24, line=dict(color=COLORS["warning"], dash="dot", width=1.5),
                        annotation_text="24h warning",
                        annotation_font=dict(color=COLORS["warning"], size=10),
                        annotation_position="top right")
    fig_fresh.add_vline(x=48, line=dict(color=COLORS["negative"], dash="dot", width=1.5),
                        annotation_text="48h critical",
                        annotation_font=dict(color=COLORS["negative"], size=10),
                        annotation_position="top right")
    apply_theme(fig_fresh)
    fig_fresh.update_layout(
        height=210,
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(range=[0, MAX_H + 14], showgrid=False, title="Hours"),
        showlegend=False,
    )
    st.plotly_chart(fig_fresh, width="stretch", config=PLOTLY_CONFIG)

    # ── Scheduled task info ────────────────────────────────────────────────────
    st.markdown(section_label("Scheduled Tasks"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Daily Pipeline", "06:15 Mon–Fri", accent=COLORS["blue"]),   unsafe_allow_html=True)
    c2.markdown(kpi_card("Daily Rebalance","16:30 Mon–Fri", accent=COLORS["purple"]), unsafe_allow_html=True)
    c3.markdown(kpi_card("Scheduler",       "Windows Task Scheduler", accent=COLORS["neutral"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    eq_symbols = list_symbols("equity")
    cr_symbols = list_symbols("crypto")

    # ── Universe KPIs ──────────────────────────────────────────────────────────
    st.markdown(section_label("Universe Coverage"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity Symbols", len(eq_symbols))
    c2.metric("Crypto Symbols", len(cr_symbols))
    c3.metric("Total Symbols",  len(eq_symbols) + len(cr_symbols))

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    # ── Equity row counts ──────────────────────────────────────────────────────
    st.markdown(section_label(f"Equity Symbol Status — Last {LOOKBACK} Days"), unsafe_allow_html=True)
    end_d   = date.today()
    start_d = end_d - timedelta(days=LOOKBACK)

    if eq_symbols:
        sample = sorted(eq_symbols)[:26]
        df_bars = load_bars(sample, start_d, end_d, asset_class="equity")
        if not df_bars.is_empty():
            counts = (
                df_bars.group_by("symbol")
                .agg(
                    pl.len().alias("rows"),
                    pl.max("date").alias("latest_date"),
                    pl.min("date").alias("earliest_date"),
                )
                .sort("symbol")
                .with_columns(
                    pl.when(pl.col("latest_date") < (end_d - timedelta(days=5)))
                    .then(pl.lit("STALE"))
                    .otherwise(pl.lit("OK"))
                    .alias("status")
                )
            )
            n_stale = counts.filter(pl.col("status") == "STALE").height
            if n_stale > 0:
                st.warning(f"{n_stale} symbol(s) stale (latest date > 5 days old)")
            else:
                st.success("All symbols have fresh data")

            counts_pd = counts.to_pandas()

            col_tbl, col_chart = st.columns([1, 2])

            with col_tbl:
                st.dataframe(
                    counts_pd.style.apply(
                        lambda col: [
                            f"color:{COLORS['negative']};font-weight:600" if v == "STALE"
                            else f"color:{COLORS['positive']}"
                            for v in col
                        ] if col.name == "status" else [""] * len(col),
                        axis=0,
                    ),
                    width="stretch",
                    hide_index=True,
                    height=420,
                )

            with col_chart:
                counts_sorted = counts_pd.sort_values("rows", ascending=True)
                bar_c = [
                    COLORS["negative"] if s == "STALE" else COLORS["positive"]
                    for s in counts_sorted["status"]
                ]
                fig_rc = go.Figure(go.Bar(
                    x=counts_sorted["rows"],
                    y=counts_sorted["symbol"],
                    orientation="h",
                    marker=dict(color=bar_c, line=dict(width=0)),
                    text=counts_sorted["rows"],
                    textposition="outside",
                    textfont=dict(color=COLORS["neutral"], size=10),
                    hovertemplate="<b>%{y}</b>: %{x} rows<extra></extra>",
                ))
                apply_theme(fig_rc, title="Rows per Symbol")
                fig_rc.update_layout(
                    height=max(300, 22 * len(counts_sorted)),
                    yaxis=dict(showgrid=False),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_rc, width="stretch", config=PLOTLY_CONFIG)
        else:
            st.warning("No equity data found for this window.")
    else:
        st.info("No equity symbols in storage. Run backfill first.")

    # ── Crypto coverage ────────────────────────────────────────────────────────
    if cr_symbols:
        st.markdown(section_label(f"Crypto Symbol Status — Last {LOOKBACK} Days"), unsafe_allow_html=True)
        cr_bars = load_bars(sorted(cr_symbols)[:9], start_d, end_d, asset_class="crypto")
        if not cr_bars.is_empty():
            cr_counts = (
                cr_bars.group_by("symbol")
                .agg(
                    pl.len().alias("rows"),
                    pl.max("date").alias("latest_date"),
                )
                .sort("symbol")
                .with_columns(
                    pl.when(pl.col("latest_date") < (end_d - timedelta(days=5)))
                    .then(pl.lit("STALE")).otherwise(pl.lit("OK")).alias("status")
                )
            )
            cr_pd = cr_counts.to_pandas()
            st.dataframe(
                cr_pd.style.apply(
                    lambda col: [
                        f"color:{COLORS['negative']}" if v == "STALE"
                        else f"color:{COLORS['positive']}"
                        for v in col
                    ] if col.name == "status" else [""] * len(col),
                    axis=0,
                ),
                width="stretch",
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO STATE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_port:
    if not PORTFOLIO_LOG.exists():
        st.info("No portfolio log yet. Run: `uv run python orchestration/generate_signals.py`")
    else:
        plog = pl.read_parquet(PORTFOLIO_LOG).sort("date")
        if plog.is_empty():
            st.info("Portfolio log is empty.")
        else:
            latest = plog.tail(1).to_dicts()[0]

            # ── Snapshot KPIs ──────────────────────────────────────────────────
            st.markdown(section_label(f"Latest Snapshot · {latest['date']}"), unsafe_allow_html=True)

            passed = latest["pre_trade_passed"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positions",      latest["n_positions"])
            c2.metric("1-Day VaR 95%",  f"{latest['var_1d_95']:.2%}")
            c3.metric("Gross Exposure", f"{latest['gross_exposure']:.1%}")
            with c4:
                st.metric("Pre-trade Check", "")
                st.markdown(
                    badge("PASS", "positive") if passed else badge("FAIL", "negative"),
                    unsafe_allow_html=True,
                )

            # Worst stress
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("1-Day VaR 99%",     f"{latest['var_1d_99']:.2%}")
            c2.metric("Worst Scenario",    str(latest.get("worst_stress_scenario", "N/A")))
            c3.metric("Worst Stress P&L",  f"{latest.get('worst_stress_pnl', 0):.1%}")
            c4.metric("As-of Date",        str(latest["date"]))

            st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)

            # ── Risk trend ──────────────────────────────────────────────────────
            if len(plog) > 2:
                st.markdown(section_label("Risk Metrics Over Time"), unsafe_allow_html=True)
                plog_pd = plog.select(["date", "var_1d_95", "var_1d_99",
                                       "gross_exposure", "n_positions"]).to_pandas()
                plog_pd["date"] = pd.to_datetime(plog_pd["date"])

                fig_trend = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.6, 0.4],
                )
                fig_trend.add_trace(go.Scatter(
                    x=plog_pd["date"], y=plog_pd["var_1d_95"],
                    name="VaR 95%", line=dict(color=COLORS["negative"], width=2),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>VaR 95</extra>",
                ), row=1, col=1)
                fig_trend.add_trace(go.Scatter(
                    x=plog_pd["date"], y=plog_pd["var_1d_99"],
                    name="VaR 99%", line=dict(color=COLORS["orange"], width=1.8, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>VaR 99</extra>",
                ), row=1, col=1)
                fig_trend.add_trace(go.Bar(
                    x=plog_pd["date"], y=plog_pd["gross_exposure"],
                    name="Gross Exp.",
                    marker=dict(color=COLORS["blue"], opacity=0.7, line=dict(width=0)),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.1%}<extra>Gross Exp.</extra>",
                ), row=2, col=1)
                apply_subplot_theme(fig_trend, height=340)
                fig_trend.update_yaxes(tickformat=".2%", row=1, col=1)
                fig_trend.update_yaxes(tickformat=".0%", row=2, col=1)
                fig_trend.update_layout(hovermode="x unified", showlegend=True)
                st.plotly_chart(fig_trend, width="stretch", config=PLOTLY_CONFIG)

            # ── Current weights bar ────────────────────────────────────────────
            if TARGET_WEIGHTS.exists():
                tw = pl.read_parquet(TARGET_WEIGHTS)
                if not tw.is_empty():
                    last_tw_date = tw["date"].max()
                    last_tw = tw.filter(pl.col("date") == last_tw_date)
                    syms_tw = last_tw["symbol"].to_list()
                    wts_tw  = last_tw["weight"].to_list()

                    st.markdown(section_label(f"Current Target Weights · {last_tw_date}"), unsafe_allow_html=True)
                    fig_wts = go.Figure(go.Bar(
                        x=wts_tw, y=syms_tw,
                        orientation="h",
                        marker=dict(color=COLORS["teal"], line=dict(width=0), opacity=0.85),
                        text=[f"{w:.1%}" for w in wts_tw],
                        textposition="outside",
                        textfont=dict(color=COLORS["neutral"], size=11),
                        hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                    ))
                    apply_theme(fig_wts)
                    fig_wts.update_layout(
                        height=max(140, 34 * len(syms_tw)),
                        xaxis=dict(tickformat=".0%", showgrid=False),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_wts, width="stretch", config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOGS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_logs:
    st.markdown(section_label("Log Viewer"), unsafe_allow_html=True)

    sub_pipe, sub_ingest, sub_signals = st.tabs(["Pipeline", "Ingest", "Signals"])

    for sub_tab, log_path, log_name in [
        (sub_pipe,    PIPELINE_LOG, "pipeline.log"),
        (sub_ingest,  INGEST_LOG,   "ingest.log"),
        (sub_signals, SIGNALS_LOG,  "signals.log"),
    ]:
        with sub_tab:
            lines  = _tail_log(log_path)
            errors = _count_errors(lines)

            col_stat, col_meta = st.columns([2, 3])
            with col_stat:
                if errors:
                    st.markdown(
                        f'<div style="display:inline-block;">'
                        + badge(f"{errors} ERROR{'S' if errors > 1 else ''}", "negative")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="display:inline-block;">'
                        + badge("NO ERRORS", "positive")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            with col_meta:
                exists_str = "exists" if log_path.exists() else "not found"
                size_str   = (
                    f"{log_path.stat().st_size / 1024:.1f} KB"
                    if log_path.exists() else ""
                )
                st.caption(f"{log_name} · {exists_str} · {size_str} · {len(lines)} lines shown")

            if lines:
                # Colorize error lines in display
                colorized = []
                for line in lines:
                    if " ERROR " in line or "FAILED" in line:
                        colorized.append(f"\u001b[31m{line}\u001b[0m")  # ANSI red
                    elif " WARNING " in line or "ALERT" in line:
                        colorized.append(f"\u001b[33m{line}\u001b[0m")  # ANSI yellow
                    else:
                        colorized.append(line)

                with st.expander(f"Show {log_name} ({len(lines)} lines)", expanded=errors > 0):
                    st.code("\n".join(lines), language=None)
