"""Pipeline health dashboard — Streamlit app (Dashboard #1).

Tabs: Status · Data Quality · Logs
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import plotly.graph_objects as go
import streamlit as st
import json

from config.settings import DATA_DIR, LOGS_DIR, PROJECT_ROOT
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme,
    kpi_card, section_label, badge, page_header, status_banner,
)
from storage.parquet_store import list_symbols, load_bars

st.markdown(CSS, unsafe_allow_html=True)

LOOKBACK        = 30
TARGET_WEIGHTS  = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
PORTFOLIO_LOG   = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"
PIPELINE_LOG    = LOGS_DIR / "pipeline.log"
INGEST_LOG      = LOGS_DIR / "ingest.log"
SIGNALS_LOG     = LOGS_DIR / "signals.log"
_HEARTBEAT_PATH = PROJECT_ROOT / ".pipeline_heartbeat.json"


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
    if _HEARTBEAT_PATH.exists():
        try:
            hb = json.loads(_HEARTBEAT_PATH.read_text(encoding="utf-8"))
            ts = hb.get("ts_utc", "unknown")
            ok = hb.get("status") == "ok"
            try:
                dt    = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                age_h = (datetime.now() - dt).total_seconds() / 3600
            except Exception:
                age_h = float("inf")
            return ts, age_h, ok
        except Exception:
            pass

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


def _tail_log(path: Path, n: int = 200) -> list[str]:
    if not path.exists():
        return ["Log file not found"]
    lines = path.read_text(errors="replace").splitlines()
    return lines[-n:]


def _count_errors(lines: list[str]) -> int:
    return sum(1 for l in lines if " ERROR " in l or "FAILED" in l or "ALERT" in l)


def _parse_log_events(path: Path, n: int = 400) -> list[dict]:
    """Return structured error/warning events from the last n lines of a log."""
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()[-n:]
    events = []
    for line in lines:
        if " ERROR " in line or "FAILED" in line or "ALERT" in line:
            level = "ERROR"
        elif " WARNING " in line or "WARN" in line:
            level = "WARNING"
        else:
            continue
        parts  = line.split()
        ts_str = next((p for p in parts[:3] if ":" in p and len(p) >= 5), "")
        events.append({"level": level, "ts": ts_str, "message": line})
    return events


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

tab_status, tab_data, tab_logs = st.tabs(
    ["  Status  ", "  Data Quality  ", "  Logs  "]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATUS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_status:
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

    # ── Pipeline Timeline ─────────────────────────────────────────────────────
    st.markdown(section_label("Pipeline Timeline"), unsafe_allow_html=True)

    _now = datetime.now()
    _timeline_data = [
        ("Equity Ingest",   eq_ts,   eq_age,   eq_age  < 30),
        ("Crypto Ingest",   cr_ts,   cr_age,   cr_age  < 30),
        ("Signal Generate", sig_ts,  sig_age,  sig_fresh),
        ("Pipeline Run",    pipe_ts, pipe_age, pipe_ok),
    ]
    _UNKNOWN_STRINGS = {"No data", "No files", "No log file", "No signals file",
                        "No complete marker", "Log read error", "No log", "unknown"}
    fig_tl = go.Figure()
    for lbl, ts_str, age_h, ok in _timeline_data:
        if age_h == float("inf") or ts_str in _UNKNOWN_STRINGS:
            continue
        try:
            end_dt   = _now - timedelta(hours=age_h)
            start_dt = end_dt - timedelta(minutes=5)
            color    = COLORS["positive"] if ok else COLORS["warning"]
            fig_tl.add_trace(go.Bar(
                x=[(end_dt - start_dt).total_seconds() / 3600],
                y=[lbl],
                base=[(start_dt - _now).total_seconds() / 3600],
                orientation="h",
                marker=dict(color=color, line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"<b>{lbl}</b><br>Last run: {ts_str}<br>{age_h:.1f}h ago<extra></extra>",
            ))
        except Exception:
            pass

    fig_tl.add_vline(x=0,   line=dict(color=COLORS["positive"], width=2),
                     annotation_text="Now",
                     annotation_font=dict(color=COLORS["positive"], size=10),
                     annotation_position="top")
    fig_tl.add_vline(x=-24, line=dict(color=COLORS["warning"], width=1, dash="dot"),
                     annotation_text="24h",
                     annotation_font=dict(color=COLORS["warning"], size=9),
                     annotation_position="top")
    fig_tl.add_vline(x=-48, line=dict(color=COLORS["negative"], width=1, dash="dot"),
                     annotation_text="48h",
                     annotation_font=dict(color=COLORS["negative"], size=9),
                     annotation_position="top")
    apply_theme(fig_tl)
    fig_tl.update_layout(
        height=170,
        xaxis=dict(title="Hours ago", range=[-72, 1], showgrid=False,
                   tickvals=[-72, -48, -24, -12, -6, 0],
                   ticktext=["72h", "48h", "24h", "12h", "6h", "Now"]),
        yaxis=dict(showgrid=False),
        showlegend=False,
        barmode="overlay",
    )
    st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CONFIG)

    # ── Data freshness visual ──────────────────────────────────────────────────
    st.markdown(section_label("Data Freshness"), unsafe_allow_html=True)

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
    age_text = [f"{a:.1f}h" if a < float("inf") else "N/A" for a in ages_raw]

    fig_fresh = go.Figure(go.Bar(
        x=ages_disp, y=sources, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=age_text, textposition="outside",
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
    c1.markdown(kpi_card("Daily Pipeline",  "06:15 Mon–Fri",  accent=COLORS["blue"]),   unsafe_allow_html=True)
    c2.markdown(kpi_card("Daily Rebalance", "16:30 Mon–Fri",  accent=COLORS["purple"]), unsafe_allow_html=True)
    c3.markdown(kpi_card("Scheduler",       "Task Scheduler", accent=COLORS["neutral"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    eq_symbols = list_symbols("equity")
    cr_symbols = list_symbols("crypto")

    st.markdown(section_label("Universe Coverage"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity Symbols", len(eq_symbols))
    c2.metric("Crypto Symbols", len(cr_symbols))
    c3.metric("Total Symbols",  len(eq_symbols) + len(cr_symbols))

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

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
                    width="stretch", hide_index=True, height=420,
                )

            with col_chart:
                counts_sorted = counts_pd.sort_values("rows", ascending=True)
                bar_c = [
                    COLORS["negative"] if s == "STALE" else COLORS["positive"]
                    for s in counts_sorted["status"]
                ]
                fig_rc = go.Figure(go.Bar(
                    x=counts_sorted["rows"], y=counts_sorted["symbol"], orientation="h",
                    marker=dict(color=bar_c, line=dict(width=0)),
                    text=counts_sorted["rows"], textposition="outside",
                    textfont=dict(color=COLORS["neutral"], size=10),
                    hovertemplate="<b>%{y}</b>: %{x} rows<extra></extra>",
                ))
                apply_theme(fig_rc, title="Rows per Symbol")
                fig_rc.update_layout(
                    height=max(300, 22 * len(counts_sorted)),
                    yaxis=dict(showgrid=False), xaxis=dict(showgrid=False), showlegend=False,
                )
                st.plotly_chart(fig_rc, width="stretch", config=PLOTLY_CONFIG)
        else:
            st.warning("No equity data found for this window.")
    else:
        st.info("No equity symbols in storage. Run backfill first.")

    if cr_symbols:
        st.markdown(section_label(f"Crypto Symbol Status — Last {LOOKBACK} Days"), unsafe_allow_html=True)
        cr_bars = load_bars(sorted(cr_symbols)[:9], start_d, end_d, asset_class="crypto")
        if not cr_bars.is_empty():
            cr_counts = (
                cr_bars.group_by("symbol")
                .agg(pl.len().alias("rows"), pl.max("date").alias("latest_date"))
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
                width="stretch", hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOGS  (structured error/warning event feed + raw expanders)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_logs:
    st.markdown(section_label("Event Feed"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;margin-bottom:10px;">'
        "Filtered errors and warnings across all pipeline components. "
        "Expand the Raw Logs section below for full output."
        "</div>", unsafe_allow_html=True,
    )

    all_events: list[dict] = []
    for _lp, _ln in [(PIPELINE_LOG, "pipeline"), (INGEST_LOG, "ingest"), (SIGNALS_LOG, "signals")]:
        for ev in _parse_log_events(_lp):
            ev["component"] = _ln
            all_events.append(ev)

    errors_only   = [e for e in all_events if e["level"] == "ERROR"]
    warnings_only = [e for e in all_events if e["level"] == "WARNING"]

    kc1, kc2, kc3 = st.columns(3)
    kc1.markdown(kpi_card("Errors",   str(len(errors_only)),
                           accent=COLORS["negative"] if errors_only else COLORS["positive"]),
                 unsafe_allow_html=True)
    kc2.markdown(kpi_card("Warnings", str(len(warnings_only)),
                           accent=COLORS["warning"] if warnings_only else COLORS["neutral"]),
                 unsafe_allow_html=True)
    kc3.markdown(kpi_card("Components scanned", "3", accent=COLORS["blue"]),
                 unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)

    def _event_card(ev: dict, border_color: str) -> str:
        return (
            f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {border_color};border-radius:0 6px 6px 0;'
            f'padding:6px 12px;margin:3px 0;font-family:monospace;font-size:0.75rem;'
            f'color:{COLORS["text"]};">'
            f'<span style="color:{COLORS["text_muted"]};">[{ev["component"]}]</span> '
            f'{ev["message"]}</div>'
        )

    if errors_only:
        st.markdown(
            f'<div style="background:{COLORS["negative"]}14;border-left:3px solid {COLORS["negative"]};'
            f'border-radius:0 6px 6px 0;padding:8px 14px;margin:6px 0;">'
            f'<span style="color:{COLORS["negative"]};font-weight:700;font-size:0.72rem;'
            f'text-transform:uppercase;letter-spacing:0.08em;">Errors ({len(errors_only)})</span></div>',
            unsafe_allow_html=True,
        )
        for ev in errors_only[-20:]:
            st.markdown(_event_card(ev, COLORS["negative"]), unsafe_allow_html=True)

    if warnings_only:
        st.markdown(
            f'<div style="background:{COLORS["warning"]}14;border-left:3px solid {COLORS["warning"]};'
            f'border-radius:0 6px 6px 0;padding:8px 14px;margin:6px 0;">'
            f'<span style="color:{COLORS["warning"]};font-weight:700;font-size:0.72rem;'
            f'text-transform:uppercase;letter-spacing:0.08em;">Warnings ({len(warnings_only)})</span></div>',
            unsafe_allow_html=True,
        )
        for ev in warnings_only[-10:]:
            st.markdown(_event_card(ev, COLORS["warning"]), unsafe_allow_html=True)

    if not errors_only and not warnings_only:
        st.markdown(badge("NO ERRORS OR WARNINGS", "positive"), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Raw Logs"), unsafe_allow_html=True)

    sub_pipe, sub_ingest, sub_signals = st.tabs(["Pipeline", "Ingest", "Signals"])
    for sub_tab, log_path, log_name in [
        (sub_pipe,    PIPELINE_LOG, "pipeline.log"),
        (sub_ingest,  INGEST_LOG,   "ingest.log"),
        (sub_signals, SIGNALS_LOG,  "signals.log"),
    ]:
        with sub_tab:
            lines      = _tail_log(log_path)
            exists_str = "exists" if log_path.exists() else "not found"
            size_str   = f"{log_path.stat().st_size / 1024:.1f} KB" if log_path.exists() else ""
            n_errors   = _count_errors(lines)
            st.caption(f"{log_name} · {exists_str} · {size_str} · {n_errors} error lines")
            with st.expander(f"{log_name} ({len(lines)} lines shown)", expanded=n_errors > 0):
                st.code("\n".join(lines), language=None)
