"""Pipeline health dashboard — Streamlit app (Dashboard #1).

Tabs: Status · Data Quality · Logs
"""

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
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


def _log_rotation_warning(path: Path) -> str | None:
    """Return a warning string if the log appears to have been recently rotated."""
    if not path.exists():
        return None
    size_kb = path.stat().st_size / 1024
    age_min = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 60
    if size_kb < 5 and age_min < 120:
        return f"Log may have been rotated recently (file is {size_kb:.1f} KB, modified {age_min:.0f} min ago)"
    # Check if oldest visible line is recent
    try:
        first = path.read_text(errors="replace").splitlines()[0]
        # Try to parse a date from the first line
        parts = first.split()
        if len(parts) >= 2:
            try:
                dt = datetime.fromisoformat(f"{parts[0]} {parts[1][:8]}")
                age_h = (datetime.now() - dt).total_seconds() / 3600
                if age_h < 6:
                    return f"Log history starts {age_h:.1f}h ago — older entries may be in a rotated archive"
            except Exception:
                pass
    except Exception:
        pass
    return None


def _parse_log_events(path: Path, n: int = 2000) -> list[dict]:
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
        ts_str = ""
        ts_dt  = None
        if len(parts) >= 2:
            try:
                ts_dt  = datetime.fromisoformat(f"{parts[0]} {parts[1][:8]}")
                ts_str = ts_dt.strftime("%m-%d %H:%M")
            except Exception:
                ts_str = parts[0] if parts else ""
        events.append({
            "level":   level,
            "ts":      ts_str,
            "ts_dt":   ts_dt,
            "message": line,
        })
    return events


# Known symbols for Data Quality cross-linking
_ALL_KNOWN_SYMBOLS: set[str] = set()
try:
    from config.universes import EQUITY_UNIVERSE, CRYPTO_UNIVERSE
    _ALL_KNOWN_SYMBOLS = set(EQUITY_UNIVERSE) | {s.replace("/", "_") for s in CRYPTO_UNIVERSE}
except Exception:
    pass


def _extract_symbols(message: str) -> list[str]:
    """Find known ticker symbols mentioned in a log line."""
    import re
    words = set(re.findall(r"\b[A-Z0-9_]{2,10}\b", message))
    return sorted(words & _ALL_KNOWN_SYMBOLS)


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
        "Pipeline Health",
        "Verify data is flowing, fresh, and error-free before any research or trading decision.",
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

    # ── Emergency Kill-Switch ─────────────────────────────────────────────────
    _halt_path   = PROJECT_ROOT / "QP_HALT"
    _halt_active = _halt_path.exists()
    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    _hc1, _hc2 = st.columns([1, 3])
    with _hc1:
        if _halt_active:
            if st.button("✅ Resume Pipeline", type="secondary", use_container_width=True,
                         help="Remove QP_HALT file — pipeline resumes on next scheduled run"):
                _halt_path.unlink(missing_ok=True)
                st.success("Kill-switch cleared. Pipeline will run normally at next scheduled time.")
                st.rerun()
        else:
            if st.button("⛔ HALT Pipeline", type="primary", use_container_width=True,
                         help="Create QP_HALT file — pipeline stops at next entry point check"):
                _halt_path.touch()
                st.warning("Kill-switch ACTIVE — pipeline will not ingest, generate signals, or rebalance.")
                st.rerun()
    with _hc2:
        if _halt_active:
            st.markdown(
                f'<div style="background:rgba(255,77,77,0.12);border:1px solid {COLORS["negative"]};'
                f'border-radius:6px;padding:9px 14px;font-size:0.82rem;">'
                f'<span style="color:{COLORS["negative"]};font-weight:700;">⛔ KILL-SWITCH ACTIVE</span>'
                f'<span style="color:{COLORS["neutral"]};"> — all pipeline entry points will exit immediately.'
                f' Remove <code>QP_HALT</code> from the project root to resume.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:rgba(0,212,170,0.06);border:1px solid {COLORS["border"]};'
                f'border-radius:6px;padding:9px 14px;font-size:0.82rem;">'
                f'<span style="color:{COLORS["text_muted"]};">Pipeline running normally. '
                f'Press <b>⛔ HALT Pipeline</b> to stop all future ingest, signal, and rebalance runs '
                f'without touching any data files.</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)

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


# ── Data Quality helpers ───────────────────────────────────────────────────────

def _availability_heatmap(df: pl.DataFrame, symbols: list[str],
                           start_d: date, end_d: date, title: str):
    """Build a date×symbol availability heatmap (1=present, 0=missing)."""
    all_dates = pd.date_range(start_d, end_d, freq="B")  # business days only
    present = (
        df.select(["date", "symbol"])
        .with_columns(pl.col("date").cast(pl.Date))
        .to_pandas()
        .assign(present=1)
        .pivot_table(index="date", columns="symbol", values="present", fill_value=0)
    )
    present.index = pd.to_datetime(present.index)
    present = present.reindex(all_dates, fill_value=0)
    present = present.reindex(columns=sorted(symbols), fill_value=0)

    z      = present.values.T.tolist()
    x_labs = [d.strftime("%m/%d") for d in present.index]
    y_labs = list(present.columns)

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labs, y=y_labs,
        colorscale=[[0, COLORS["negative"]], [1, COLORS["green"]]],
        showscale=False,
        hovertemplate="<b>%{y}</b> on %{x}: %{z}<extra></extra>",
        xgap=1, ygap=1,
    ))
    apply_theme(fig, title=title)
    fig.update_layout(
        height=max(220, 18 * len(y_labs)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9)),
    )
    return fig


def _schema_validation(df: pl.DataFrame, spike_threshold: float = 0.15) -> pd.DataFrame:
    """Per-symbol schema checks: nulls, zero prices, and single-day spikes."""
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    results = []
    for sym, grp in df.sort("date").partition_by("symbol", as_dict=True).items():
        prices = grp[price_col].drop_nulls()
        n      = len(grp)
        n_null = grp[price_col].null_count()
        n_zero = (grp[price_col].fill_null(0) <= 0).sum()
        rets   = prices.pct_change().drop_nulls()
        n_spike = (rets.abs() > spike_threshold).sum()
        issues = []
        if n_null > 0:
            issues.append(f"{n_null} null prices")
        if n_zero > 0:
            issues.append(f"{n_zero} zero/neg prices")
        if n_spike > 0:
            issues.append(f"{n_spike} day(s) >{spike_threshold*100:.0f}% move")
        results.append({
            "symbol":    sym,
            "rows":      n,
            "null_close":n_null,
            "zero_price":n_zero,
            "spike_days":int(n_spike),
            "verdict":   "⚠ Issues" if issues else "✓ Clean",
            "details":   ", ".join(issues) if issues else "—",
        })
    return pd.DataFrame(results).sort_values("symbol")


def _trigger_reingest(asset_class: str, days_back: int = 30) -> str:
    """Run targeted backfill for the last N days of an asset class."""
    start_str = (date.today() - timedelta(days=days_back)).isoformat()
    script    = PROJECT_ROOT / "orchestration" / "backfill_history.py"
    cmd = [sys.executable, str(script),
           "--asset-class", asset_class, "--start", start_str]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=300, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return f"✓ Re-ingest complete ({asset_class}, last {days_back} days)"
        return f"✗ Re-ingest failed:\n{result.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return "✗ Timed out after 5 minutes"
    except Exception as exc:
        return f"✗ Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    eq_symbols = list_symbols("equity")
    cr_symbols = list_symbols("crypto")
    end_d      = date.today()
    start_d    = end_d - timedelta(days=LOOKBACK)

    # ── Universe Coverage ──────────────────────────────────────────────────────
    st.markdown(section_label("Universe Coverage"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity Symbols", len(eq_symbols))
    c2.metric("Crypto Symbols", len(cr_symbols))
    c3.metric("Total Symbols",  len(eq_symbols) + len(cr_symbols))
    st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)

    # ── Equity ─────────────────────────────────────────────────────────────────
    if eq_symbols:
        sample   = sorted(eq_symbols)[:26]
        df_bars  = load_bars(sample, start_d, end_d, asset_class="equity")

        if not df_bars.is_empty():
            # Status table
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
                    .then(pl.lit("STALE")).otherwise(pl.lit("OK")).alias("status")
                )
            )
            counts_pd = counts.to_pandas()
            stale_eq  = counts_pd[counts_pd["status"] == "STALE"]["symbol"].tolist()

            if stale_eq:
                st.warning(f"{len(stale_eq)} equity symbol(s) stale: {', '.join(stale_eq)}")
            else:
                st.success("All equity symbols have fresh data")

            # ── 1. Historical availability heatmap ────────────────────────────
            st.markdown(section_label(f"Equity Data Availability — Last {LOOKBACK} Days"),
                        unsafe_allow_html=True)
            st.plotly_chart(
                _availability_heatmap(df_bars, sample, start_d, end_d,
                                      "Green = data present · Red = missing"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

            # ── 2. Schema validation ──────────────────────────────────────────
            st.markdown(section_label("Equity Schema Validation (>15% daily move = spike)"),
                        unsafe_allow_html=True)
            schema_eq = _schema_validation(df_bars, spike_threshold=0.15)
            issues_eq = schema_eq[schema_eq["verdict"] != "✓ Clean"]
            if issues_eq.empty:
                st.success("No schema issues detected in equity data")
            else:
                st.warning(f"{len(issues_eq)} symbol(s) with data quality issues")

            st.dataframe(
                schema_eq.style.apply(
                    lambda col: [
                        f"color:{COLORS['negative']};font-weight:600"
                        if v == "⚠ Issues" else f"color:{COLORS['positive']}"
                        for v in col
                    ] if col.name == "verdict" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True, hide_index=True,
            )

            # ── Row counts chart (existing, kept) ─────────────────────────────
            st.markdown(section_label("Row Counts per Symbol"), unsafe_allow_html=True)
            counts_sorted = counts_pd.sort_values("rows", ascending=True)
            bar_c = [COLORS["negative"] if s == "STALE" else COLORS["positive"]
                     for s in counts_sorted["status"]]
            fig_rc = go.Figure(go.Bar(
                x=counts_sorted["rows"], y=counts_sorted["symbol"], orientation="h",
                marker=dict(color=bar_c, line=dict(width=0)),
                text=counts_sorted["rows"], textposition="outside",
                textfont=dict(color=COLORS["neutral"], size=10),
                hovertemplate="<b>%{y}</b>: %{x} rows<extra></extra>",
            ))
            apply_theme(fig_rc, title="")
            fig_rc.update_layout(
                height=max(300, 22 * len(counts_sorted)),
                yaxis=dict(showgrid=False), xaxis=dict(showgrid=False), showlegend=False,
            )
            st.plotly_chart(fig_rc, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.warning("No equity data found for this window.")
    else:
        st.info("No equity symbols in storage. Run backfill first.")

    # ── Crypto ─────────────────────────────────────────────────────────────────
    if cr_symbols:
        cr_sample = sorted(cr_symbols)[:10]
        cr_bars   = load_bars(cr_sample, start_d, end_d, asset_class="crypto")

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
            cr_pd      = cr_counts.to_pandas()
            stale_cr   = cr_pd[cr_pd["status"] == "STALE"]["symbol"].tolist()

            st.markdown(section_label(f"Crypto Data Availability — Last {LOOKBACK} Days"),
                        unsafe_allow_html=True)
            if stale_cr:
                st.warning(f"{len(stale_cr)} crypto symbol(s) stale: {', '.join(stale_cr)}")
            else:
                st.success("All crypto symbols have fresh data")

            st.plotly_chart(
                _availability_heatmap(cr_bars, cr_sample, start_d, end_d,
                                      "Green = data present · Red = missing"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

            st.markdown(section_label("Crypto Schema Validation (>30% daily move = spike)"),
                        unsafe_allow_html=True)
            schema_cr = _schema_validation(cr_bars, spike_threshold=0.30)
            issues_cr = schema_cr[schema_cr["verdict"] != "✓ Clean"]
            if issues_cr.empty:
                st.success("No schema issues detected in crypto data")
            else:
                st.warning(f"{len(issues_cr)} crypto symbol(s) with data quality issues")
            st.dataframe(schema_cr, use_container_width=True, hide_index=True)

    # ── 3. Re-ingest trigger ───────────────────────────────────────────────────
    st.markdown(section_label("Re-Ingest Stale Data"), unsafe_allow_html=True)
    st.caption(
        "Runs a targeted backfill for the last 30 days of the selected asset class. "
        "All symbols in that class are refreshed — this takes 1–3 minutes."
    )

    all_stale = []
    if eq_symbols and "stale_eq" in dir():
        all_stale += [(s, "equity") for s in stale_eq]
    if cr_symbols and "stale_cr" in dir():
        all_stale += [(s, "crypto") for s in stale_cr]

    ri_col1, ri_col2 = st.columns([2, 1])
    with ri_col1:
        if all_stale:
            selected = st.selectbox(
                "Stale symbol to re-ingest",
                options=[f"{s} ({ac})" for s, ac in all_stale],
                key="dq_reingest_select",
            )
            target_ac = "equity" if "(equity)" in selected else "crypto"
        else:
            st.selectbox("Stale symbol to re-ingest",
                         options=["— all symbols fresh —"],
                         disabled=True, key="dq_reingest_select")
            target_ac = None

    with ri_col2:
        st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
        trigger = st.button(
            "▶ Re-Ingest",
            disabled=(target_ac is None),
            key="dq_reingest_btn",
            use_container_width=True,
        )

    if trigger and target_ac:
        with st.spinner(f"Re-ingesting {target_ac} data for last 30 days…"):
            msg = _trigger_reingest(target_ac, days_back=30)
        if msg.startswith("✓"):
            st.success(msg)
            st.cache_data.clear()
        else:
            st.error(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOGS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_logs:

    # ── Filter controls ────────────────────────────────────────────────────────
    lc1, lc2, lc3, lc4 = st.columns([2.5, 1.5, 1.5, 1.5])
    with lc1:
        log_search = st.text_input(
            "Search", placeholder="symbol, error keyword…",
            key="log_search", label_visibility="collapsed",
        )
        st.caption("🔍 Search events")
    with lc2:
        sev_filter = st.selectbox("Severity", ["All", "ERROR only", "WARNING only"],
                                  key="log_sev")
    with lc3:
        comp_filter = st.multiselect(
            "Component", ["pipeline", "ingest", "signals"],
            default=["pipeline", "ingest", "signals"],
            key="log_comp",
        )
    with lc4:
        window_filter = st.selectbox(
            "Time window", ["Last 24 h", "Last 48 h", "Last 7 d", "All time"],
            key="log_window",
        )

    # ── Parse & filter events ──────────────────────────────────────────────────
    _window_h = {"Last 24 h": 24, "Last 48 h": 48, "Last 7 d": 168, "All time": None}
    cutoff_h  = _window_h[window_filter]
    cutoff_dt = datetime.now() - timedelta(hours=cutoff_h) if cutoff_h else None

    all_events: list[dict] = []
    for _lp, _ln in [(PIPELINE_LOG, "pipeline"), (INGEST_LOG, "ingest"), (SIGNALS_LOG, "signals")]:
        if _ln not in comp_filter:
            continue
        for ev in _parse_log_events(_lp):
            ev["component"] = _ln
            # date filter
            if cutoff_dt and ev["ts_dt"] and ev["ts_dt"] < cutoff_dt:
                continue
            # severity filter
            if sev_filter == "ERROR only"   and ev["level"] != "ERROR":
                continue
            if sev_filter == "WARNING only" and ev["level"] != "WARNING":
                continue
            # search filter
            if log_search and log_search.lower() not in ev["message"].lower():
                continue
            all_events.append(ev)

    errors_f   = [e for e in all_events if e["level"] == "ERROR"]
    warnings_f = [e for e in all_events if e["level"] == "WARNING"]

    # ── KPI cards ──────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(kpi_card("Errors",   str(len(errors_f)),
                           accent=COLORS["negative"] if errors_f else COLORS["positive"]),
                 unsafe_allow_html=True)
    kc2.markdown(kpi_card("Warnings", str(len(warnings_f)),
                           accent=COLORS["warning"] if warnings_f else COLORS["neutral"]),
                 unsafe_allow_html=True)
    kc3.markdown(kpi_card("Total events", str(len(all_events)), accent=COLORS["blue"]),
                 unsafe_allow_html=True)
    kc4.markdown(kpi_card("Window", window_filter.replace(" h", "h").replace(" d", "d"),
                           accent=COLORS["gold"]),
                 unsafe_allow_html=True)

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    # ── Event feed ─────────────────────────────────────────────────────────────
    st.markdown(section_label("Event Feed"), unsafe_allow_html=True)

    if not all_events:
        if log_search or sev_filter != "All" or window_filter != "All time":
            st.info("No events match the current filters.")
        else:
            st.markdown(badge("NO ERRORS OR WARNINGS", "positive"), unsafe_allow_html=True)
    else:
        # Group by level, newest first
        for group_label, group_events, border_color in [
            (f"Errors ({len(errors_f)})",   errors_f,   COLORS["negative"]),
            (f"Warnings ({len(warnings_f)})", warnings_f, COLORS["warning"]),
        ]:
            if not group_events:
                continue
            st.markdown(
                f'<div style="background:{border_color}14;border-left:3px solid {border_color};'
                f'border-radius:0 6px 6px 0;padding:7px 14px;margin:10px 0 4px;">'
                f'<span style="color:{border_color};font-weight:700;font-size:0.72rem;'
                f'text-transform:uppercase;letter-spacing:0.08em;">{group_label}</span></div>',
                unsafe_allow_html=True,
            )
            for ev in group_events[-30:]:
                syms = _extract_symbols(ev["message"])
                sym_badges = " ".join(
                    f'<span style="background:rgba(201,162,39,0.15);color:{COLORS["gold"]};'
                    f'font-size:0.68rem;font-weight:700;padding:1px 6px;border-radius:3px;'
                    f'border:1px solid rgba(201,162,39,0.3);margin-left:4px;" '
                    f'title="Check Data Quality tab for {s}">{s} ↗ DQ</span>'
                    for s in syms
                )
                ts_html = (
                    f'<span style="color:{COLORS["text_muted"]};margin-right:8px;">{ev["ts"]}</span>'
                    if ev["ts"] else ""
                )
                comp_html = (
                    f'<span style="color:{COLORS["purple"]};margin-right:6px;">'
                    f'[{ev["component"]}]</span>'
                )
                st.markdown(
                    f'<div style="background:{COLORS["card_bg"]};'
                    f'border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {border_color};'
                    f'border-radius:0 6px 6px 0;'
                    f'padding:6px 12px;margin:2px 0;font-family:monospace;font-size:0.74rem;'
                    f'color:{COLORS["text"]};word-break:break-all;">'
                    f'{ts_html}{comp_html}{ev["message"]}{sym_badges}</div>',
                    unsafe_allow_html=True,
                )

    # ── If symbols found across all errors, show DQ hint ──────────────────────
    all_syms_in_errors = set()
    for ev in errors_f:
        all_syms_in_errors.update(_extract_symbols(ev["message"]))
    if all_syms_in_errors:
        st.markdown(
            f'<div style="background:{COLORS["gold_bg"]};border:1px solid rgba(201,162,39,0.3);'
            f'border-radius:8px;padding:10px 14px;margin:10px 0;font-size:0.78rem;">'
            f'<span style="color:{COLORS["gold"]};font-weight:700;">↗ Data Quality hint:</span> '
            f'<span style="color:{COLORS["neutral"]};">Errors mention '
            f'<b>{", ".join(sorted(all_syms_in_errors))}</b>. '
            f'Open the <b>Data Quality</b> tab to check freshness and schema for these symbols.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Dead-Letter Log ────────────────────────────────────────────────────────
    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Dead-Letter Log (Failed Data Downloads)"), unsafe_allow_html=True)
    st.caption("Symbols that failed to download after all retries. Tab-separated: timestamp · adapter · symbol · reason.")

    _DEAD_LETTERS = LOGS_DIR / "dead_letters.log"
    if not _DEAD_LETTERS.exists():
        st.markdown(badge("NO DEAD LETTERS", "positive"), unsafe_allow_html=True)
        st.caption("logs/dead_letters.log not found — no download failures recorded.")
    else:
        _dl_raw = [l for l in _DEAD_LETTERS.read_text(errors="replace").splitlines() if l.strip()]
        if not _dl_raw:
            st.markdown(badge("NO DEAD LETTERS", "positive"), unsafe_allow_html=True)
        else:
            # Apply search filter if active
            _dl_filtered = [l for l in _dl_raw if not log_search or log_search.lower() in l.lower()]
            _dl_rows = []
            for _line in _dl_filtered[-200:]:
                _p = _line.split("\t")
                _dl_rows.append({
                    "Timestamp": _p[0] if len(_p) > 0 else "—",
                    "Adapter":   _p[1] if len(_p) > 1 else "—",
                    "Symbol":    _p[2] if len(_p) > 2 else "—",
                    "Reason":    _p[3] if len(_p) > 3 else _line,
                })
            _dl_df = pd.DataFrame(_dl_rows)
            _n_unique = _dl_df["Symbol"].nunique()
            _dl_size  = _DEAD_LETTERS.stat().st_size / 1024

            dl_k1, dl_k2, dl_k3, dl_k4 = st.columns(4)
            dl_k1.markdown(kpi_card("Dead Letters", str(len(_dl_raw)),
                                     accent=COLORS["negative"]),        unsafe_allow_html=True)
            dl_k2.markdown(kpi_card("Unique Symbols", str(_n_unique),
                                     accent=COLORS["warning"]),         unsafe_allow_html=True)
            dl_k3.markdown(kpi_card("Adapters", str(_dl_df["Adapter"].nunique()),
                                     accent=COLORS["blue"]),            unsafe_allow_html=True)
            dl_k4.markdown(kpi_card("Log Size", f"{_dl_size:.1f} KB",
                                     accent=COLORS["neutral"]),         unsafe_allow_html=True)

            with st.expander(
                f"Dead-Letter Entries — {len(_dl_raw)} total, showing last {min(len(_dl_rows), 200)}",
                expanded=True,
            ):
                st.dataframe(_dl_df, use_container_width=True, hide_index=True,
                              height=min(400, 38 * len(_dl_rows) + 40))

            _clr_col, _ = st.columns([1, 4])
            with _clr_col:
                if st.button("🗑 Clear Dead-Letter Log", key="clear_dl", type="secondary",
                             use_container_width=True,
                             help="Truncates the log — symbols will be retried on next pipeline run"):
                    _DEAD_LETTERS.write_text("")
                    st.success("Dead-letter log cleared.")
                    st.rerun()

    # ── Raw logs (bounded + rotation-aware) ────────────────────────────────────
    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Raw Logs"), unsafe_allow_html=True)

    raw_lines_opt = st.select_slider(
        "Lines to show", options=[50, 100, 200, 500, 1000], value=200,
        key="log_raw_lines",
    )

    sub_pipe, sub_ingest, sub_signals = st.tabs(["Pipeline", "Ingest", "Signals"])
    for sub_tab, log_path, log_name in [
        (sub_pipe,    PIPELINE_LOG, "pipeline.log"),
        (sub_ingest,  INGEST_LOG,   "ingest.log"),
        (sub_signals, SIGNALS_LOG,  "signals.log"),
    ]:
        with sub_tab:
            # Rotation warning
            rot_warn = _log_rotation_warning(log_path)
            if rot_warn:
                st.warning(f"⚠ {rot_warn}", icon="🔄")

            lines      = _tail_log(log_path, n=raw_lines_opt)
            exists_str = "exists" if log_path.exists() else "not found"
            size_str   = (
                f"{log_path.stat().st_size / 1024:.1f} KB"
                if log_path.exists() else "—"
            )
            total_lines = (
                len(log_path.read_text(errors="replace").splitlines())
                if log_path.exists() else 0
            )
            n_errors = _count_errors(lines)

            st.caption(
                f"{log_name} · {exists_str} · {size_str} · "
                f"{total_lines:,} total lines · showing last {len(lines)} · "
                f"{n_errors} error lines in view"
            )

            # Apply search filter to raw view too
            if log_search:
                lines = [l for l in lines if log_search.lower() in l.lower()]
                st.caption(f"🔍 Filtered to {len(lines)} matching lines")

            with st.expander(
                f"{log_name} — last {len(lines)} lines",
                expanded=n_errors > 0,
            ):
                st.code("\n".join(lines), language=None)
