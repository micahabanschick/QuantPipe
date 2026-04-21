"""Pipeline health dashboard — Streamlit app (Dashboard #1).

Shows: last ingestion time, symbol counts, row counts per symbol,
validation pass/fail summary, cron job status, and recent log tail.

Run with: streamlit run reports/health_dashboard.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from reports._theme import CSS, COLORS, apply_theme, badge
from storage.parquet_store import list_symbols, load_bars

st.set_page_config(page_title="QuantPipe — Pipeline Health", layout="wide", page_icon="🔧")
st.markdown(CSS, unsafe_allow_html=True)

LOOKBACK = 30
TARGET_WEIGHTS_PATH = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
PORTFOLIO_LOG_PATH = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_modified(asset_class: str) -> tuple[str, float]:
    base = DATA_DIR / "bronze" / asset_class / "daily"
    if not base.exists():
        return "No data", float("inf")
    files = list(base.rglob("*.parquet"))
    if not files:
        return "No files", float("inf")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    ts = datetime.fromtimestamp(latest.stat().st_mtime)
    age_hours = (datetime.now() - ts).total_seconds() / 3600
    return ts.strftime("%Y-%m-%d %H:%M"), age_hours


def _last_log_run(log_path: Path) -> tuple[str, bool]:
    if not log_path.exists():
        return "No log file", False
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "Pipeline complete" in line or "pipeline complete" in line.lower():
                parts = line.split()
                ts = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "unknown"
                success = "0 failure" in line or "0 ingestion failure" in line
                return ts, success
        return "No complete marker found", False
    except Exception:
        return "Log read error", False


def _tail_log(path: Path, n: int = 60) -> list[str]:
    if not path.exists():
        return ["Log file not found"]
    with open(path) as f:
        lines = f.readlines()
    return [line.rstrip() for line in lines[-n:]]


def _count_errors(lines: list[str]) -> int:
    return sum(1 for line in lines if " ERROR " in line or "FAILED" in line or "ALERT" in line)


def _signals_freshness() -> tuple[str, bool]:
    if not TARGET_WEIGHTS_PATH.exists():
        return "No signals file", False
    mtime = datetime.fromtimestamp(TARGET_WEIGHTS_PATH.stat().st_mtime)
    is_today = mtime.date() == date.today()
    return mtime.strftime("%Y-%m-%d %H:%M"), is_today


# ── Compute status ────────────────────────────────────────────────────────────

eq_ts, eq_age = _last_modified("equity")
cr_ts, cr_age = _last_modified("crypto")
sig_ts, sig_fresh = _signals_freshness()
pipe_ts, pipe_ok = _last_log_run(LOGS_DIR / "pipeline.log")

all_healthy = eq_age < 30 and cr_age < 30 and sig_fresh and pipe_ok
any_critical = eq_age == float("inf") or cr_age == float("inf")

if any_critical:
    status_color = COLORS["negative"]
    status_text = "CRITICAL — Data missing"
elif not all_healthy:
    status_color = COLORS["warning"]
    status_text = "WARNING — Some data stale or pipeline incomplete"
else:
    status_color = COLORS["positive"]
    status_text = "HEALTHY — All systems nominal"

# ── Header + status banner ────────────────────────────────────────────────────

st.markdown(
    f'<div style="background:{status_color}18; border:1px solid {status_color}; '
    f'border-radius:8px; padding:12px 20px; margin-bottom:20px; '
    f'display:flex; align-items:center; gap:12px;">'
    f'<span style="font-size:1.1rem; font-weight:700; color:{status_color};">'
    f'● {status_text}</span>'
    f'<span style="color:{COLORS["neutral"]}; font-size:0.85rem; margin-left:auto;">'
    f'As of {date.today()}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.title("QuantPipe — Pipeline Health")

# ── Section 1: Pipeline status ────────────────────────────────────────────────

st.subheader("Pipeline Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Equity Ingest", eq_ts,
              delta=f"{eq_age:.0f}h ago" if eq_age < float("inf") else None,
              delta_color="normal" if eq_age < 30 else "inverse")

with col2:
    st.metric("Crypto Ingest", cr_ts,
              delta=f"{cr_age:.0f}h ago" if cr_age < float("inf") else None,
              delta_color="normal" if cr_age < 30 else "inverse")

with col3:
    st.metric("Signals", sig_ts,
              delta="today" if sig_fresh else "stale",
              delta_color="normal" if sig_fresh else "inverse")

with col4:
    st.metric("Last Pipeline Run", pipe_ts,
              delta="OK" if pipe_ok else "check logs",
              delta_color="normal" if pipe_ok else "inverse")

st.divider()

# ── Section 2: Universe sizes ─────────────────────────────────────────────────

st.subheader("Universe in Storage")

eq_symbols = list_symbols("equity")
cr_symbols = list_symbols("crypto")

col1, col2 = st.columns(2)
col1.metric("Equity Symbols", len(eq_symbols))
col2.metric("Crypto Symbols", len(cr_symbols))

st.divider()

# ── Section 3: Row counts table ───────────────────────────────────────────────

st.subheader(f"Equity Row Counts — Last {LOOKBACK} Days")
end = date.today()
start = end - timedelta(days=LOOKBACK)

if eq_symbols:
    sample = sorted(eq_symbols)[:26]
    df = load_bars(sample, start, end, asset_class="equity")
    if not df.is_empty():
        counts = (
            df.group_by("symbol")
            .agg(
                pl.len().alias("rows"),
                pl.max("date").alias("latest_date"),
                pl.min("date").alias("earliest_date"),
            )
            .sort("symbol")
        )
        counts = counts.with_columns(
            pl.when(pl.col("latest_date") < (end - timedelta(days=5)))
            .then(pl.lit("STALE"))
            .otherwise(pl.lit("OK"))
            .alias("status")
        )
        n_stale = counts.filter(pl.col("status") == "STALE").height
        if n_stale > 0:
            st.warning(f"{n_stale} symbol(s) have stale data (>5 days old)")
        else:
            st.success("All symbols have fresh data")

        counts_pd = counts.to_pandas()
        st.dataframe(
            counts_pd.style.apply(
                lambda col: [
                    f"color: {COLORS['negative']}" if v == "STALE"
                    else f"color: {COLORS['positive']}"
                    for v in col
                ] if col.name == "status" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No equity data found for this window. Run backfill first.")
else:
    st.info("No equity symbols found in storage. Run backfill first.")

st.divider()

# ── Section 4: Portfolio snapshot ────────────────────────────────────────────

st.subheader("Latest Portfolio Snapshot")

if PORTFOLIO_LOG_PATH.exists():
    plog = pl.read_parquet(PORTFOLIO_LOG_PATH).sort("date")
    if not plog.is_empty():
        latest = plog.tail(1).to_dicts()[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positions", latest["n_positions"])
        col2.metric("VaR 95%", f"{latest['var_1d_95']:.2%}")
        col3.metric("Gross Exposure", f"{latest['gross_exposure']:.1%}")

        passed = latest["pre_trade_passed"]
        with col4:
            st.metric("Pre-trade Check", "")
            st.markdown(badge("PASS", "positive") if passed else badge("FAIL", "negative"),
                        unsafe_allow_html=True)

        if len(plog) > 3:
            plog_pd = plog.select(["date", "var_1d_95", "gross_exposure"]).to_pandas()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plog_pd["date"], y=plog_pd["var_1d_95"],
                name="VaR 95%",
                line=dict(color=COLORS["negative"], width=2),
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>VaR: %{y:.2%}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=plog_pd["date"], y=plog_pd["gross_exposure"],
                name="Gross Exposure",
                line=dict(color=COLORS["blue"], width=2, dash="dot"),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Exp: %{y:.1%}<extra></extra>",
            ))
            apply_theme(fig, "Risk Metrics Trend")
            fig.update_layout(
                height=260,
                yaxis=dict(title="VaR", tickformat=".2%"),
                yaxis2=dict(
                    title="Gross Exp.",
                    tickformat=".0%",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Portfolio log is empty.")
else:
    st.info("No portfolio log yet. Run: `uv run python orchestration/generate_signals.py`")

st.divider()

# ── Section 5: Log tails ──────────────────────────────────────────────────────

st.subheader("Recent Logs")

tab_pipeline, tab_ingest, tab_signals = st.tabs(["Pipeline", "Ingest", "Signals"])

for tab, log_name in [
    (tab_pipeline, "pipeline.log"),
    (tab_ingest, "ingest.log"),
    (tab_signals, "signals.log"),
]:
    with tab:
        lines = _tail_log(LOGS_DIR / log_name)
        errors = _count_errors(lines)
        if errors:
            st.error(f"{errors} error(s) in {log_name}")
        else:
            st.success(f"No errors in {log_name}")
        with st.expander(f"Show {log_name} tail ({len(lines)} lines)"):
            st.code("\n".join(lines), language=None)
