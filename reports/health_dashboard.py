"""Pipeline health dashboard — Streamlit app (Dashboard #1).

Shows: last ingestion time, row counts, validation status, recent failures.
Run with: streamlit run reports/health_dashboard.py

This is the ops dashboard — not P&L. Keep it focused on pipeline health.
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from config.settings import DATA_DIR, LOGS_DIR
from storage.parquet_store import list_symbols, load_bars

st.set_page_config(page_title="QuantPipe — Pipeline Health", layout="wide")
st.title("QuantPipe — Pipeline Health")
st.caption(f"As of {date.today()}")

LOOKBACK = 30   # days to show in row-count chart


# ── Helper functions ──────────────────────────────────────────────────────────

def _last_modified(asset_class: str) -> str:
    """Return ISO timestamp of most recently modified Parquet file."""
    base = DATA_DIR / "bronze" / asset_class / "daily"
    if not base.exists():
        return "No data"
    files = list(base.rglob("*.parquet"))
    if not files:
        return "No files"
    latest = max(files, key=lambda p: p.stat().st_mtime)
    from datetime import datetime
    ts = datetime.fromtimestamp(latest.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _tail_log(path: Path, n: int = 50) -> list[str]:
    if not path.exists():
        return ["Log file not found"]
    with open(path) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


# ── Layout ────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Last successful ingestion")
    st.metric("Equity", _last_modified("equity"))
    st.metric("Crypto", _last_modified("crypto"))

with col2:
    st.subheader("Universe size in storage")
    eq_symbols = list_symbols("equity")
    cr_symbols = list_symbols("crypto")
    st.metric("Equity symbols", len(eq_symbols))
    st.metric("Crypto symbols", len(cr_symbols))

st.divider()

# Row count spot-check for equity universe
st.subheader(f"Equity row counts — last {LOOKBACK} days")
end = date.today()
start = end - timedelta(days=LOOKBACK)

if eq_symbols:
    sample = eq_symbols[:20]   # show first 20 to avoid giant table
    df = load_bars(sample, start, end, asset_class="equity")
    if not df.is_empty():
        counts = (
            df.group_by("symbol")
            .agg(pl.count("date").alias("rows"), pl.max("date").alias("latest_date"))
            .sort("symbol")
        )
        st.dataframe(counts.to_pandas(), use_container_width=True)
    else:
        st.warning("No equity data found in storage for this window.")
else:
    st.info("No equity symbols found in storage. Run backfill first.")

st.divider()

# Recent log tail
st.subheader("Recent ingestion log (last 50 lines)")
log_lines = _tail_log(LOGS_DIR / "ingest.log")
failures = [l for l in log_lines if "ERROR" in l or "FAILED" in l or "ALERT" in l]

if failures:
    st.error(f"{len(failures)} failure(s) in recent log")
    for line in failures:
        st.code(line, language=None)
else:
    st.success("No errors in recent log")

with st.expander("Full log tail"):
    st.code("\n".join(log_lines), language=None)
