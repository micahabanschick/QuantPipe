"""Live Trading Dashboard — real-money account monitoring via IBKR.

Connects to TWS/Gateway to pull live positions and NAV, then overlays
the same deployment-marker equity curve used by the paper trading view.
Intentionally minimal — the paper trading tab is the primary monitoring tool.
"""

import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import DATA_DIR, IBKR_CLIENT_ID, IBKR_HOST, IBKR_PORT
from reports._theme import COLORS, apply_theme

log = logging.getLogger(__name__)

_GOLD = DATA_DIR / "gold" / "equity"
_TH_PATH     = _GOLD / "trading_history.parquet"
_DEPLOY_PATH = _GOLD / "deployment_history.jsonl"
_BROKER = "ibkr"


def _tcp_probe(host: str, port: int, timeout: float = 1.5) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_live_nav_history() -> pd.DataFrame:
    """Load NAV snapshots for the live IBKR broker."""
    if not _TH_PATH.exists():
        return pd.DataFrame()
    import polars as pl
    df = pl.read_parquet(_TH_PATH).filter(pl.col("broker") == _BROKER)
    return df.sort("date").to_pandas()


def _fetch_live_snapshot(host: str, port: int, client_id: int) -> dict | None:
    """Connect to IBKR and pull current NAV, positions, and cash."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(host, port, clientId=client_id, readonly=True)
        positions = ib.positions()
        account_vals = ib.accountValues()
        ib.disconnect()

        cash = 0.0
        nav = 0.0
        for av in account_vals:
            if av.tag == "TotalCashValue" and av.currency == "USD":
                cash = float(av.value)
            if av.tag == "NetLiquidation" and av.currency == "USD":
                nav = float(av.value)

        pos_rows = []
        for p in positions:
            pos_rows.append({
                "symbol": p.contract.localSymbol or p.contract.symbol,
                "qty": float(p.position),
                "avg_cost": float(p.avgCost),
                "market_value": float(p.position * p.avgCost),
            })

        return {"nav": nav, "cash": cash, "positions": pos_rows,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Live Trading")
st.caption("Real-money IBKR account monitoring. "
           "Paper trading is your primary active account — this tab is for future use.")

# ── Connection panel ──────────────────────────────────────────────────────────
with st.expander("IBKR Connection", expanded=True):
    ca, cb, cc = st.columns(3)
    with ca:
        host = st.text_input("Host", value=str(IBKR_HOST), key="lt_host")
    with cb:
        port = st.number_input("Port (live: 7496 / 4001)", value=7496,
                               min_value=1, max_value=65535, key="lt_port")
    with cc:
        client_id = st.number_input("Client ID", value=int(IBKR_CLIENT_ID),
                                    min_value=0, max_value=999, key="lt_client_id")

    reachable = _tcp_probe(host, int(port))
    if reachable:
        st.success(f"TWS/Gateway reachable at {host}:{port}")
    else:
        st.warning(f"Nothing found at {host}:{port} — connect to the **live** session in TWS/Gateway "
                   f"(port 7496 for TWS, 4001 for Gateway).")

# ── Live snapshot ─────────────────────────────────────────────────────────────
if reachable:
    if st.button("Fetch Live Snapshot", type="primary", key="lt_fetch"):
        with st.spinner("Connecting to IBKR…"):
            snap = _fetch_live_snapshot(host, int(port), int(client_id))
        if "error" in snap:
            st.error(f"Could not fetch live data: {snap['error']}")
        else:
            st.session_state["lt_snapshot"] = snap

    if "lt_snapshot" in st.session_state:
        snap = st.session_state["lt_snapshot"]
        st.caption(f"Snapshot as of {snap.get('fetched_at', '—')} UTC")

        m1, m2 = st.columns(2)
        m1.metric("Net Liquidation", f"${snap['nav']:,.0f}")
        m2.metric("Cash", f"${snap['cash']:,.0f}")

        if snap["positions"]:
            pos_df = pd.DataFrame(snap["positions"])
            pos_df["market_value"] = (pos_df["market_value"]).round(0).astype(int)
            pos_df["avg_cost"] = pos_df["avg_cost"].map("${:,.2f}".format)
            st.markdown("### Open Positions")
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions in live account.")

# ── Recorded live NAV history ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Live Account NAV History")
nav_hist = _load_live_nav_history()

if not nav_hist.empty:
    nav_hist["date"] = pd.to_datetime(nav_hist["date"])
    fig = go.Figure(go.Scatter(
        x=nav_hist["date"], y=nav_hist["nav"],
        name="Live NAV",
        line=dict(color=COLORS["info"], width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color=COLORS["info"]),
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))
    apply_theme(fig, title="Live Account NAV", height=360)
    fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No live NAV history recorded yet. NAV snapshots are written automatically "
        "after each `--broker ibkr --ibkr-live` rebalance."
    )

st.markdown("---")
st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
     border-radius:8px;padding:16px 20px;">
  <div style="color:{COLORS['warning']};font-weight:700;margin-bottom:8px;">
    Live trading is currently disabled
  </div>
  <div style="color:{COLORS['neutral']};font-size:0.85rem;line-height:1.7;">
    To enable live execution, go to the <strong>Portfolio → Trade</strong> tab,
    select <em>Live Trading</em>, and confirm. All live orders require TWS connected
    to your real-money account on port 7496 (TWS) or 4001 (Gateway).
  </div>
</div>
""", unsafe_allow_html=True)
