"""Paper Trading Dashboard — live portfolio monitoring for the paper account.

Data sources (all in data/gold/equity/):
  trading_history.parquet   — NAV snapshots written after every rebalance
  target_weights.parquet    — position weights at each rebalance date
  order_journal.parquet     — every order attempt with status
  deployment_history.jsonl  — one entry per deployment config save

Equity curve is computed daily from target_weights × bronze-layer prices,
anchored at the NAV snapshot from each rebalance. Deployment changes are
shown as vertical dotted lines on the curve.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import DATA_DIR
from reports._theme import COLORS, apply_theme, badge

log = logging.getLogger(__name__)

_GOLD = DATA_DIR / "gold" / "equity"
_TW_PATH    = _GOLD / "target_weights.parquet"
_TH_PATH    = _GOLD / "trading_history.parquet"
_OJ_PATH    = _GOLD / "order_journal.parquet"
_DEPLOY_PATH = _GOLD / "deployment_history.jsonl"

_INITIAL_NAV = 100_000.0   # paper broker starting cash


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_nav_snapshots() -> pd.DataFrame:
    """NAV recorded after each rebalance (sparse — only at trade dates)."""
    if not _TH_PATH.exists():
        return pd.DataFrame(columns=["date", "nav", "cash", "n_positions"])
    import polars as pl
    df = pl.read_parquet(_TH_PATH).filter(pl.col("broker") == _BROKER)
    return df.sort("date").to_pandas()


def _load_target_weights() -> pd.DataFrame:
    """All rebalance weights (date, symbol, weight)."""
    if not _TW_PATH.exists():
        return pd.DataFrame()
    import polars as pl
    return pl.read_parquet(_TW_PATH).sort("date").to_pandas()


def _load_deployment_events() -> list[dict]:
    """Return list of {timestamp, version, label} for deployment history."""
    if not _DEPLOY_PATH.exists():
        return []
    events = []
    for line in _DEPLOY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            slugs = [s["slug"] for s in e.get("strategies", [])]
            e["label"] = f"v{e['version']}: {', '.join(slugs)}"
            e["ts"] = pd.Timestamp(e["timestamp"]).tz_convert(None)  # make tz-naive
            events.append(e)
        except Exception:
            pass
    return events


def _load_order_journal() -> pd.DataFrame:
    if not _OJ_PATH.exists():
        return pd.DataFrame()
    import polars as pl
    df = pl.read_parquet(_OJ_PATH).filter(pl.col("broker") == _BROKER)
    return df.sort("rebalance_date", descending=True).to_pandas()


@st.cache_data(ttl=60)
def _build_equity_curve() -> pd.Series:
    """Compute daily NAV from target_weights × daily prices.

    Returns a pd.Series indexed by pd.Timestamp with NAV values.
    Anchored at the actual recorded NAV from trading_history if available,
    otherwise starts at _INITIAL_NAV on the first rebalance date.
    """
    tw = _load_target_weights()
    if tw.empty:
        return pd.Series(dtype=float)

    # Use rebalance_date (when the strategy actually traded) as the curve anchor,
    # not date (when signals were computed, which may be today with no bronze data).
    date_col = "rebalance_date" if "rebalance_date" in tw.columns else "date"
    tw[date_col] = pd.to_datetime(tw[date_col]).dt.date
    rebal_dates = sorted(tw[date_col].unique())
    all_symbols = tw["symbol"].unique().tolist()

    # Load prices from bronze layer
    try:
        from storage.parquet_store import load_bars
        start = rebal_dates[0]
        end = date.today()
        raw = load_bars(all_symbols, start, end, "equity")
        if raw.is_empty():
            log.warning(f"No bronze prices found for {all_symbols} from {start} to {end}")
            return pd.Series(dtype=float)
        price_col = "adj_close" if "adj_close" in raw.columns else "close"
        prices_wide = (
            raw.to_pandas()
            .pivot(index="date", columns="symbol", values=price_col)
            .sort_index()
        )
        prices_wide.index = pd.to_datetime(prices_wide.index)
    except Exception as exc:
        log.warning(f"Could not load prices for equity curve: {exc}")
        return pd.Series(dtype=float)

    # Build anchor NAV map from trading_history snapshots
    snap = _load_nav_snapshots()
    snap_nav: dict[date, float] = {}
    if not snap.empty:
        snap["date"] = pd.to_datetime(snap["date"]).dt.date
        for _, row in snap.iterrows():
            snap_nav[row["date"]] = float(row["nav"])

    # Walk through trading days and accumulate NAV
    nav_series: dict[pd.Timestamp, float] = {}
    nav = _INITIAL_NAV
    current_weights: dict[str, float] = {}
    prev_ts: pd.Timestamp | None = None

    for ts in prices_wide.index:
        d = ts.date()

        # Check for rebalance on this date (matched against rebalance_date)
        tw_today = tw[tw[date_col] == d]
        if not tw_today.empty:
            current_weights = dict(zip(tw_today["symbol"], tw_today["weight"]))
            # Use actual recorded NAV if available, else keep running estimate
            if d in snap_nav:
                nav = snap_nav[d]
            # Record this date (and reset prev for clean return calculation)
            nav_series[ts] = nav
            prev_ts = ts
            continue

        if not current_weights or prev_ts is None:
            continue

        # Compute portfolio daily return
        port_ret = 0.0
        for sym, w in current_weights.items():
            if sym not in prices_wide.columns:
                continue
            p_prev = prices_wide.at[prev_ts, sym] if prev_ts in prices_wide.index else np.nan
            p_curr = prices_wide.at[ts, sym]
            if pd.notna(p_prev) and pd.notna(p_curr) and p_prev > 0:
                port_ret += w * (p_curr / p_prev - 1)

        nav *= (1.0 + port_ret)
        nav_series[ts] = nav
        prev_ts = ts

    return pd.Series(nav_series).sort_index()


# ── Metric helpers ─────────────────────────────────────────────────────────────

_MIN_DAYS_ANNUALIZED = 63   # ~3 months before CAGR / Sharpe are statistically meaningful

def _compute_metrics(eq: pd.Series) -> dict:
    if eq.empty or len(eq) < 2:
        return {}
    n_days = len(eq)
    daily_ret = eq.pct_change().dropna()
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    n_years = max(n_days / 252, 1e-6)

    # Annualised metrics are unreliable below ~3 months of data — return None so
    # the UI can display "n/a" instead of a misleading extrapolated number.
    reliable = n_days >= _MIN_DAYS_ANNUALIZED
    cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if reliable else None
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if (reliable and daily_ret.std() > 0) else None

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min())
    return {
        "nav":       float(eq.iloc[-1]),
        "n_days":    n_days,
        "total_ret": total_ret,
        "cagr":      cagr,
        "sharpe":    sharpe,
        "max_dd":    max_dd,
        "start":     eq.index[0],
        "end":       eq.index[-1],
    }


def _pct(v, decimals=1, sign=True):
    fmt = f"{v * 100:+.{decimals}f}%" if sign else f"{v * 100:.{decimals}f}%"
    return fmt if v is not None else "—"


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _add_deployment_vlines(fig: go.Figure, events: list[dict], eq: pd.Series) -> None:
    """Add a vertical dotted line + annotation for each deployment event."""
    if eq.empty:
        return
    t_min, t_max = eq.index[0], eq.index[-1]
    for e in events:
        ts = e["ts"]
        if not (t_min <= ts <= t_max):
            continue
        # Find NAV at this timestamp for annotation height
        closest = eq.index.get_indexer([ts], method="nearest")[0]
        y_val = float(eq.iloc[closest]) if closest >= 0 else float(eq.mean())
        fig.add_vline(
            x=ts.timestamp() * 1000,  # Plotly needs milliseconds for datetime axes
            line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=ts, y=y_val,
            text=f"Deploy {e['label']}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["warning"],
            font=dict(size=10, color=COLORS["warning"]),
            bgcolor=COLORS["card_bg"], bordercolor=COLORS["warning"],
            borderwidth=1, borderpad=4,
            ax=30, ay=-40,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Paper Trading")
st.caption("Live portfolio monitoring — equity curve updates after each rebalance.")

# ── Refresh controls ──────────────────────────────────────────────────────────
hdr_left, hdr_mid, hdr_right = st.columns([3, 2, 1])
with hdr_left:
    show_deployments = st.toggle("Show deployment markers", value=True,
                                 key="pt_show_deploy")
with hdr_mid:
    _mode = st.radio("Mode", ["Paper", "Live"], index=0, horizontal=True, key="pt_mode",
                     help="Switch between paper and live account view")
with hdr_right:
    if st.button("Refresh", key="pt_refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

_BROKER = "paper" if _mode == "Paper" else "live"

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Computing equity curve…"):
    eq = _build_equity_curve()

metrics    = _compute_metrics(eq)
snap       = _load_nav_snapshots()
deploy_evt = _load_deployment_events() if show_deployments else []
orders     = _load_order_journal()
tw         = _load_target_weights()

# ── KPI cards ─────────────────────────────────────────────────────────────────
if metrics:
    nav_now   = metrics["nav"]
    total_ret = metrics["total_ret"]
    sharpe    = metrics["sharpe"]
    max_dd    = metrics["max_dd"]
    n_days    = metrics["n_days"]
    pnl       = nav_now - _INITIAL_NAV

    pnl_color = COLORS["positive"] if pnl >= 0 else COLORS["negative"]

    # Warn when annualised metrics are statistically unreliable
    if n_days < _MIN_DAYS_ANNUALIZED:
        st.warning(
            f"**{n_days} trading days of data** — CAGR and Sharpe require at least "
            f"{_MIN_DAYS_ANNUALIZED} days (~3 months) to be meaningful. "
            f"They will appear as **n/a** until then.",
            icon="⚠️",
        )

    cagr_str   = _pct(metrics["cagr"])   if metrics["cagr"]   is not None else "n/a"
    sharpe_str = f"{sharpe:.2f}"          if sharpe             is not None else "n/a"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio NAV", f"${nav_now:,.0f}",
               delta=f"${pnl:+,.0f} vs $100K sim start")
    k2.metric("Total Return", _pct(total_ret),
               delta=f"CAGR {cagr_str}")
    k3.metric("Sharpe Ratio", sharpe_str,
               help="Annualised. Shown as n/a when fewer than 63 trading days available.")
    k4.metric("Max Drawdown", _pct(max_dd, sign=False),
               delta=f"from {metrics['start'].strftime('%Y-%m-%d')}",
               delta_color="off")
else:
    st.info(
        "No trading history yet. Run a rebalance first:\n\n"
        "```\nuv run python orchestration/rebalance.py --broker paper\n```"
    )

st.markdown("---")

# ── Equity curve ──────────────────────────────────────────────────────────────
if not eq.empty:
    fig = go.Figure()

    # Optional backtest overlay from portfolio_log.parquet
    _pl_path = _GOLD / "portfolio_log.parquet"
    if _pl_path.exists():
        try:
            import polars as _pl
            _pl_df = _pl.read_parquet(_pl_path).sort("date")
            if "portfolio_value" in _pl_df.columns:
                _pl_pd = _pl_df.select(["date", "portfolio_value"]).to_pandas()
                _pl_pd["date"] = pd.to_datetime(_pl_pd["date"])
                _pl_pd = _pl_pd.set_index("date")
                # Scale to same starting NAV as paper portfolio
                if not _pl_pd.empty and not eq.empty:
                    _scale = eq.iloc[0] / _pl_pd["portfolio_value"].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=_pl_pd.index, y=(_pl_pd["portfolio_value"] * _scale).values,
                        name="Backtest (scaled)",
                        line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
                        opacity=0.6,
                        hovertemplate="Backtest: $%{y:,.0f}<extra></extra>",
                    ))
        except Exception:
            pass

    # Main equity curve
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name=f"{_mode} Portfolio",
        line=dict(color=COLORS["positive"], width=2.5),
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))

    # Rebalance dots (from NAV snapshots)
    if not snap.empty:
        snap_ts = pd.to_datetime(snap["date"])
        snap_navs = []
        for ts in snap_ts:
            idx = eq.index.get_indexer([ts], method="nearest")[0]
            snap_navs.append(float(eq.iloc[idx]) if idx >= 0 else np.nan)
        fig.add_trace(go.Scatter(
            x=snap_ts, y=snap_navs,
            name="Rebalance",
            mode="markers",
            marker=dict(color=COLORS["positive"], size=8,
                        line=dict(color=COLORS["bg"], width=2)),
            hovertemplate="Rebalance<br>$%{y:,.0f}<extra></extra>",
        ))

    # Deployment vertical lines
    if show_deployments:
        _add_deployment_vlines(fig, deploy_evt, eq)

    # Drawdown shading (fill between NAV and running peak)
    peak = eq.cummax()
    dd_vals = eq - peak
    fig.add_trace(go.Scatter(
        x=eq.index.tolist() + eq.index[::-1].tolist(),
        y=peak.tolist() + eq[::-1].tolist(),
        fill="toself",
        fillcolor=f"rgba(255,75,75,0.07)",
        line=dict(width=0),
        name="Drawdown",
        showlegend=True,
        hoverinfo="skip",
    ))

    apply_theme(fig, title="Paper Portfolio Equity Curve", height=420)
    fig.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    if len(eq) > 5:
        dd_pct = ((eq - eq.cummax()) / eq.cummax()) * 100
        fig_dd = go.Figure(go.Scatter(
            x=dd_pct.index, y=dd_pct.values,
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.12)",
            line=dict(color=COLORS["negative"], width=1.5),
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
        apply_theme(fig_dd, title="Drawdown", height=180)
        fig_dd.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True)

else:
    st.info("No price history available yet. Run `orchestration/generate_signals.py` then a rebalance.")

st.markdown("---")

# ── Current positions ─────────────────────────────────────────────────────────
col_pos, col_trades = st.columns([1, 1])

with col_pos:
    st.markdown("### Current Positions")
    if not tw.empty:
        tw["date"] = pd.to_datetime(tw["date"]).dt.date
        latest_date = tw["date"].max()
        latest_tw = tw[tw["date"] == latest_date].sort_values("weight", ascending=False)

        # Compute dollar values if we have a NAV
        current_nav = metrics.get("nav", _INITIAL_NAV)
        latest_tw = latest_tw.copy()
        latest_tw["value ($)"] = (latest_tw["weight"] * current_nav).round(0).astype(int)
        latest_tw["weight (%)"] = (latest_tw["weight"] * 100).round(2)

        display = latest_tw[["symbol", "weight (%)", "value ($)"]].reset_index(drop=True)
        st.caption(f"As of rebalance {latest_date}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Pie chart
        fig_pie = go.Figure(go.Pie(
            labels=latest_tw["symbol"].tolist(),
            values=latest_tw["weight"].tolist(),
            marker=dict(colors=COLORS["series"], line=dict(color=COLORS["bg"], width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color=COLORS["text"]),
            hole=0.5,
        ))
        apply_theme(fig_pie, height=260)
        fig_pie.update_layout(showlegend=False, paper_bgcolor=COLORS["card_bg"],
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No target weights found.")

with col_trades:
    st.markdown("### Trade History")
    if not orders.empty:
        display_orders = orders[[
            "rebalance_date", "symbol", "qty", "est_price", "status", "order_id"
        ]].copy()
        display_orders.columns = ["Date", "Symbol", "Qty", "Est. Price", "Status", "Order ID"]
        display_orders["Est. Price"] = display_orders["Est. Price"].map("${:,.2f}".format)
        display_orders["Qty"] = display_orders["Qty"].map("{:+.0f}".format)
        st.dataframe(display_orders.head(30), use_container_width=True, hide_index=True)
    else:
        st.info("No orders recorded yet.")

# ── Deployment history ────────────────────────────────────────────────────────
if deploy_evt:
    st.markdown("---")
    st.markdown("### Deployment History")
    rows = []
    for e in reversed(deploy_evt):
        strategies_str = ", ".join(
            f"{s['name']} ({s['allocation_weight']*100:.0f}%)"
            for s in e.get("strategies", [])
        )
        rows.append({
            "Version": e["version"],
            "Deployed at": e["timestamp"],
            "Active strategies": strategies_str,
        })
    st.table(pd.DataFrame(rows))

# ── Slippage Tracking ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Slippage Analysis")
if not orders.empty and "est_price" in orders.columns:
    _slippage_df = orders.copy()
    fill_col = next((c for c in ["fill_price", "actual_price", "avg_fill_px"] if c in _slippage_df.columns), None)
    if fill_col:
        _slippage_df["slippage_bps"] = ((_slippage_df[fill_col] - _slippage_df["est_price"]) / _slippage_df["est_price"].replace(0, np.nan) * 10_000).round(2)
        _slippage_df["abs_slip_bps"] = _slippage_df["slippage_bps"].abs()
        _sdisp = _slippage_df[["rebalance_date", "symbol", "qty", "est_price", fill_col, "slippage_bps", "status"]].copy()
        _sdisp.columns = ["Date", "Symbol", "Qty", "Est. Price", "Fill Price", "Slip (bps)", "Status"]
        _smean = float(_slippage_df["abs_slip_bps"].mean())
        _smax  = float(_slippage_df["abs_slip_bps"].max())
        _sc1, _sc2 = st.columns(2)
        _sc1.metric("Mean |Slippage|", f"{_smean:.1f} bps")
        _sc2.metric("Max |Slippage|", f"{_smax:.1f} bps")
        st.dataframe(_sdisp, hide_index=True, use_container_width=True, height=200)
    else:
        st.info("No fill price column in order journal — slippage cannot be computed yet.")
        st.caption("Expected column name: `fill_price`, `actual_price`, or `avg_fill_px`")
else:
    st.info("No orders in journal yet.")

# ── Position Reconciliation ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Position Reconciliation")
st.caption("Compares target weights (last rebalance) against estimated current exposure based on price drift.")
if not tw.empty and metrics:
    tw["date"] = pd.to_datetime(tw["date"]).dt.date
    _latest_tw_date = tw["date"].max()
    _latest_tw = tw[tw["date"] == _latest_tw_date][["symbol", "weight"]].set_index("symbol")["weight"]

    # Estimate current weights via price drift
    _current_est = _latest_tw.copy()
    try:
        from storage.parquet_store import load_bars as _lb
        _rec_prices = _lb(_latest_tw.index.tolist(), _latest_tw_date, date.today(), "equity")
        if not _rec_prices.is_empty():
            _rec_wide = _rec_prices.sort("date").to_pandas().pivot(index="date", columns="symbol", values="close")
            if len(_rec_wide) >= 2:
                _growth = _rec_wide.iloc[-1] / _rec_wide.iloc[0]
                _drifted = _latest_tw * _growth.reindex(_latest_tw.index).fillna(1.0)
                _current_est = _drifted / _drifted.sum()
    except Exception:
        pass

    _recon_df = pd.DataFrame({
        "Symbol":  _latest_tw.index.tolist(),
        "Target":  [f"{_latest_tw.get(s, 0):.2%}" for s in _latest_tw.index],
        "Estimated": [f"{_current_est.get(s, 0):.2%}" for s in _latest_tw.index],
        "Drift":   [f"{(_current_est.get(s, 0) - _latest_tw.get(s, 0)):+.2%}" for s in _latest_tw.index],
    })
    st.caption(f"Target as of {_latest_tw_date}")
    st.dataframe(_recon_df, hide_index=True, use_container_width=True)
else:
    st.info("No target weights available for reconciliation.")
