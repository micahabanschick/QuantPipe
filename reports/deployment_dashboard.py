"""Deployment Dashboard — Portfolio group, page 2.

Tabs:
  1. Strategy Config  — active/inactive toggles, allocation weights, save config
  2. Drift Monitor    — target vs actual weight drift detection
  3. Trade            — IB paper / live execution
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import COLORS, apply_theme, page_header, CSS, kpi_card, section_label

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


def _pct(v, decimals=1):
    return f"{v * 100:+.{decimals}f}%" if v is not None else "—"


def _fmt(v, pct=True, decimals=1):
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%" if pct else f"{v:.{decimals}f}"


@st.cache_resource(show_spinner=False)
def _get_discover_fn():
    from portfolio.multi_strategy import discover_strategies
    return discover_strategies


@st.cache_resource(show_spinner=False)
def _get_config_fn():
    from portfolio.multi_strategy import read_deployment_config
    return read_deployment_config


def _discover():
    return _get_discover_fn()()


def _load_config():
    return _get_config_fn()()


# ── Page ───────────────────────────────────────────────────────────────────────

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    page_header("Deployment", "Configure strategy weights, monitor allocation drift, and execute rebalances."),
    unsafe_allow_html=True,
)

metas  = _discover()
config = _load_config()
results: dict = st.session_state.get("backtest_results", {})

tab_deploy, tab_drift, tab_trade = st.tabs([
    "  Strategy Config  ",
    "  Drift Monitor  ",
    "  Trade  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Strategy Config
# ══════════════════════════════════════════════════════════════════════════════

with tab_deploy:
    from portfolio.multi_strategy import (
        write_deployment_config, DeploymentConfig, DeployedStrategy,
    )

    st.markdown("### Deployment Configuration")
    st.caption(
        "Toggle strategies active/inactive and set allocation weights. "
        "Weights are normalised before use. Changes take effect on the next "
        "`orchestration/generate_signals.py` run."
    )

    if not metas:
        st.warning("No strategies found in the `strategies/` directory.")
    else:
        existing_map = {s.slug: s for s in config.strategies} if config else {}
        updated: list[DeployedStrategy] = []

        for i, m in enumerate(metas):
            ex = existing_map.get(m.slug)
            default_active = ex.active if ex else True
            default_weight = ex.allocation_weight if ex else round(1.0 / len(metas), 3)
            default_params = ex.backtest_params if ex else m.default_params
            has_result = m.slug in results

            with st.expander(f"{m.name}  [{'cached' if has_result else 'no data'}]", expanded=True):
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    active = st.checkbox("Active", value=default_active, key=f"active_{m.slug}")
                with c2:
                    weight = st.number_input(
                        "Allocation weight", min_value=0.0, max_value=1.0,
                        value=float(default_weight), step=0.05,
                        key=f"weight_{m.slug}", disabled=not active,
                    )
                with c3:
                    if has_result:
                        mm = results[m.slug].metrics
                        st.markdown(
                            f"CAGR **{_pct(mm.get('cagr'))}** · "
                            f"Sharpe **{mm.get('sharpe', 0):.2f}** · "
                            f"Max DD **{_pct(mm.get('max_drawdown'))}**"
                        )
                    else:
                        st.caption("Run backtests in **Multi-Strategy → Comparison** first.")

            updated.append(DeployedStrategy(
                slug=m.slug, name=m.name, active=active,
                allocation_weight=weight if active else 0.0,
                backtest_params=default_params,
            ))

        total = sum(s.allocation_weight for s in updated if s.active) or 1.0
        active_count = sum(1 for s in updated if s.active)
        st.markdown(f"**{active_count}** active strategies · raw weight sum = **{total:.3f}**")

        rows = [
            {"Strategy": s.name, "Raw": f"{s.allocation_weight:.3f}",
             "Normalised": f"{s.allocation_weight / total * 100:.1f}%" if s.active else "—"}
            for s in updated if s.active
        ]
        if rows:
            st.table(pd.DataFrame(rows))

        if st.button("Save Deployment Config", type="primary", key="save_deploy"):
            write_deployment_config(DeploymentConfig(
                version=(config.version if config else 0),
                updated_at="",
                strategies=updated,
            ))
            st.success("Saved. Run the pipeline to apply.")
            st.cache_resource.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Drift Monitor
# ══════════════════════════════════════════════════════════════════════════════

with tab_drift:
    import polars as pl
    from config.settings import DATA_DIR

    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"

    st.markdown("### Allocation Drift Monitor")
    st.caption("Compares latest target weights against actual weights after price drift since last rebalance.")

    if not tw_path.exists():
        st.info("No target weights found. Run `orchestration/generate_signals.py` first.")
    else:
        tw = pl.read_parquet(tw_path)
        latest_date = tw["date"].max()
        target = tw.filter(pl.col("date") == latest_date).sort("weight", descending=True)
        target_pd = target.to_pandas().set_index("symbol")["weight"]

        actual_pd = target_pd.copy()
        try:
            from storage.parquet_store import load_bars
            from datetime import date as _date
            syms   = target_pd.index.tolist()
            prices = load_bars(syms, latest_date, _date.today(), "equity")
            if not prices.is_empty():
                pivot = prices.sort("date").to_pandas().pivot(index="date", columns="symbol", values="close")
                if len(pivot) >= 2:
                    growth = pivot.iloc[-1] / pivot.iloc[0]
                    drifted = target_pd * growth.reindex(target_pd.index).fillna(1.0)
                    actual_pd = drifted / drifted.sum()
        except Exception:
            pass

        drift = actual_pd - target_pd
        drift_sorted = drift.sort_values()
        colors = [COLORS["negative"] if v < -0.02 else COLORS["warning"] if v < 0
                  else COLORS["positive"] if v > 0.02 else COLORS["neutral"] for v in drift_sorted.values]

        c_drift, c_weights = st.columns(2)
        with c_drift:
            st.markdown("**Drift (Actual − Target)**")
            fig_d = go.Figure(go.Bar(
                x=drift_sorted.values, y=drift_sorted.index.tolist(), orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:+.2%}" for v in drift_sorted.values], textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{y}</b>: %{x:+.2%}<extra></extra>",
            ))
            fig_d.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
            fig_d.add_vline(x=0.05,  line=dict(color=COLORS["warning"], width=1, dash="dot"))
            fig_d.add_vline(x=-0.05, line=dict(color=COLORS["warning"], width=1, dash="dot"))
            apply_theme(fig_d)
            fig_d.update_layout(height=max(200, 38 * len(drift)),
                                  xaxis=dict(tickformat=".1%", showgrid=False),
                                  yaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)

        with c_weights:
            st.markdown("**Target vs Actual Weights**")
            all_syms = sorted(set(target_pd.index) | set(actual_pd.index))
            df_cmp = pd.DataFrame({
                "Symbol": all_syms,
                "Target": [f"{target_pd.get(s, 0):.2%}" for s in all_syms],
                "Actual": [f"{actual_pd.get(s, 0):.2%}" for s in all_syms],
                "Drift":  [f"{(actual_pd.get(s, 0) - target_pd.get(s, 0)):+.2%}" for s in all_syms],
            })
            st.dataframe(df_cmp, hide_index=True, use_container_width=True,
                          height=max(200, 38 * len(all_syms)))

        max_drift = float(drift.abs().max()) if not drift.empty else 0
        if max_drift > 0.05:
            st.warning(f"Maximum drift is {max_drift:.1%} — consider rebalancing.")
        elif max_drift > 0.02:
            st.info(f"Maximum drift is {max_drift:.1%} — within tolerance.")
        else:
            st.success(f"Portfolio is on target (max drift {max_drift:.1%}).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trade
# ══════════════════════════════════════════════════════════════════════════════

with tab_trade:
    from config.settings import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
    import polars as pl
    from config.settings import DATA_DIR

    _IB_PORTS = {
        7497: "TWS — paper",
        7496: "TWS — live",
        4002: "IB Gateway — paper",
        4001: "IB Gateway — live",
    }

    def _tcp_probe(host: str, port: int, timeout: float = 1.5) -> bool:
        import socket
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _auto_detect_ib(host: str) -> dict[int, str]:
        return {p: label for p, label in _IB_PORTS.items() if _tcp_probe(host, p)}

    def _show_connection_help(host: str, port) -> None:
        port_str = f":{port}" if port else ""
        st.error(f"Nothing found at **{host}{port_str}**")
        st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['warning']};
     border-radius:8px;padding:16px 20px;margin-top:8px;line-height:1.9;">
  <div style="color:{COLORS['warning']};font-weight:700;font-size:0.9rem;margin-bottom:10px;">Setup checklist</div>
  <ol style="color:{COLORS['text']};font-size:0.85rem;margin:0;padding-left:20px;">
    <li>Open <strong>TWS</strong> or <strong>IB Gateway</strong> and log in.</li>
    <li>Enable API: <em>Edit → Global Configuration → API → Settings</em></li>
    <li>Check <strong>"Enable ActiveX and Socket Clients"</strong>.</li>
    <li>Set Socket port: TWS paper <code>7497</code> · TWS live <code>7496</code> · Gateway paper <code>4002</code> · Gateway live <code>4001</code></li>
    <li>Uncheck <strong>"Read-Only API"</strong>.</li>
    <li>Add <code>127.0.0.1</code> to Trusted IP Addresses. Click OK / Apply.</li>
  </ol>
</div>""", unsafe_allow_html=True)

    st.markdown("### Interactive Brokers Execution")

    with st.expander("Connection Settings", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            host = st.text_input("Host", value=str(IBKR_HOST), key="ib_host")
        with col_b:
            port = st.number_input("Port", value=int(IBKR_PORT), min_value=1, max_value=65535, key="ib_port")
        with col_c:
            client_id = st.number_input("Client ID", value=int(IBKR_CLIENT_ID), min_value=0, max_value=999, key="ib_client_id")

        st.caption("TWS paper **7497** · TWS live **7496** · IB Gateway paper **4002** · IB Gateway live **4001**")

        btn_test, btn_detect = st.columns(2)
        with btn_test:
            if st.button("Test Connection", key="test_ib_conn", use_container_width=True):
                if _tcp_probe(host, int(port)):
                    st.success(f"Reachable at {host}:{port}")
                else:
                    _show_connection_help(host, int(port))
        with btn_detect:
            if st.button("Auto-Detect Ports", key="detect_ib", use_container_width=True):
                with st.spinner(f"Scanning {host}…"):
                    found = _auto_detect_ib(host)
                if found:
                    labels = ", ".join(f"**{p}** ({lbl})" for p, lbl in found.items())
                    st.success(f"Found IB listening on: {labels}")
                    st.info("Update the Port field above.")
                else:
                    _show_connection_help(host, None)

    st.markdown("### Trading Mode")
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        mode    = st.radio("Select mode", ["Paper Trading", "Live Trading"], key="ib_mode")
        dry_run = st.checkbox("Dry run (compute orders only, do not place)", value=True, key="ib_dry_run")
    is_paper = (mode == "Paper Trading")
    with col_info:
        if is_paper:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['positive']};border-radius:8px;padding:14px 18px;">
  <div style="color:{COLORS['positive']};font-weight:700;font-size:0.9rem;">PAPER TRADING</div>
  <div style="color:{COLORS['neutral']};font-size:0.82rem;margin-top:6px;">
    Orders placed against your IB paper account. No real money at risk.</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['negative']};border-radius:8px;padding:14px 18px;">
  <div style="color:{COLORS['negative']};font-weight:800;font-size:0.9rem;">LIVE TRADING — REAL MONEY</div>
  <div style="color:{COLORS['neutral']};font-size:0.82rem;margin-top:6px;">
    Orders placed on your <strong>live IB account</strong>. Review target weights before proceeding.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("### Pre-Flight Check")
    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
    if not tw_path.exists():
        st.error("No target weights found. Run `orchestration/generate_signals.py` first.")
    else:
        tw = pl.read_parquet(tw_path)
        latest_date = tw["date"].max()
        latest = tw.filter(pl.col("date") == latest_date).sort("weight", descending=True)
        st.caption(f"Target weights as of **{latest_date}** (run generate_signals to refresh):")
        st.dataframe(latest.to_pandas().reset_index(drop=True), use_container_width=True)
        from datetime import date
        age_days = (date.today() - latest_date).days if latest_date else 999
        if age_days > 5:
            st.warning(f"Target weights are {age_days} days old. Consider regenerating signals first.")

        st.markdown("### Execute")
        confirm_live = True
        if not is_paper:
            confirm_live = st.checkbox("I confirm this will place REAL orders on my live IB account",
                                        value=False, key="ib_live_confirm")

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            btn_label    = "Execute Rebalance" if not dry_run else "Compute Orders (Dry Run)"
            btn_disabled = not is_paper and not confirm_live
            execute = st.button(btn_label, type="primary", disabled=btn_disabled, key="ib_execute_btn")

        if execute:
            mode_str = "paper" if is_paper else "LIVE"
            dry_str  = " (DRY RUN)" if dry_run else ""
            st.markdown(f"**Running IB rebalance — {mode_str}{dry_str}**")
            cmd = [sys.executable, str(_ROOT / "orchestration" / "rebalance.py"),
                   "--broker", "ibkr", "--ibkr-host", host,
                   "--ibkr-port", str(port), "--ibkr-client-id", str(client_id)]
            if not is_paper:
                cmd.append("--ibkr-live")
            if dry_run:
                cmd.append("--dry-run")

            log_box = st.empty()
            lines: list[str] = []
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, cwd=str(_ROOT))
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    log_box.code("\n".join(lines[-60:]), language="")
                proc.wait(timeout=300)
                rc = proc.returncode
                if rc == 0:
                    st.success("Rebalance completed successfully.")
                elif rc == 1:
                    st.warning("Finished with warnings (pre-trade check or drift). Check output.")
                else:
                    st.error(f"Rebalance failed (exit code {rc}).")
            except FileNotFoundError:
                st.error("Could not find `orchestration/rebalance.py`.")
            except subprocess.TimeoutExpired:
                proc.kill()
                st.error("Rebalance timed out after 5 minutes.")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
