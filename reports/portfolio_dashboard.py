"""Portfolio Management Dashboard — multi-strategy control centre.

Tabs:
  1. Overview        — deployed strategies, combined equity, live allocation pie
  2. Comparison      — side-by-side metrics, overlaid equity curves, drawdown
  3. Optimizer       — correlation heatmap, allocation optimizer, one-click deploy
  4. Deployment      — active/inactive toggles, weight sliders, save config
  5. Blended Preview — current combined symbol positions
  6. Trade           — IB paper / live execution
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import COLORS, apply_theme, badge

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _color(i: int) -> str:
    return COLORS["series"][i % len(COLORS["series"])]


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _pct(v, decimals=1):
    return f"{v * 100:+.{decimals}f}%" if v is not None else "—"


def _fmt(v, pct=True, decimals=1):
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%" if pct else f"{v:.{decimals}f}"


def _equity_series(result) -> pd.Series:
    idx = pd.to_datetime(result.equity_dates)
    return pd.Series(result.equity_values, index=idx, name=result.slug)


def _normalise(s: pd.Series) -> pd.Series:
    v0 = s.iloc[0]
    return s / v0 * 10_000 if v0 else s


def _drawdown(s: pd.Series) -> pd.Series:
    peak = s.cummax()
    return (s - peak) / peak


# ── Data loaders (cache_resource avoids Streamlit hashing complex objects) ────

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


def _run_backtests(metas, force=False):
    from portfolio.multi_strategy import get_or_run_backtest
    results = {}
    prog = st.progress(0, text="Running backtests…")
    n = max(len(metas), 1)
    for i, m in enumerate(metas):
        prog.progress(i / n, text=f"  {m.name} ({i + 1}/{n})")
        try:
            r = get_or_run_backtest(m, force=force)
            results[m.slug] = r
        except Exception as exc:
            st.warning(f"Backtest failed for **{m.name}**: {exc}")
    prog.progress(1.0, text="Done.")
    time.sleep(0.3)
    prog.empty()
    return results


_METRIC_KEYS = [
    ("cagr",         "CAGR",         True),
    ("sharpe",       "Sharpe",       False),
    ("max_drawdown", "Max DD",       True),
    ("sortino",      "Sortino",      False),
    ("total_return", "Total Return", True),
    ("calmar",       "Calmar",       False),
]


def _metric_table(results: dict) -> pd.DataFrame:
    rows = []
    for slug, r in results.items():
        m = r.metrics
        row = {"Strategy": r.name}
        for key, label, is_pct in _METRIC_KEYS:
            v = m.get(key)
            row[label] = _fmt(v, pct=is_pct, decimals=2 if not is_pct else 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_overview(results: dict, config):
    from portfolio.multi_strategy import build_return_matrix

    if not results:
        st.info("No backtest results yet — click **Run Backtests** above.")
        return

    active_slugs = (
        {s.slug for s in config.strategies if s.active}
        if config else set(results.keys())
    )
    active_results = {k: v for k, v in results.items() if k in active_slugs}

    if not active_results:
        st.warning("No active strategies — configure deployment in the **Deployment** tab.")
        return

    # Build allocation weights
    alloc: dict[str, float] = {}
    if config:
        for s in config.strategies:
            if s.active and s.slug in active_results:
                alloc[s.slug] = s.allocation_weight
    if not alloc:
        n = len(active_results)
        alloc = {slug: 1.0 / n for slug in active_results}

    total_alloc = sum(alloc.values()) or 1.0
    alloc_norm = {k: v / total_alloc for k, v in alloc.items()}

    # Blended equity curve
    ret_matrix = build_return_matrix(list(active_results.values()))
    blended_eq = pd.Series(dtype=float)
    if not ret_matrix.empty:
        w = np.array([alloc_norm.get(c, 0.0) for c in ret_matrix.columns])
        if w.sum() > 1e-10:
            w /= w.sum()
        blended_ret = (ret_matrix * w).sum(axis=1)
        blended_eq = 10_000.0 * (1 + blended_ret).cumprod()

    # KPI cards — one per strategy + pie chart
    n_cards = len(active_results)
    card_cols = st.columns(n_cards + 1)

    for i, (slug, r) in enumerate(active_results.items()):
        alloc_pct = alloc_norm.get(slug, 0.0)
        m = r.metrics
        color = _color(i)
        cagr_str = _pct(m.get("cagr"))
        sharpe_str = f"{m.get('sharpe', 0):.2f}"
        dd_str = _pct(m.get("max_drawdown"))
        with card_cols[i]:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
     border-left:3px solid {color};border-radius:8px;padding:14px 16px;margin-bottom:8px;">
  <div style="font-size:0.72rem;color:{COLORS['text_muted']};text-transform:uppercase;
       letter-spacing:0.07em;margin-bottom:6px;">{r.name}</div>
  <div style="font-size:1.6rem;font-weight:800;color:{color};line-height:1;">
    {int(alloc_pct * 100)}%</div>
  <div style="font-size:0.72rem;color:{COLORS['neutral']};margin-top:4px;">allocation</div>
  <div style="border-top:1px solid {COLORS['border_dim']};margin:10px 0 8px;"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{cagr_str}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">CAGR</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{sharpe_str}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Sharpe</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['negative']};">{dd_str}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Max DD</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with card_cols[-1]:
        labels = [active_results[s].name for s in alloc_norm if s in active_results]
        values = [alloc_norm[s] for s in alloc_norm if s in active_results]
        colors = [_color(i) for i in range(len(labels))]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label+percent", textfont=dict(size=11, color=COLORS["text"]),
            hole=0.55,
        ))
        apply_theme(fig_pie, title="Allocation", height=220)
        fig_pie.update_layout(showlegend=False, paper_bgcolor=COLORS["card_bg"])
        st.plotly_chart(fig_pie, use_container_width=True)

    # Blended equity chart
    if not blended_eq.empty:
        fig = go.Figure()
        for i, (slug, r) in enumerate(active_results.items()):
            eq = _normalise(_equity_series(r))
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values, name=r.name,
                line=dict(color=_color(i), width=1.5, dash="dot"), opacity=0.65,
            ))
        fig.add_trace(go.Scatter(
            x=blended_eq.index, y=blended_eq.values,
            name="Blended Portfolio",
            line=dict(color=COLORS["positive"], width=2.8),
        ))
        first = next(iter(active_results.values()))
        if first.benchmark_dates:
            spy_idx = pd.to_datetime(first.benchmark_dates)
            spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
            fig.add_trace(go.Scatter(
                x=spy_idx, y=spy_vals.values, name="SPY",
                line=dict(color=COLORS["neutral"], width=1.2, dash="dash"), opacity=0.7,
            ))
        apply_theme(fig, title="Blended Portfolio vs Components ($10k start)", height=380)
        fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _tab_comparison(results: dict):
    if not results:
        st.info("No results yet — click **Run Backtests** above.")
        return

    st.markdown("### Performance Metrics")
    df = _metric_table(results)
    if not df.empty:
        st.dataframe(df, use_container_width=True)

    st.markdown("### Equity Curves (normalised to $10,000)")
    fig_eq = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        eq = _normalise(_equity_series(r))
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=r.name,
            line=dict(color=_color(i), width=2),
        ))
    first = next(iter(results.values()))
    if first.benchmark_dates:
        spy_idx = pd.to_datetime(first.benchmark_dates)
        spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
        fig_eq.add_trace(go.Scatter(
            x=spy_idx, y=spy_vals.values, name="SPY",
            line=dict(color=COLORS["neutral"], width=1.5, dash="dash"), opacity=0.8,
        ))
    apply_theme(fig_eq, height=400)
    fig_eq.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("### Drawdown")
    fig_dd = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        dd = _drawdown(_equity_series(r)) * 100
        fill_color = _hex_rgba(_color(i), 0.08)
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=r.name,
            line=dict(color=_color(i), width=1.8),
            fill="tozeroy", fillcolor=fill_color,
        ))
    apply_theme(fig_dd, height=260)
    fig_dd.update_layout(yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("### Rolling 12-Month Sharpe")
    fig_sh = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        daily_ret = _equity_series(r).pct_change().dropna()
        roll_sh = (
            daily_ret.rolling(252).mean() /
            daily_ret.rolling(252).std().replace(0.0, np.nan)
        ) * np.sqrt(252)
        fig_sh.add_trace(go.Scatter(
            x=roll_sh.index, y=roll_sh.values, name=r.name,
            line=dict(color=_color(i), width=1.8),
        ))
    fig_sh.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    apply_theme(fig_sh, height=260)
    st.plotly_chart(fig_sh, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def _tab_optimizer(results: dict, config, metas: list):
    from portfolio.multi_strategy import (
        build_return_matrix, strategy_correlation, optimize_allocations,
        write_deployment_config, read_deployment_config,
        DeploymentConfig, DeployedStrategy, discover_strategies,
    )

    if len(results) < 2:
        st.info("Need at least 2 strategy backtests to run the optimizer.")
        return

    ret_matrix = build_return_matrix(list(results.values()))
    if ret_matrix.empty or len(ret_matrix.columns) < 2:
        st.warning("Insufficient overlapping return history to compute correlations.")
        return

    # Correlation heatmap
    corr = strategy_correlation(ret_matrix)
    slug_to_name = {slug: r.name for slug, r in results.items()}
    names = [slug_to_name.get(c, c) for c in corr.columns]

    st.markdown("### Strategy Correlation Matrix")
    z_text = [[f"{v:.2f}" for v in row] for row in corr.values]
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names,
        colorscale=[[0, COLORS["negative"]], [0.5, COLORS["card_bg"]], [1, COLORS["positive"]]],
        zmin=-1, zmax=1,
        text=z_text, texttemplate="%{text}",
        textfont=dict(size=13, color=COLORS["text"]),
        showscale=True,
        colorbar=dict(tickfont=dict(color=COLORS["neutral"])),
    ))
    apply_theme(fig_corr, height=300)
    fig_corr.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Per-strategy stats
    ann_ret = ret_matrix.mean() * 252
    ann_vol = ret_matrix.std() * np.sqrt(252)
    ann_sharpe = ann_ret / ann_vol.replace(0.0, np.nan)

    st.markdown("### Individual Strategy Stats (annualised)")
    n_cols = max(len(ret_matrix.columns), 1)
    stat_cols = st.columns(n_cols)
    for i, col in enumerate(ret_matrix.columns):
        name = slug_to_name.get(col, col)
        ret_str = f"{ann_ret[col] * 100:+.1f}%"
        vol_str = f"{ann_vol[col] * 100:.1f}%"
        sh_str = f"{ann_sharpe[col]:.2f}" if not np.isnan(ann_sharpe[col]) else "—"
        with stat_cols[i]:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
     border-left:3px solid {_color(i)};border-radius:8px;padding:12px 14px;">
  <div style="font-size:0.7rem;color:{COLORS['text_muted']};text-transform:uppercase;">{name}</div>
  <div style="font-size:0.84rem;font-weight:700;color:{COLORS['text']};margin-top:6px;">
    Ret {ret_str} · Vol {vol_str} · Sharpe {sh_str}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # Optimizer controls
    st.markdown("### Optimize Allocations")
    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.selectbox(
            "Optimization method",
            ["min_variance", "max_sharpe", "vol_scaled", "equal"],
            format_func=lambda x: {
                "min_variance": "Minimum Variance",
                "max_sharpe": "Maximum Sharpe",
                "vol_scaled": "Inverse-Vol Weighted",
                "equal": "Equal Weight",
            }[x],
        )
        max_w = st.slider("Max weight per strategy", 0.30, 1.00, 0.80, 0.05)
    with c2:
        if st.button("Run Optimizer", type="primary", key="run_optimizer"):
            with st.spinner("Optimizing…"):
                try:
                    allocs = optimize_allocations(ret_matrix, method=method, max_weight=max_w)
                    st.session_state["opt_result"] = allocs
                    st.session_state["opt_method"] = method
                except Exception as exc:
                    st.error(f"Optimizer failed: {exc}")

    if "opt_result" in st.session_state:
        allocs = st.session_state["opt_result"]
        lbl = st.session_state.get("opt_method", "")
        st.markdown(f"**Result ({lbl}):**")
        bar_names = [slug_to_name.get(s, s) for s in allocs]
        bar_vals = [v * 100 for v in allocs.values()]
        bar_colors = [_color(i) for i in range(len(allocs))]
        fig_bar = go.Figure(go.Bar(
            x=bar_names, y=bar_vals,
            marker=dict(color=bar_colors, line=dict(color=COLORS["bg"], width=1)),
            text=[f"{v:.1f}%" for v in bar_vals], textposition="outside",
        ))
        apply_theme(fig_bar, height=260)
        fig_bar.update_layout(yaxis=dict(ticksuffix="%", range=[0, 110]))
        st.plotly_chart(fig_bar, use_container_width=True)

        if st.button("Deploy These Allocations", type="primary", key="deploy_opt"):
            _save_allocations(allocs, results, config, metas)
            st.success("Deployment config saved.")
            st.cache_resource.clear()
            st.rerun()


def _save_allocations(allocs: dict, results: dict, config, metas: list):
    from portfolio.multi_strategy import (
        write_deployment_config, read_deployment_config,
        DeploymentConfig, DeployedStrategy,
    )
    existing = config or read_deployment_config()
    old_params = {s.slug: s.backtest_params for s in existing.strategies} if existing else {}
    meta_map = {m.slug: m for m in metas}
    strategies = []
    for slug, weight in allocs.items():
        m = meta_map.get(slug)
        strategies.append(DeployedStrategy(
            slug=slug,
            name=results[slug].name if slug in results else slug,
            active=True,
            allocation_weight=round(weight, 6),
            backtest_params=old_params.get(slug, m.default_params if m else {}),
        ))
    write_deployment_config(DeploymentConfig(
        version=(existing.version if existing else 0),
        updated_at="",
        strategies=strategies,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deployment
# ══════════════════════════════════════════════════════════════════════════════

def _tab_deployment(metas: list, results: dict, config):
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
        return

    existing_map = {s.slug: s for s in config.strategies} if config else {}
    updated: list[DeployedStrategy] = []

    for i, m in enumerate(metas):
        ex = existing_map.get(m.slug)
        default_active = ex.active if ex else True
        default_weight = ex.allocation_weight if ex else round(1.0 / len(metas), 3)
        default_params = ex.backtest_params if ex else m.default_params
        has_result = m.slug in results
        cache_str = "cached" if has_result else "no data"

        with st.expander(f"{m.name}  [{cache_str}]", expanded=True):
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
                    st.caption("Run backtests in the Comparison tab first.")

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
# TAB 5 — Blended Preview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_blended(config, results: dict):
    from config.settings import DATA_DIR
    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"

    st.markdown("### Live Target Weights")
    if tw_path.exists():
        import polars as pl
        tw = pl.read_parquet(tw_path)
        latest_date = tw["date"].max()
        latest = tw.filter(pl.col("date") == latest_date).sort("weight", descending=True)
        st.caption(f"As of {latest_date}")
        st.dataframe(latest.to_pandas().reset_index(drop=True), use_container_width=True)

        rows_pd = latest.to_pandas()
        if not rows_pd.empty:
            fig_pie = go.Figure(go.Pie(
                labels=rows_pd["symbol"].tolist(),
                values=rows_pd["weight"].tolist(),
                marker=dict(colors=COLORS["series"], line=dict(color=COLORS["bg"], width=2)),
                textinfo="label+percent", textfont=dict(size=11, color=COLORS["text"]),
                hole=0.5,
            ))
            apply_theme(fig_pie, title="Live Position Weights", height=320)
            fig_pie.update_layout(showlegend=False, paper_bgcolor=COLORS["card_bg"])
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No target weights found. Run `orchestration/generate_signals.py` first.")

    if config:
        st.markdown("### Active Deployment Config")
        total = sum(s.allocation_weight for s in config.strategies if s.active) or 1.0
        rows = [
            {
                "Strategy": s.name,
                "Active": "Yes" if s.active else "No",
                "Raw": f"{s.allocation_weight:.3f}",
                "Normalised": f"{s.allocation_weight / total * 100:.1f}%" if s.active else "—",
            }
            for s in config.strategies
        ]
        st.table(pd.DataFrame(rows))
        st.caption(f"Version {config.version} · updated {config.updated_at or 'unknown'}")
    else:
        st.info("No deployment config yet — go to the **Deployment** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Trade (IB paper / live execution)
# ══════════════════════════════════════════════════════════════════════════════

def _show_connection_help(host: str, port: int | None) -> None:
    """Render a clear setup guide when IB can't be reached."""
    port_str = f":{port}" if port else ""
    st.error(f"Nothing found at **{host}{port_str}**")
    st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['warning']};
     border-radius:8px;padding:16px 20px;margin-top:8px;line-height:1.9;">
  <div style="color:{COLORS['warning']};font-weight:700;font-size:0.9rem;margin-bottom:10px;">
    Setup checklist
  </div>
  <ol style="color:{COLORS['text']};font-size:0.85rem;margin:0;padding-left:20px;">
    <li>Open <strong>TWS</strong> or <strong>IB Gateway</strong> and log in to your account.</li>
    <li>In TWS: <em>Edit → Global Configuration → API → Settings</em><br>
        In Gateway: <em>Configure → Settings → API → Settings</em></li>
    <li>Check <strong>"Enable ActiveX and Socket Clients"</strong>.</li>
    <li>Set <em>Socket port</em>:
      <ul style="margin:4px 0;">
        <li>TWS paper&nbsp;&nbsp; → <code>7497</code></li>
        <li>TWS live&nbsp;&nbsp;&nbsp; → <code>7496</code></li>
        <li>Gateway paper → <code>4002</code></li>
        <li>Gateway live&nbsp; → <code>4001</code></li>
      </ul>
    </li>
    <li>Uncheck <strong>"Read-Only API"</strong> (required to place orders).</li>
    <li>Make sure <em>Trusted IP Addresses</em> includes <code>127.0.0.1</code>.</li>
    <li>Click <strong>OK / Apply</strong>, then use <em>Auto-Detect Ports</em> here.</li>
  </ol>
</div>""", unsafe_allow_html=True)


# Standard IB ports: TWS paper, TWS live, Gateway paper, Gateway live
_IB_PORTS = {
    7497: "TWS — paper",
    7496: "TWS — live",
    4002: "IB Gateway — paper",
    4001: "IB Gateway — live",
}


def _tcp_probe(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if something is listening on host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _auto_detect_ib(host: str) -> dict[int, str]:
    """Return {port: label} for every IB port that answers."""
    return {p: label for p, label in _IB_PORTS.items() if _tcp_probe(host, p)}


def _tab_trade():
    from config.settings import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID

    st.markdown("### Interactive Brokers Execution")

    # ── Connection settings ───────────────────────────────────────────────────
    with st.expander("Connection Settings", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            host = st.text_input("Host", value=str(IBKR_HOST), key="ib_host")
        with col_b:
            port = st.number_input("Port", value=int(IBKR_PORT), min_value=1, max_value=65535,
                                   key="ib_port")
        with col_c:
            client_id = st.number_input("Client ID", value=int(IBKR_CLIENT_ID), min_value=0,
                                        max_value=999, key="ib_client_id")

        st.caption(
            "TWS paper **7497** · TWS live **7496** · "
            "IB Gateway paper **4002** · IB Gateway live **4001**"
        )

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
                    st.info("Update the Port field above to one of these values.")
                else:
                    _show_connection_help(host, None)

    # ── Trading mode ──────────────────────────────────────────────────────────
    st.markdown("### Trading Mode")
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        mode = st.radio(
            "Select mode",
            ["Paper Trading", "Live Trading"],
            key="ib_mode",
            help="Paper trading uses your IB paper account (no real money). "
                 "Live trading places real orders.",
        )
        dry_run = st.checkbox("Dry run (compute orders only, do not place)", value=True,
                              key="ib_dry_run")

    is_paper = (mode == "Paper Trading")

    with col_info:
        if is_paper:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['positive']};
     border-radius:8px;padding:14px 18px;">
  <div style="color:{COLORS['positive']};font-weight:700;font-size:0.9rem;">
    PAPER TRADING</div>
  <div style="color:{COLORS['neutral']};font-size:0.82rem;margin-top:6px;">
    Orders are placed against your IB paper account. No real money at risk.
    Make sure you are connected to the <strong>paper session</strong> in TWS/Gateway.</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['negative']};
     border-radius:8px;padding:14px 18px;">
  <div style="color:{COLORS['negative']};font-weight:800;font-size:0.9rem;">
    LIVE TRADING — REAL MONEY</div>
  <div style="color:{COLORS['neutral']};font-size:0.82rem;margin-top:6px;">
    Orders will be placed on your <strong>live IB account</strong>.
    Make sure you are connected to the <strong>live session</strong> in TWS/Gateway.
    Target weights must be reviewed before proceeding.</div>
</div>""", unsafe_allow_html=True)

    # ── Pre-flight: show target weights ───────────────────────────────────────
    st.markdown("### Pre-Flight Check")
    from config.settings import DATA_DIR
    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"

    if not tw_path.exists():
        st.error("No target weights found. Run `orchestration/generate_signals.py` first.")
        return

    import polars as pl
    tw = pl.read_parquet(tw_path)
    latest_date = tw["date"].max()
    latest = tw.filter(pl.col("date") == latest_date).sort("weight", descending=True)
    st.caption(f"Target weights as of **{latest_date}** (run generate_signals to refresh):")
    st.dataframe(latest.to_pandas().reset_index(drop=True), use_container_width=True)

    from datetime import date
    age_days = (date.today() - latest_date).days if latest_date else 999
    if age_days > 5:
        st.warning(f"Target weights are {age_days} days old. Consider regenerating signals first.")

    # ── Execute rebalance ─────────────────────────────────────────────────────
    st.markdown("### Execute")

    confirm_live = True
    if not is_paper:
        confirm_live = st.checkbox(
            "I confirm this will place REAL orders on my live IB account",
            value=False, key="ib_live_confirm",
        )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        btn_label = "Execute Rebalance" if not dry_run else "Compute Orders (Dry Run)"
        btn_disabled = not is_paper and not confirm_live
        execute = st.button(btn_label, type="primary", disabled=btn_disabled,
                            key="ib_execute_btn")

    if execute:
        _run_ibkr_rebalance(
            host=host, port=int(port), client_id=int(client_id),
            is_paper=is_paper, dry_run=dry_run,
        )


def _run_ibkr_rebalance(host: str, port: int, client_id: int,
                        is_paper: bool, dry_run: bool) -> None:
    """Run rebalance.py as a subprocess and stream output."""
    mode_str = "paper" if is_paper else "LIVE"
    dry_str = " (DRY RUN)" if dry_run else ""
    st.markdown(f"**Running IB rebalance — {mode_str}{dry_str}**")

    cmd = [
        sys.executable,
        str(_ROOT / "orchestration" / "rebalance.py"),
        "--broker", "ibkr",
        "--ibkr-host", host,
        "--ibkr-port", str(port),
        "--ibkr-client-id", str(client_id),
    ]
    if not is_paper:
        cmd.append("--ibkr-live")
    if dry_run:
        cmd.append("--dry-run")

    log_box = st.empty()
    lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_ROOT),
        )
        for line in proc.stdout:
            lines.append(line.rstrip())
            log_box.code("\n".join(lines[-60:]), language="")

        proc.wait(timeout=300)
        rc = proc.returncode
        if rc == 0:
            st.success("Rebalance completed successfully.")
        elif rc == 1:
            st.warning("Rebalance finished with warnings (pre-trade check or drift). Check output above.")
        else:
            st.error(f"Rebalance failed (exit code {rc}). Check output above.")
    except FileNotFoundError:
        st.error("Could not find `orchestration/rebalance.py`. Check your project structure.")
    except subprocess.TimeoutExpired:
        proc.kill()
        st.error("Rebalance timed out after 5 minutes.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Portfolio Management")
st.caption("Compare strategies, optimise allocations, and control execution.")

metas = _discover()
config = _load_config()

if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = {}
results: dict = st.session_state["backtest_results"]

# Top controls
ctrl_left, ctrl_right = st.columns([3, 1])
with ctrl_left:
    if metas:
        st.markdown("**" + str(len(metas)) + " strategies found** — " +
                    ", ".join(f"`{m.slug}`" for m in metas))
    else:
        st.warning("No strategies found in `strategies/`.")

with ctrl_right:
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("Run Backtests", use_container_width=True, type="primary",
                     key="top_run"):
            new_r = _run_backtests(metas, force=False)
            st.session_state["backtest_results"].update(new_r)
            st.rerun()
    with rc2:
        if st.button("Force Refresh", use_container_width=True, key="top_force"):
            new_r = _run_backtests(metas, force=True)
            st.session_state["backtest_results"] = new_r
            st.rerun()

st.markdown("---")

(tab_overview, tab_compare, tab_optimizer,
 tab_deploy, tab_blended, tab_trade) = st.tabs([
    "Overview", "Comparison", "Optimizer",
    "Deployment", "Blended Preview", "Trade",
])

with tab_overview:
    _tab_overview(results, config)

with tab_compare:
    _tab_comparison(results)

with tab_optimizer:
    _tab_optimizer(results, config, metas)

with tab_deploy:
    _tab_deployment(metas, results, config)

with tab_blended:
    _tab_blended(config, results)

with tab_trade:
    _tab_trade()
