"""Portfolio Management Dashboard — multi-strategy control centre.

Tabs:
  1. Overview        — deployed strategies, combined equity, live allocation pie
  2. Comparison      — side-by-side metrics, overlaid equity curves, drawdown
  3. Optimizer       — correlation heatmap, allocation optimizer, one-click deploy
  4. Deployment      — active/inactive toggles, weight sliders, save config
  5. Blended Preview — current combined symbol positions
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import COLORS, apply_theme as apply_layout, badge

log = logging.getLogger(__name__)

# ── Lazy imports (heavy, only when tab is opened) ──────────────────────────────

@st.cache_data(ttl=30)
def _discover():
    from portfolio.multi_strategy import discover_strategies
    return discover_strategies()


def _run_backtests(metas, force=False):
    from portfolio.multi_strategy import get_or_run_backtest
    results = {}
    n = len(metas)
    prog = st.progress(0, text="Running backtests…")
    for i, m in enumerate(metas):
        prog.progress((i) / n, text=f"Backtest: {m.name} ({i+1}/{n})")
        try:
            r = get_or_run_backtest(m, force=force)
            results[m.slug] = r
        except Exception as exc:
            st.warning(f"Backtest failed for **{m.name}**: {exc}")
    prog.progress(1.0, text="Done.")
    time.sleep(0.4)
    prog.empty()
    return results


@st.cache_data(ttl=120)
def _load_config():
    from portfolio.multi_strategy import read_deployment_config
    return read_deployment_config()


# ── Colour helpers ─────────────────────────────────────────────────────────────

_SERIES = COLORS["series"]


def _color(i: int) -> str:
    return _SERIES[i % len(_SERIES)]


def _pct(v, decimals=1):
    return f"{v*100:+.{decimals}f}%" if v is not None else "—"


def _fmt_metric(v, pct=True, decimals=1):
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.{decimals}f}%"
    return f"{v:.{decimals}f}"


# ── Equity-curve helper ────────────────────────────────────────────────────────

def _equity_series(result) -> pd.Series:
    idx = pd.to_datetime(result.equity_dates)
    return pd.Series(result.equity_values, index=idx, name=result.slug)


def _normalise(series: pd.Series) -> pd.Series:
    v0 = series.iloc[0]
    return series / v0 * 10_000 if v0 else series


def _drawdown(series: pd.Series) -> pd.Series:
    roll_max = series.cummax()
    return (series - roll_max) / roll_max


# ── Metric card row ────────────────────────────────────────────────────────────

_METRIC_KEYS = [
    ("cagr",          "CAGR",          True),
    ("sharpe",        "Sharpe",        False),
    ("max_drawdown",  "Max DD",        True),
    ("sortino",       "Sortino",       False),
    ("total_return",  "Total Return",  True),
    ("calmar",        "Calmar",        False),
]


def _metric_table(results: dict) -> pd.DataFrame:
    rows = []
    for slug, r in results.items():
        m = r.metrics
        row = {"Strategy": r.name}
        for key, label, is_pct in _METRIC_KEYS:
            v = m.get(key)
            row[label] = _fmt_metric(v, pct=is_pct, decimals=2 if not is_pct else 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_overview(results: dict, config):
    if not results:
        st.info("No backtest results available. Run backtests in the Comparison tab.")
        return

    active_slugs = (
        {s.slug for s in config.strategies if s.active}
        if config else set(results.keys())
    )
    active_results = {k: v for k, v in results.items() if k in active_slugs}

    st.markdown("### Deployed Portfolio")
    if not active_results:
        st.warning("No active strategies. Configure deployment in the **Deployment** tab.")
        return

    # Blended equity curve
    from portfolio.multi_strategy import build_return_matrix, blend_weights

    alloc: dict[str, float] = {}
    if config:
        for s in config.strategies:
            if s.active and s.slug in active_results:
                alloc[s.slug] = s.allocation_weight
    else:
        n = len(active_results)
        alloc = {slug: 1.0 / n for slug in active_results}

    total_alloc = sum(alloc.values()) or 1.0
    alloc_norm = {k: v / total_alloc for k, v in alloc.items()}

    # Build blended equity by weighting daily returns
    ret_matrix = build_return_matrix(list(active_results.values()))
    if not ret_matrix.empty:
        weights_arr = np.array([alloc_norm.get(c, 0.0) for c in ret_matrix.columns])
        weights_arr /= weights_arr.sum()
        blended_ret = (ret_matrix * weights_arr).sum(axis=1)
        blended_eq = 10_000 * (1 + blended_ret).cumprod()
    else:
        blended_eq = pd.Series(dtype=float)

    # KPI cards
    cols = st.columns(len(active_results) + 1)
    for i, (slug, r) in enumerate(active_results.items()):
        with cols[i]:
            alloc_pct = alloc_norm.get(slug, 0.0)
            m = r.metrics
            cagr = _pct(m.get("cagr"))
            sharpe = f"{m.get('sharpe', 0):.2f}"
            dd = _pct(m.get("max_drawdown"))
            color = _color(i)
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
            border-left:3px solid {color};border-radius:8px;padding:14px 16px;
            margin-bottom:8px;">
  <div style="font-size:0.72rem;color:{COLORS['text_muted']};text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:6px;">{r.name}</div>
  <div style="font-size:1.6rem;font-weight:800;color:{color};line-height:1;">
    {int(alloc_pct*100)}%
  </div>
  <div style="font-size:0.72rem;color:{COLORS['neutral']};margin-top:4px;">allocation</div>
  <div style="border-top:1px solid {COLORS['border_dim']};margin:10px 0 8px;"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;">
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{cagr}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">CAGR</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{sharpe}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Sharpe</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['negative']};">{dd}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Max DD</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with cols[-1]:
        # Allocation pie
        labels = [active_results[s].name for s in alloc_norm]
        values = [alloc_norm[s] for s in alloc_norm]
        colors = [_color(i) for i in range(len(labels))]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label+percent", textfont=dict(size=11, color=COLORS["text"]),
            hole=0.55,
        ))
        apply_layout(fig, height=220, title="Allocation")
        fig.update_layout(margin=dict(l=0, r=0, t=36, b=0),
                          showlegend=False, paper_bgcolor=COLORS["card_bg"])
        st.plotly_chart(fig, use_container_width=True)

    # Blended equity chart
    if not blended_eq.empty:
        fig2 = go.Figure()
        for i, (slug, r) in enumerate(active_results.items()):
            eq = _normalise(_equity_series(r))
            fig2.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                name=r.name,
                line=dict(color=_color(i), width=1.5, dash="dot"),
                opacity=0.6,
            ))
        fig2.add_trace(go.Scatter(
            x=blended_eq.index, y=blended_eq.values,
            name="Blended Portfolio",
            line=dict(color=COLORS["positive"], width=2.5),
        ))

        # SPY benchmark from first result
        first = next(iter(active_results.values()))
        if first.benchmark_dates:
            spy_idx = pd.to_datetime(first.benchmark_dates)
            spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
            fig2.add_trace(go.Scatter(
                x=spy_idx, y=spy_vals.values,
                name="SPY",
                line=dict(color=COLORS["neutral"], width=1.2, dash="dash"),
                opacity=0.7,
            ))

        apply_layout(fig2, height=380, title="Blended Portfolio vs Components ($10k start)")
        fig2.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _tab_comparison(results: dict, metas: list):
    if not results:
        st.info("No results yet. Use the **Run / Refresh Backtests** button above.")
        return

    # Metrics table
    st.markdown("### Performance Metrics")
    df_metrics = _metric_table(results)
    st.dataframe(df_metrics, use_container_width=True)

    # Equity curves
    st.markdown("### Equity Curves (normalised to $10,000)")
    fig = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        eq = _normalise(_equity_series(r))
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=r.name,
            line=dict(color=_color(i), width=2),
        ))

    # SPY benchmark
    first = next(iter(results.values()), None)
    if first and first.benchmark_dates:
        spy_idx = pd.to_datetime(first.benchmark_dates)
        spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
        fig.add_trace(go.Scatter(
            x=spy_idx, y=spy_vals.values,
            name="SPY (benchmark)",
            line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
            opacity=0.8,
        ))

    apply_layout(fig, height=400, title="")
    fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.markdown("### Drawdown")
    fig_dd = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        eq = _equity_series(r)
        dd = _drawdown(eq) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=r.name,
            line=dict(color=_color(i), width=1.8),
            fill="tozeroy",
            fillcolor=_color(i).replace(")", ",0.08)").replace("rgb", "rgba") if _color(i).startswith("rgb") else _color(i) + "15",
        ))
    apply_layout(fig_dd, height=280, title="")
    fig_dd.update_layout(yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)

    # Rolling Sharpe
    st.markdown("### Rolling 12-Month Sharpe")
    fig_sh = go.Figure()
    for i, (slug, r) in enumerate(results.items()):
        eq = _equity_series(r)
        daily_ret = eq.pct_change().dropna()
        rolling_sh = (
            daily_ret.rolling(252).mean() /
            daily_ret.rolling(252).std().replace(0, np.nan)
        ) * np.sqrt(252)
        fig_sh.add_trace(go.Scatter(
            x=rolling_sh.index, y=rolling_sh.values,
            name=r.name,
            line=dict(color=_color(i), width=1.8),
        ))
    fig_sh.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    apply_layout(fig_sh, height=280, title="")
    st.plotly_chart(fig_sh, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def _tab_optimizer(results: dict, config):
    from portfolio.multi_strategy import (
        build_return_matrix,
        optimize_allocations,
        strategy_correlation,
        write_deployment_config,
        read_deployment_config,
        DeploymentConfig,
        DeployedStrategy,
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
    names = [results.get(c, type("_", (), {"name": c})()).name if hasattr(results.get(c), "name") else c
             for c in corr.columns]

    st.markdown("### Strategy Correlation Matrix")
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=names, y=names,
        colorscale=[[0, COLORS["negative"]], [0.5, COLORS["card_bg"]], [1, COLORS["positive"]]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color=COLORS["text"]),
        showscale=True,
        colorbar=dict(tickfont=dict(color=COLORS["neutral"])),
    ))
    apply_layout(fig_corr, height=300, title="")
    fig_corr.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Annualized stats
    ann_ret = ret_matrix.mean() * 252
    ann_vol = ret_matrix.std() * np.sqrt(252)
    ann_sharpe = ann_ret / ann_vol.replace(0, np.nan)

    st.markdown("### Individual Strategy Stats (annualised)")
    stat_cols = st.columns(len(ret_matrix.columns))
    for i, col in enumerate(ret_matrix.columns):
        with stat_cols[i]:
            name = results[col].name if col in results else col
            st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
            border-left:3px solid {_color(i)};border-radius:8px;padding:12px 14px;">
  <div style="font-size:0.7rem;color:{COLORS['text_muted']};text-transform:uppercase;">{name}</div>
  <div style="font-size:0.85rem;font-weight:700;color:{COLORS['text']};margin-top:6px;">
    Ret: {ann_ret[col]*100:+.1f}%  Vol: {ann_vol[col]*100:.1f}%  Sharpe: {ann_sharpe[col]:.2f}
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
        if st.button("Run Optimizer", type="primary"):
            with st.spinner("Optimizing…"):
                try:
                    allocs = optimize_allocations(ret_matrix, method=method, max_weight=max_w)
                    st.session_state["optimizer_result"] = allocs
                    st.session_state["optimizer_method"] = method
                except Exception as exc:
                    st.error(f"Optimizer failed: {exc}")

    if "optimizer_result" in st.session_state:
        allocs = st.session_state["optimizer_result"]
        method_label = st.session_state.get("optimizer_method", method)

        st.markdown(f"**Result ({method_label}):**")
        fig_alloc = go.Figure(go.Bar(
            x=[results[s].name if s in results else s for s in allocs],
            y=[v * 100 for v in allocs.values()],
            marker=dict(
                color=[_color(i) for i in range(len(allocs))],
                line=dict(color=COLORS["bg"], width=1),
            ),
            text=[f"{v*100:.1f}%" for v in allocs.values()],
            textposition="outside",
        ))
        apply_layout(fig_alloc, height=260, title="")
        fig_alloc.update_layout(yaxis=dict(ticksuffix="%", range=[0, 110]))
        st.plotly_chart(fig_alloc, use_container_width=True)

        if st.button("Deploy These Allocations", type="primary"):
            _deploy_allocations(allocs, results, config)
            st.success("Deployment config saved. Reload the page to see changes.")
            st.cache_data.clear()


def _deploy_allocations(allocs: dict, results: dict, config):
    from portfolio.multi_strategy import (
        write_deployment_config,
        read_deployment_config,
        DeploymentConfig,
        DeployedStrategy,
        discover_strategies,
    )

    existing = config or read_deployment_config()
    existing_params = {}
    if existing:
        for s in existing.strategies:
            existing_params[s.slug] = s.backtest_params

    metas = {m.slug: m for m in discover_strategies()}
    strategies = []
    for slug, weight in allocs.items():
        m = metas.get(slug)
        strategies.append(DeployedStrategy(
            slug=slug,
            name=results[slug].name if slug in results else slug,
            active=True,
            allocation_weight=round(weight, 6),
            backtest_params=existing_params.get(slug, m.default_params if m else {}),
        ))
    new_config = DeploymentConfig(
        version=(existing.version if existing else 0),
        updated_at="",
        strategies=strategies,
    )
    write_deployment_config(new_config)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deployment Control
# ══════════════════════════════════════════════════════════════════════════════

def _tab_deployment(metas: list, results: dict, config):
    from portfolio.multi_strategy import (
        write_deployment_config,
        DeploymentConfig,
        DeployedStrategy,
    )

    st.markdown("### Deployment Configuration")
    st.caption("Control which strategies are active and their allocation weights. "
               "Weights are normalised automatically. Changes take effect on the next signal generation run.")

    if not metas:
        st.warning("No strategies found in the strategies/ directory.")
        return

    existing_map: dict[str, DeployedStrategy] = {}
    if config:
        for s in config.strategies:
            existing_map[s.slug] = s

    updated: list[DeployedStrategy] = []
    total_weight_ref = [0.0]

    for i, m in enumerate(metas):
        ex = existing_map.get(m.slug)
        default_active = ex.active if ex else True
        default_weight = ex.allocation_weight if ex else round(1.0 / len(metas), 3)
        default_params = ex.backtest_params if ex else m.default_params

        has_result = m.slug in results
        status_badge = badge("cached", "blue") if has_result else badge("no data", "neutral")

        with st.expander(f"{m.name}  {status_badge}", expanded=True):
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                active = st.checkbox("Active", value=default_active, key=f"active_{m.slug}")
            with c2:
                weight = st.number_input(
                    "Allocation weight", min_value=0.0, max_value=1.0,
                    value=float(default_weight), step=0.05,
                    key=f"weight_{m.slug}",
                    disabled=not active,
                )
            with c3:
                if has_result:
                    r = results[m.slug]
                    m_data = r.metrics
                    st.markdown(
                        f"CAGR **{_pct(m_data.get('cagr'))}**  "
                        f"Sharpe **{m_data.get('sharpe', 0):.2f}**  "
                        f"Max DD **{_pct(m_data.get('max_drawdown'))}**"
                    )
                else:
                    st.caption("Run backtests in the Comparison tab first.")

            updated.append(DeployedStrategy(
                slug=m.slug,
                name=m.name,
                active=active,
                allocation_weight=weight if active else 0.0,
                backtest_params=default_params,
            ))

    # Show normalised weights preview
    total = sum(s.allocation_weight for s in updated if s.active) or 1.0
    active_count = sum(1 for s in updated if s.active)
    st.markdown(f"**{active_count} active** strategies · total raw weight = **{total:.2f}**")

    preview_rows = []
    for s in updated:
        if s.active:
            preview_rows.append({
                "Strategy": s.name,
                "Raw weight": f"{s.allocation_weight:.3f}",
                "Normalised": f"{s.allocation_weight/total*100:.1f}%",
            })
    if preview_rows:
        st.table(pd.DataFrame(preview_rows))

    if st.button("Save Deployment Config", type="primary"):
        new_config = DeploymentConfig(
            version=(config.version if config else 0),
            updated_at="",
            strategies=updated,
        )
        write_deployment_config(new_config)
        st.success("Configuration saved. Run the pipeline to apply changes.")
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Blended Preview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_blended(config, results: dict):
    st.markdown("### Blended Symbol-Level Positions")
    st.caption(
        "Estimated combined holdings if the deployed strategies were all running today. "
        "Actual live weights are driven by generate_signals.py and stored in "
        "`data/gold/equity/target_weights.parquet`."
    )

    from config.settings import DATA_DIR
    tw_path = DATA_DIR / "gold" / "equity" / "target_weights.parquet"

    if tw_path.exists():
        import polars as pl
        tw = pl.read_parquet(tw_path)
        latest_date = tw["date"].max()
        latest = tw.filter(pl.col("date") == latest_date)
        st.markdown(f"**Live target weights** (as of {latest_date}):")
        st.dataframe(
            latest.sort("weight", descending=True).to_pandas().reset_index(drop=True),
            use_container_width=True,
        )

        # Pie chart
        rows_pd = latest.sort("weight", descending=True).to_pandas()
        if not rows_pd.empty:
            fig = go.Figure(go.Pie(
                labels=rows_pd["symbol"].tolist(),
                values=rows_pd["weight"].tolist(),
                marker=dict(colors=_SERIES, line=dict(color=COLORS["bg"], width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, color=COLORS["text"]),
                hole=0.5,
            ))
            from reports._theme import apply_theme
            apply_theme(fig, height=320, title="Live Position Weights")
            fig.update_layout(showlegend=False, paper_bgcolor=COLORS["card_bg"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No target weights found. Run `uv run python orchestration/generate_signals.py` first.")

    # Config summary
    if config:
        st.markdown("### Active Deployment Config")
        total = sum(s.allocation_weight for s in config.strategies if s.active) or 1.0
        rows = []
        for s in config.strategies:
            rows.append({
                "Strategy": s.name,
                "Active": "✓" if s.active else "✗",
                "Raw weight": f"{s.allocation_weight:.3f}",
                "Normalised": f"{s.allocation_weight/total*100:.1f}%" if s.active else "—",
            })
        st.table(pd.DataFrame(rows))
        updated_str = config.updated_at or "unknown"
        st.caption(f"Config version {config.version} · last updated {updated_str}")
    else:
        st.info("No deployment config saved yet. Go to the **Deployment** tab to configure.")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Portfolio Management")
st.caption("Compare strategies, optimize allocations, and control what gets traded.")

metas = _discover()
config = _load_config()

# Session-state backtest cache
if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = {}

results: dict = st.session_state["backtest_results"]

# Top-level controls
ctrl_left, ctrl_right = st.columns([3, 1])
with ctrl_left:
    if metas:
        st.markdown(f"**{len(metas)} strategies found** — "
                    + ", ".join(f"`{m.slug}`" for m in metas))
    else:
        st.warning("No strategies found in `strategies/` directory.")

with ctrl_right:
    run_col, force_col = st.columns(2)
    with run_col:
        if st.button("▶ Run Backtests", use_container_width=True, type="primary"):
            new_results = _run_backtests(metas, force=False)
            st.session_state["backtest_results"].update(new_results)
            results = st.session_state["backtest_results"]
    with force_col:
        if st.button("↺ Force Refresh", use_container_width=True):
            new_results = _run_backtests(metas, force=True)
            st.session_state["backtest_results"] = new_results
            results = st.session_state["backtest_results"]

st.markdown("---")

tab_overview, tab_compare, tab_optimizer, tab_deploy, tab_blended = st.tabs([
    "📊 Overview", "📈 Comparison", "🎯 Optimizer", "⚙️ Deployment", "🔍 Blended Preview"
])

with tab_overview:
    _tab_overview(results, config)

with tab_compare:
    _tab_comparison(results, metas)

with tab_optimizer:
    _tab_optimizer(results, config)

with tab_deploy:
    _tab_deployment(metas, results, config)

with tab_blended:
    _tab_blended(config, results)
