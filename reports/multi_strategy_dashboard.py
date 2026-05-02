"""Blends — compare and optimise strategy blends. Read-only; use Deployment to save configs.

Tabs:
  1. Overview    — active strategies, blended equity curve, allocation pie
  2. Comparison  — metrics table, overlaid curves, drawdown, rolling Sharpe, monthly returns
  3. Optimizer   — correlation heatmap, allocation optimizer, efficient frontier
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import COLORS, apply_theme, badge, page_header, CSS, kpi_card, section_label

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _color(i: int) -> str:
    return COLORS["series"][i % len(COLORS["series"])]


def _hex_rgba(hex_color: str, alpha: float) -> str:
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


# ── Data loaders ───────────────────────────────────────────────────────────────

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
            row[label] = _fmt(v, pct=is_pct, decimals=1 if is_pct else 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()



# ── Page ───────────────────────────────────────────────────────────────────────

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    page_header("Blends", "Compare strategy blends and optimise allocations. Deploy changes in the Deployment tab."),
    unsafe_allow_html=True,
)

metas  = _discover()
config = _load_config()

if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = {}
results: dict = st.session_state["backtest_results"]

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
        if st.button("Run Backtests", use_container_width=True, type="primary", key="ms_run"):
            new_r = _run_backtests(metas, force=False)
            st.session_state["backtest_results"].update(new_r)
            st.rerun()
    with rc2:
        if st.button("Force Refresh", use_container_width=True, key="ms_force"):
            new_r = _run_backtests(metas, force=True)
            st.session_state["backtest_results"] = new_r
            st.rerun()

st.markdown("---")

tab_overview, tab_compare, tab_optimizer = st.tabs(["  Overview  ", "  Comparison  ", "  Optimizer  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    from portfolio.multi_strategy import build_return_matrix

    if not results:
        st.info("No backtest results yet — click **Run Backtests** above.")
    else:
        active_slugs = (
            {s.slug for s in config.strategies if s.active}
            if config else set(results.keys())
        )
        active_results = {k: v for k, v in results.items() if k in active_slugs}

        if not active_results:
            st.warning("No active strategies — configure deployment in **Deployment**.")
        else:
            alloc: dict[str, float] = {}
            if config:
                for s in config.strategies:
                    if s.active and s.slug in active_results:
                        alloc[s.slug] = s.allocation_weight
            if not alloc:
                n = len(active_results)
                alloc = {slug: 1.0 / n for slug in active_results}
            total_alloc = sum(alloc.values()) or 1.0
            alloc_norm  = {k: v / total_alloc for k, v in alloc.items()}

            ret_matrix = build_return_matrix(list(active_results.values()))
            blended_eq = pd.Series(dtype=float)
            if not ret_matrix.empty:
                w = np.array([alloc_norm.get(c, 0.0) for c in ret_matrix.columns])
                if w.sum() > 1e-10:
                    w /= w.sum()
                blended_eq = 10_000.0 * (1 + (ret_matrix * w).sum(axis=1)).cumprod()

            active_allocated = [(slug, r) for slug, r in active_results.items()
                                if alloc_norm.get(slug, 0.0) > 1e-6]
            n_cards = len(active_allocated)
            if n_cards == 0:
                st.info("No strategies have a non-zero allocation. Set weights in the Deployment tab.")
            card_cols = st.columns(max(n_cards, 1) + 1)
            for i, (slug, r) in enumerate(active_allocated):
                alloc_pct = alloc_norm.get(slug, 0.0)
                m = r.metrics
                color = _color(i)
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
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{_pct(m.get('cagr'))}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">CAGR</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['text']};">{m.get('sharpe', 0):.2f}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Sharpe</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.78rem;font-weight:700;color:{COLORS['negative']};">{_pct(m.get('max_drawdown'))}</div>
      <div style="font-size:0.62rem;color:{COLORS['text_muted']};">Max DD</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            with card_cols[-1]:
                labels = [active_results[s].name for s in alloc_norm if s in active_results]
                values = [alloc_norm[s] for s in alloc_norm if s in active_results]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=[_color(i) for i in range(len(labels))],
                                line=dict(color=COLORS["bg"], width=2)),
                    textinfo="label+percent", textfont=dict(size=11, color=COLORS["text"]), hole=0.55,
                ))
                apply_theme(fig_pie, title="Allocation", height=220)
                fig_pie.update_layout(showlegend=False, paper_bgcolor=COLORS["card_bg"])
                st.plotly_chart(fig_pie, use_container_width=True)

            if not blended_eq.empty:
                _bret_s = blended_eq.pct_change().dropna()
                _n_yrs  = max(len(blended_eq) / 252, 1e-6)
                _b_cagr = (blended_eq.iloc[-1] / blended_eq.iloc[0]) ** (1 / _n_yrs) - 1
                _b_vol  = float(_bret_s.std() * np.sqrt(252))
                _b_sh   = _b_cagr / _b_vol if _b_vol > 1e-10 else 0
                _b_peak = blended_eq.cummax()
                _b_dd   = float(((blended_eq - _b_peak) / _b_peak).min())
                _b_cal  = _b_cagr / abs(_b_dd) if abs(_b_dd) > 1e-10 else 0
                st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
                st.markdown(section_label("Blended Portfolio"), unsafe_allow_html=True)
                _k1, _k2, _k3, _k4 = st.columns(4)
                _k1.markdown(kpi_card("Blended CAGR",  f"{_b_cagr:.1%}", accent=COLORS["positive"]), unsafe_allow_html=True)
                _k2.markdown(kpi_card("Blended Sharpe", f"{_b_sh:.2f}",  accent=COLORS["teal"]),     unsafe_allow_html=True)
                _k3.markdown(kpi_card("Blended Max DD", f"{_b_dd:.1%}",  accent=COLORS["negative"]), unsafe_allow_html=True)
                _k4.markdown(kpi_card("Blended Calmar", f"{_b_cal:.2f}", accent=COLORS["blue"]),     unsafe_allow_html=True)
                st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

                fig = go.Figure()
                for i, (slug, r) in enumerate(active_results.items()):
                    eq = _normalise(_equity_series(r))
                    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=r.name,
                                              line=dict(color=_color(i), width=1.5, dash="dot"), opacity=0.65))
                fig.add_trace(go.Scatter(x=blended_eq.index, y=blended_eq.values,
                                          name="Blended Portfolio",
                                          line=dict(color=COLORS["positive"], width=2.8)))
                first = next(iter(active_results.values()))
                if first.benchmark_dates:
                    spy_idx = pd.to_datetime(first.benchmark_dates)
                    spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
                    fig.add_trace(go.Scatter(x=spy_idx, y=spy_vals.values, name="SPY",
                                              line=dict(color=COLORS["neutral"], width=1.2, dash="dash"), opacity=0.7))
                apply_theme(fig, title="Blended Portfolio vs Components ($10k start)", height=380)
                fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
                st.plotly_chart(fig, use_container_width=True)

            # ── Cross-strategy P&L Attribution ────────────────────────────────
            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
            st.markdown(section_label("Cross-Strategy P&L Attribution"), unsafe_allow_html=True)
            st.caption("Cumulative return contribution from each strategy, weighted by allocation.")
            if not blended_eq.empty and active_results:
                _attr_fig = go.Figure()
                _strat_contribs = {}
                for _i, (_slug, _r) in enumerate(active_results.items()):
                    _eq_s = _equity_series(_r)
                    _daily_r = _eq_s.pct_change().dropna()
                    _w = alloc_norm.get(_slug, 0.0)
                    _contrib_s = (_daily_r * _w).cumsum()
                    _strat_contribs[_r.name] = _contrib_s
                    _attr_fig.add_trace(go.Scatter(
                        x=_contrib_s.index,
                        y=_contrib_s.values * 100,
                        mode="lines",
                        name=_r.name,
                        stackgroup="one",
                        line=dict(color=_color(_i), width=0),
                        fillcolor=_color(_i).replace(")", ", 0.55)").replace("rgb(", "rgba(")
                                  if _color(_i).startswith("rgb") else _color(_i),
                        hovertemplate=f"<b>{_r.name}</b><br>%{{x|%Y-%m-%d}}: %{{y:+.2f}}%<extra></extra>",
                    ))
                apply_theme(_attr_fig, legend_inside=False)
                _attr_fig.update_layout(
                    height=280,
                    yaxis=dict(title="Cumulative contribution (%)", ticksuffix="%"),
                    xaxis=dict(showgrid=False),
                    hovermode="x unified",
                )
                st.plotly_chart(_attr_fig, use_container_width=True)

                # Summary table: period contribution per strategy
                _attr_rows = []
                for _name, _cs in _strat_contribs.items():
                    _slug2 = next((s for s, r in active_results.items() if r.name == _name), "")
                    _attr_rows.append({
                        "Strategy":      _name,
                        "Allocation":    f"{alloc_norm.get(_slug2, 0):.1%}",
                        "Total Contrib": f"{float(_cs.iloc[-1])*100:+.2f}%",
                        "Avg Daily":     f"{float(_cs.diff().mean())*100:+.4f}%",
                    })
                st.dataframe(pd.DataFrame(_attr_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    if not results:
        st.info("No results yet — click **Run Backtests** above.")
    else:
        st.markdown("### Performance Metrics")
        df = _metric_table(results)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

        _dp_col, _ = st.columns([2, 8])
        with _dp_col:
            if st.button("🚀 Deploy to Paper", type="primary", use_container_width=True,
                         help="Run generate_signals then rebalance in paper trading mode"):
                with st.spinner("Deploying to paper account…"):
                    try:
                        import subprocess
                        _gs = subprocess.run(
                            [sys.executable, str(_ROOT / "orchestration" / "generate_signals.py")],
                            capture_output=True, text=True, cwd=str(_ROOT), timeout=120,
                        )
                        if _gs.returncode != 0:
                            st.error(f"generate_signals failed:\n{_gs.stderr[-500:]}")
                        else:
                            st.success("Signals generated. Go to **Deployment → Trade** to execute.")
                    except Exception as _de:
                        st.error(f"Deploy failed: {_de}")

        st.markdown("### Equity Curves (normalised to $10,000)")
        fig_eq = go.Figure()
        for i, (slug, r) in enumerate(results.items()):
            eq = _normalise(_equity_series(r))
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name=r.name,
                                         line=dict(color=_color(i), width=2)))
        first = next(iter(results.values()))
        if first.benchmark_dates:
            spy_idx = pd.to_datetime(first.benchmark_dates)
            spy_vals = pd.Series(first.benchmark_values, index=spy_idx)
            fig_eq.add_trace(go.Scatter(x=spy_idx, y=spy_vals.values, name="SPY",
                                         line=dict(color=COLORS["neutral"], width=1.5, dash="dash"), opacity=0.8))
        apply_theme(fig_eq, height=400)
        fig_eq.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("### Drawdown")
        fig_dd = go.Figure()
        for i, (slug, r) in enumerate(results.items()):
            dd = _drawdown(_equity_series(r)) * 100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=r.name,
                                         line=dict(color=_color(i), width=1.8),
                                         fill="tozeroy", fillcolor=_hex_rgba(_color(i), 0.08)))
        apply_theme(fig_dd, height=260)
        fig_dd.update_layout(yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("### Rolling 12-Month Sharpe")
        fig_sh = go.Figure()
        for i, (slug, r) in enumerate(results.items()):
            daily_ret = _equity_series(r).pct_change().dropna()
            roll_sh = (daily_ret.rolling(252).mean() /
                       daily_ret.rolling(252).std().replace(0.0, np.nan)) * np.sqrt(252)
            fig_sh.add_trace(go.Scatter(x=roll_sh.index, y=roll_sh.values, name=r.name,
                                         line=dict(color=_color(i), width=1.8)))
        fig_sh.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        apply_theme(fig_sh, height=260)
        st.plotly_chart(fig_sh, use_container_width=True)

        st.markdown("### Monthly Returns")
        _mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        for i, (slug, r) in enumerate(results.items()):
            eq      = _equity_series(r)
            monthly = eq.resample("ME").last().pct_change().dropna()
            mdf = monthly.to_frame("ret")
            mdf["Year"]  = mdf.index.year
            mdf["Month"] = mdf.index.strftime("%b")
            pivot = mdf.pivot_table(values="ret", index="Year", columns="Month")
            pivot = pivot.reindex(columns=[m for m in _mo if m in pivot.columns])
            z     = pivot.values
            text  = [[f"{v:.1%}" if v == v else "" for v in row] for row in z]
            fig_hm = go.Figure(go.Heatmap(
                z=z, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index.tolist()],
                text=text, texttemplate="%{text}", textfont=dict(size=10, color=COLORS["text"]),
                colorscale="RdYlGn", zmid=0, showscale=True,
                colorbar=dict(tickformat=".0%", thickness=12, len=0.8),
                hovertemplate="<b>%{y} %{x}</b>: %{text}<extra></extra>",
            ))
            apply_theme(fig_hm, title=f"{r.name} — Monthly Returns",
                        height=max(160, 32 * len(pivot) + 60))
            fig_hm.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
            st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimizer
# ══════════════════════════════════════════════════════════════════════════════

with tab_optimizer:
    from portfolio.multi_strategy import (
        build_return_matrix, strategy_correlation, optimize_allocations,
    )

    if len(results) < 2:
        st.info("Need at least 2 strategy backtests to run the optimizer.")
    else:
        ret_matrix = build_return_matrix(list(results.values()))
        if ret_matrix.empty or len(ret_matrix.columns) < 2:
            st.warning("Insufficient overlapping return history to compute correlations.")
        else:
            corr = strategy_correlation(ret_matrix)
            slug_to_name = {slug: r.name for slug, r in results.items()}
            names = [slug_to_name.get(c, c) for c in corr.columns]

            st.markdown("### Strategy Correlation Matrix")
            z_text = [[f"{v:.2f}" for v in row] for row in corr.values]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=names, y=names,
                colorscale=[[0, COLORS["negative"]], [0.5, COLORS["card_bg"]], [1, COLORS["positive"]]],
                zmin=-1, zmax=1, text=z_text, texttemplate="%{text}",
                textfont=dict(size=13, color=COLORS["text"]), showscale=True,
                colorbar=dict(tickfont=dict(color=COLORS["neutral"])),
            ))
            apply_theme(fig_corr, height=300)
            fig_corr.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, autorange="reversed"))
            st.plotly_chart(fig_corr, use_container_width=True)

            ann_ret = ret_matrix.mean() * 252
            ann_vol = ret_matrix.std() * np.sqrt(252)
            ann_sharpe = ann_ret / ann_vol.replace(0.0, np.nan)
            st.markdown("### Individual Strategy Stats (annualised)")
            stat_cols = st.columns(max(len(ret_matrix.columns), 1))
            for i, col in enumerate(ret_matrix.columns):
                name = slug_to_name.get(col, col)
                with stat_cols[i]:
                    st.markdown(f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['border']};
     border-left:3px solid {_color(i)};border-radius:8px;padding:12px 14px;">
  <div style="font-size:0.7rem;color:{COLORS['text_muted']};text-transform:uppercase;">{name}</div>
  <div style="font-size:0.84rem;font-weight:700;color:{COLORS['text']};margin-top:6px;">
    Ret {ann_ret[col]*100:+.1f}% · Vol {ann_vol[col]*100:.1f}% · Sharpe {"—" if np.isnan(ann_sharpe[col]) else f"{ann_sharpe[col]:.2f}"}
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("### Optimize Allocations")
            c1, c2 = st.columns([1, 2])
            with c1:
                method = st.selectbox("Optimization method",
                                       ["min_variance", "max_sharpe", "vol_scaled", "equal"],
                                       format_func=lambda x: {"min_variance": "Minimum Variance",
                                                               "max_sharpe": "Maximum Sharpe",
                                                               "vol_scaled": "Inverse-Vol Weighted",
                                                               "equal": "Equal Weight"}[x], key="opt_method_sel")
                max_w = st.slider("Max weight per strategy", 0.30, 1.00, 0.80, 0.05, key="opt_max_w")
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
                lbl    = st.session_state.get("opt_method", "")
                st.markdown(f"**Result ({lbl}):**")
                bar_names = [slug_to_name.get(s, s) for s in allocs]
                bar_vals  = [v * 100 for v in allocs.values()]

                _ci_low, _ci_high = list(bar_vals), list(bar_vals)
                try:
                    _n_boot = 200
                    _boot_allocs = {s: [] for s in allocs}
                    for _ in range(_n_boot):
                        _sample = ret_matrix.sample(n=len(ret_matrix), replace=True)
                        _boot_opt = optimize_allocations(_sample, method=method, max_weight=max_w)
                        for s in allocs:
                            _boot_allocs[s].append(_boot_opt.get(s, 0.0) * 100)
                    _ci_low  = [float(np.percentile(_boot_allocs[s], 5))  for s in allocs]
                    _ci_high = [float(np.percentile(_boot_allocs[s], 95)) for s in allocs]
                except Exception:
                    pass

                _err_low  = [max(v - lo, 0) for v, lo in zip(bar_vals, _ci_low)]
                _err_high = [max(hi - v, 0) for v, hi in zip(bar_vals, _ci_high)]

                fig_bar = go.Figure(go.Bar(
                    x=bar_names, y=bar_vals,
                    marker=dict(color=[_color(i) for i in range(len(allocs))], line=dict(color=COLORS["bg"], width=1)),
                    text=[f"{v:.1f}%" for v in bar_vals], textposition="outside",
                    error_y=dict(type="data", symmetric=False, array=_err_high, arrayminus=_err_low,
                                 color=COLORS["neutral"], thickness=2, width=8),
                ))
                apply_theme(fig_bar, height=280)
                fig_bar.update_layout(yaxis=dict(ticksuffix="%", range=[0, 115]))
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption("Error bars show 5th–95th percentile from 200 bootstrap resamples.")

                st.info("To deploy these allocations, copy the weights to **Deployment → Strategy Config**.", icon="ℹ️")

            st.markdown("### Efficient Frontier")
            _n_sims = 2000
            _n_assets = len(ret_matrix.columns)
            _rng = np.random.default_rng(42)
            _rw  = _rng.dirichlet(np.ones(_n_assets), size=_n_sims)
            _cov = ret_matrix.cov().values
            _port_v, _port_r, _port_sh = [], [], []
            for _w in _rw:
                _pr = float(ret_matrix.mul(_w, axis=1).sum(axis=1).mean() * 252)
                _pv = float(np.sqrt(max(_w @ _cov @ _w, 0)) * np.sqrt(252))
                _port_v.append(_pv); _port_r.append(_pr); _port_sh.append(_pr / _pv if _pv > 1e-10 else 0)
            fig_ef = go.Figure(go.Scatter(
                x=_port_v, y=_port_r, mode="markers",
                marker=dict(color=_port_sh, colorscale="RdYlGn", size=4, opacity=0.45,
                            colorbar=dict(title="Sharpe", thickness=12)),
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra>Random portfolio</extra>",
                showlegend=False,
            ))
            for _i, _col in enumerate(ret_matrix.columns):
                _sr = float(ret_matrix[_col].mean() * 252)
                _sv = float(ret_matrix[_col].std() * np.sqrt(252))
                fig_ef.add_trace(go.Scatter(x=[_sv], y=[_sr], mode="markers+text",
                                             marker=dict(symbol="diamond", size=12, color=_color(_i)),
                                             text=[slug_to_name.get(_col, _col)], textposition="top center",
                                             textfont=dict(size=10, color=COLORS["text"]),
                                             name=slug_to_name.get(_col, _col)))
            if "opt_result" in st.session_state:
                _ao = st.session_state["opt_result"]
                _wo = np.array([_ao.get(c, 0.0) for c in ret_matrix.columns])
                if _wo.sum() > 0:
                    _wo /= _wo.sum()
                _or = float(ret_matrix.mul(_wo, axis=1).sum(axis=1).mean() * 252)
                _ov = float(np.sqrt(max(_wo @ _cov @ _wo, 0)) * np.sqrt(252))
                fig_ef.add_trace(go.Scatter(x=[_ov], y=[_or], mode="markers", name="Optimized",
                                             marker=dict(symbol="star", size=18, color=COLORS["gold"])))
            apply_theme(fig_ef, title="Efficient Frontier (2,000 random portfolios)", height=380)
            fig_ef.update_layout(xaxis=dict(title="Annual Volatility", tickformat=".0%"),
                                  yaxis=dict(title="Annual Return",     tickformat=".0%"))
            st.plotly_chart(fig_ef, use_container_width=True)
            st.caption("Dots = random portfolios (colour = Sharpe). ◆ = individual strategies. ★ = optimizer result.")
