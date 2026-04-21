"""Research dashboard — Streamlit page (Dashboard #4).

UI-only layer. All analytics logic lives in research/.

Tabs:
  Signal Scanner  — current factor rankings and universe snapshot
  Factor Analysis — factor time-series, distribution, and information coefficient
  Walk-Forward    — OOS validation with per-fold Sharpe breakdown
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from scipy.stats import norm as _scipy_norm
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, range_selector,
    kpi_card, section_label, page_header,
)
from research.signal_scanner import (
    ALL_FEATURES, FEATURE_LABELS, PCT_FEATURES,
    get_snapshot, momentum_ranked,
)
from research.factor_analysis import (
    compute_factor_stats, compute_ic,
    factor_pivot_from_features, price_pivot_from_bars,
)
from research.walk_forward_runner import WFVConfig, fold_summary, oos_equity_normalised, run as run_wfv
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date
from features.compute import load_features

st.markdown(CSS, unsafe_allow_html=True)

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _features(symbols: tuple, start_str: str, end_str: str) -> pl.DataFrame | None:
    try:
        df = load_features(list(symbols), date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        return None if df.is_empty() else df
    except Exception:
        return None


@st.cache_data(ttl=300)
def _prices(symbols: tuple, start_str: str, end_str: str) -> pl.DataFrame | None:
    try:
        df = load_bars(list(symbols), date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        return None if df.is_empty() else df
    except Exception:
        return None


@st.cache_data(ttl=600)
def _walk_forward(
    symbols: tuple,
    start_str: str,
    end_str: str,
    train_years: int,
    test_months: int,
    top_n: int,
    cost_bps: float,
):
    try:
        return run_wfv(
            list(symbols),
            date.fromisoformat(start_str),
            date.fromisoformat(end_str),
            WFVConfig(train_years=train_years, test_months=test_months,
                      top_n=top_n, cost_bps=cost_bps),
        )
    except Exception as exc:
        return str(exc)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    lookback_years = st.slider("Feature lookback (years)", 1, 7, 6)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "QuantPipe — Research",
        "Factor analysis · Signal diagnostics · Walk-forward validation",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# ── Universe + feature load ───────────────────────────────────────────────────

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

features_df: pl.DataFrame | None = None
if _symbols:
    with st.spinner("Loading features…"):
        features_df = _features(_symbols, str(_start), str(_end))

tab_scanner, tab_factor, tab_wfv = st.tabs(
    ["  Signal Scanner  ", "  Factor Analysis  ", "  Walk-Forward  "]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scanner:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        snap = get_snapshot(features_df)

        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown(
            page_header("", f"Universe snapshot · {snap.latest_date}", ""),
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("Universe Size",        str(snap.n_universe),        accent=COLORS["teal"]),     unsafe_allow_html=True)
        c2.markdown(kpi_card("Symbols w/ Momentum",  str(snap.n_valid_momentum),  accent=COLORS["blue"]),     unsafe_allow_html=True)
        c3.markdown(kpi_card("Current Top-1",         snap.top5_momentum[0] if snap.top5_momentum else "—", accent=COLORS["positive"]), unsafe_allow_html=True)
        c4.markdown(kpi_card("As-of Date",            snap.latest_date,            accent=COLORS["neutral"]), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

        col_rank, col_heat = st.columns(2)

        # ── Momentum ranking bar ──────────────────────────────────────────────
        with col_rank:
            st.markdown(section_label("12-1 Momentum Ranking"), unsafe_allow_html=True)
            mom = momentum_ranked(snap.snap_pd)
            if not mom.empty:
                bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in mom.values]
                fig_rank = go.Figure(go.Bar(
                    x=mom.values,
                    y=mom.index.tolist(),
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.1%}" for v in mom.values],
                    textposition="outside",
                    textfont=dict(size=10, color=COLORS["text"]),
                    hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                ))
                fig_rank.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_rank)
                fig_rank.update_layout(
                    height=max(300, 26 * len(mom)),
                    xaxis=dict(tickformat=".0%", showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_rank, width="stretch", config=PLOTLY_CONFIG)

        # ── Factor z-score heatmap ────────────────────────────────────────────
        with col_heat:
            st.markdown(section_label("Factor Z-Scores"), unsafe_allow_html=True)
            z = snap.z_scores
            z_cols = [c for c in z.columns if z[c].notna().any()]
            if z_cols:
                z_sorted = z[z_cols].sort_values("momentum_12m_1m", ascending=False) \
                    if "momentum_12m_1m" in z_cols else z[z_cols]
                col_labels = [FEATURE_LABELS.get(c, c) for c in z_cols]
                text_ann = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in z_sorted.values]
                fig_heat = go.Figure(go.Heatmap(
                    z=z_sorted.values,
                    x=col_labels,
                    y=z_sorted.index.tolist(),
                    text=text_ann,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    colorscale="RdYlGn",
                    zmid=0,
                    showscale=True,
                    colorbar=dict(thickness=12, len=0.85),
                    hovertemplate="<b>%{y}</b> · %{x}: %{z:.2f}σ<extra></extra>",
                ))
                apply_theme(fig_heat)
                fig_heat.update_layout(
                    height=max(300, 26 * len(z_sorted)),
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)

        # ── Full factor snapshot table ────────────────────────────────────────
        st.markdown(section_label("Full Factor Snapshot"), unsafe_allow_html=True)
        pf = snap.present_features
        if pf:
            display = snap.snap_pd[pf].copy()
            display.index.name = "Symbol"
            display.columns = [FEATURE_LABELS.get(c, c) for c in display.columns]
            if "12-1 Momentum" in display.columns:
                display = display.sort_values("12-1 Momentum", ascending=False)
            gradient_cols = [FEATURE_LABELS.get(f, f) for f in pf if f != "dollar_volume_63d"
                             and FEATURE_LABELS.get(f, f) in display.columns]
            fmt = {FEATURE_LABELS.get(f, f): ("${:,.0f}" if f == "dollar_volume_63d" else "{:.3f}") for f in pf}
            st.dataframe(
                display.style.background_gradient(cmap="RdYlGn", axis=0, subset=gradient_cols).format(fmt),
                width="stretch",
                height=min(600, 38 * (len(display) + 1)),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_factor:
    if features_df is None:
        st.warning("No features available.")
    else:
        present = [f for f in ALL_FEATURES if f in features_df.columns]

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
        with col_ctrl1:
            selected_factor = st.selectbox("Factor", present, format_func=lambda x: FEATURE_LABELS.get(x, x))
        with col_ctrl2:
            all_syms = sorted(features_df["symbol"].unique().to_list())
            selected_syms = st.multiselect("Symbols", all_syms, default=all_syms[:8], max_selections=15)
        with col_ctrl3:
            ic_window = st.selectbox("IC window (days)", [21, 63, 126], index=0,
                                     format_func=lambda x: f"{x}d fwd")

        if not selected_syms:
            st.info("Select at least one symbol.")
        else:
            fac_pivot = factor_pivot_from_features(features_df, selected_syms, selected_factor)
            is_pct    = selected_factor in PCT_FEATURES

            # ── Time series ───────────────────────────────────────────────────
            st.markdown(section_label(f"{FEATURE_LABELS.get(selected_factor, selected_factor)} — Time Series"), unsafe_allow_html=True)
            fig_ts = go.Figure()
            for i, sym in enumerate(selected_syms):
                if sym in fac_pivot.columns:
                    fig_ts.add_trace(go.Scatter(
                        x=fac_pivot.index, y=fac_pivot[sym],
                        name=sym,
                        mode="lines",
                        line=dict(color=COLORS["series"][i % len(COLORS["series"])], width=1.5),
                        hovertemplate=f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}: %{{y:.4f}}<extra></extra>",
                    ))
            apply_theme(fig_ts, legend_inside=False)
            fig_ts.update_layout(
                height=320,
                xaxis=dict(rangeselector=range_selector()),
                hovermode="x unified",
                yaxis=dict(tickformat=".1%" if is_pct else ",.2f"),
            )
            st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CONFIG)

            col_dist, col_ic = st.columns(2)

            # ── Distribution ──────────────────────────────────────────────────
            with col_dist:
                st.markdown(section_label("Factor Distribution"), unsafe_allow_html=True)
                all_vals = fac_pivot.values.ravel()
                all_vals = all_vals[~np.isnan(all_vals)]
                fstats   = compute_factor_stats(all_vals)

                x_grid  = np.linspace(fstats.p5, fstats.p95, 200)
                pdf_fit = _scipy_norm.pdf(x_grid, fstats.mean, fstats.std)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=all_vals, histnorm="probability density", nbinsx=60,
                    marker=dict(color=COLORS["blue"], line=dict(width=0)), opacity=0.65,
                    name="Observed",
                ))
                fig_dist.add_trace(go.Scatter(
                    x=x_grid, y=pdf_fit,
                    line=dict(color=COLORS["warning"], width=2), name="Normal fit",
                ))
                apply_theme(fig_dist, legend_inside=True)
                fig_dist.update_layout(
                    height=280,
                    xaxis=dict(tickformat=".1%" if is_pct else ",.2f"),
                    yaxis=dict(title="Density"),
                    bargap=0.03,
                )
                st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)

                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Mean",  f"{fstats.mean:.4f}")
                c_b.metric("Std",   f"{fstats.std:.4f}")
                c_c.metric("Skew",  f"{fstats.skew:.2f}")
                c_d.metric("Kurt",  f"{fstats.kurt:.2f}")

            # ── Information Coefficient ───────────────────────────────────────
            with col_ic:
                st.markdown(section_label(f"Information Coefficient ({ic_window}d forward)"), unsafe_allow_html=True)

                prices_pl = _prices(_symbols, str(_start), str(_end))
                ic_result = None
                if prices_pl is not None:
                    price_piv = price_pivot_from_bars(prices_pl)
                    ic_result = compute_ic(fac_pivot, price_piv, ic_window)

                if ic_result and ic_result.values:
                    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in ic_result.values]
                    fig_ic = go.Figure()
                    fig_ic.add_trace(go.Bar(
                        x=ic_result.dates, y=ic_result.values,
                        marker=dict(color=bar_colors, line=dict(width=0)), opacity=0.55,
                        name="IC",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>IC</extra>",
                    ))
                    fig_ic.add_trace(go.Scatter(
                        x=ic_result.dates, y=ic_result.rolling_mean,
                        line=dict(color=COLORS["teal"], width=2), name="6-period MA",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>MA</extra>",
                    ))
                    fig_ic.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                    apply_theme(fig_ic, legend_inside=True)
                    fig_ic.update_layout(
                        height=280,
                        yaxis=dict(tickformat=".2f", title="Rank IC"),
                    )
                    st.plotly_chart(fig_ic, width="stretch", config=PLOTLY_CONFIG)

                    c_a, c_b = st.columns(2)
                    c_a.metric("Mean IC", f"{ic_result.mean_ic:.4f}")
                    c_b.metric("IC IR",   f"{ic_result.icir:.3f}")
                else:
                    st.info("Could not compute IC — price data unavailable.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_wfv:
    st.markdown(section_label("Walk-Forward Validation Configuration"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    wfv_train = c1.slider("Training window (years)", 2, 5, 3)
    wfv_test  = c2.slider("Test window (months)", 3, 24, 12)
    wfv_top_n = c3.slider("Top-N positions", 3, 10, 5)
    wfv_cost  = c4.number_input("Cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

    required_years = wfv_train + wfv_test / 12
    if lookback_years < required_years:
        st.warning(f"Increase Feature lookback to at least {required_years:.1f} years to produce complete folds.")

    run_btn = st.button("▶ Run Walk-Forward Validation", type="primary")
    _key    = f"wfv_{_symbols}_{wfv_train}_{wfv_test}_{wfv_top_n}_{wfv_cost}_{lookback_years}"

    if run_btn:
        with st.spinner("Running walk-forward validation — this may take 30–90 seconds…"):
            st.session_state[_key] = _walk_forward(
                _symbols, str(_start), str(_end),
                wfv_train, wfv_test, wfv_top_n, float(wfv_cost),
            )

    wfv_result = st.session_state.get(_key)

    if wfv_result is None:
        st.info("Configure parameters above and click **▶ Run Walk-Forward Validation** to start.")
    elif isinstance(wfv_result, str):
        st.error(f"Walk-forward failed: {wfv_result}")
    else:
        folds = wfv_result.folds

        # ── Summary KPIs ──────────────────────────────────────────────────────
        st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("OOS Sharpe", f"{wfv_result.combined_sharpe:.3f}",        accent=COLORS["teal"]),     unsafe_allow_html=True)
        c2.markdown(kpi_card("OOS CAGR",   f"{wfv_result.combined_cagr:.1%}",          accent=COLORS["blue"]),     unsafe_allow_html=True)
        c3.markdown(kpi_card("OOS Max DD",  f"{wfv_result.combined_max_drawdown:.1%}",  accent=COLORS["negative"]), unsafe_allow_html=True)
        c4.markdown(kpi_card("Folds",       str(len(folds)),                            accent=COLORS["neutral"]),  unsafe_allow_html=True)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

        # ── OOS equity curve ──────────────────────────────────────────────────
        st.markdown(section_label("OOS Combined Equity Curve"), unsafe_allow_html=True)
        eq_norm = oos_equity_normalised(wfv_result)
        fig_eq  = go.Figure(go.Scatter(
            x=eq_norm.index, y=eq_norm.values,
            fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
            line=dict(color=COLORS["positive"], width=2),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra>OOS</extra>",
        ))
        palette = [COLORS["card_bg"], COLORS["bg"]]
        for i, fold in enumerate(folds):
            fig_eq.add_vrect(
                x0=str(fold.test_start), x1=str(fold.test_end),
                fillcolor=palette[i % 2], opacity=0.35, layer="below", line_width=0,
                annotation_text=f"F{fold.fold + 1}",
                annotation_position="top left",
                annotation_font=dict(size=9, color=COLORS["text_muted"]),
            )
        apply_theme(fig_eq)
        fig_eq.update_layout(
            height=300,
            yaxis=dict(tickformat="$,.0f"),
            xaxis=dict(rangeselector=range_selector()),
        )
        st.plotly_chart(fig_eq, width="stretch", config=PLOTLY_CONFIG)

        # ── Per-fold table ────────────────────────────────────────────────────
        st.markdown(section_label("Per-Fold Performance"), unsafe_allow_html=True)
        rows = fold_summary(wfv_result)
        fold_df = pd.DataFrame(rows)
        fold_df["OOS CAGR"]   = fold_df["OOS CAGR"].map("{:.1%}".format)
        fold_df["OOS Max DD"] = fold_df["OOS Max DD"].map("{:.1%}".format)
        fold_df["OOS Vol"]    = fold_df["OOS Vol"].map("{:.1%}".format)
        st.dataframe(fold_df, width="stretch", hide_index=True)

        # ── Per-fold bar charts ───────────────────────────────────────────────
        col_sharpe_bar, col_cagr_bar = st.columns(2)
        sharpe_vals = [r["OOS Sharpe"] for r in rows]
        cagr_vals   = [r["OOS CAGR"] for r in rows]   # raw float before formatting
        fold_labels = [f"Fold {r['Fold']}  {r['Test Start'][:7]}" for r in rows]

        with col_sharpe_bar:
            fig_sb = go.Figure(go.Bar(
                x=fold_labels, y=sharpe_vals,
                marker=dict(color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in sharpe_vals], line=dict(width=0)),
                text=[f"{v:.2f}" for v in sharpe_vals], textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{x}</b>: %{y:.3f}<extra>Sharpe</extra>",
            ))
            fig_sb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_sb)
            fig_sb.update_layout(height=240, title="OOS Sharpe by Fold",
                                  yaxis=dict(showgrid=False), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_sb, width="stretch", config=PLOTLY_CONFIG)

        with col_cagr_bar:
            fig_cb = go.Figure(go.Bar(
                x=fold_labels, y=cagr_vals,
                marker=dict(color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in cagr_vals], line=dict(width=0)),
                text=[f"{v:.1%}" for v in cagr_vals], textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{x}</b>: %{y:.1%}<extra>CAGR</extra>",
            ))
            fig_cb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_cb)
            fig_cb.update_layout(height=240, title="OOS CAGR by Fold",
                                  yaxis=dict(tickformat=".0%", showgrid=False), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_cb, width="stretch", config=PLOTLY_CONFIG)

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
