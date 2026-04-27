"""Monte Carlo Block-Bootstrap Dashboard — Strategy group, page 3.

Tabs:
  1. Upload & Run       — CSV upload, config, run button
  2. Fan Chart          — equity fan chart + terminal wealth
  3. Metric Distributions — Sharpe, Calmar, Max DD, Sortino, target achievement
  4. Diagnostics        — ACF, rolling Sharpe stability, convergence
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, kpi_card, section_label, page_header,
)
from research.monte_carlo import MCConfig, load_returns_csv, run as run_mc

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    page_header(
        "Monte Carlo Simulation",
        "Circular block-bootstrap analysis across 10,000+ simulated paths. Upload any CSV of trade returns or equity curves.",
    ),
    unsafe_allow_html=True,
)

tab_upload, tab_fan, tab_dist, tab_diag = st.tabs([
    "  Upload & Run  ",
    "  Fan Chart  ",
    "  Metric Distributions  ",
    "  Diagnostics  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Run
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown(section_label("Simulation Configuration"), unsafe_allow_html=True)

    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1:
        mc_n_sims = st.selectbox("Simulations", [1_000, 5_000, 10_000, 25_000],
                                  index=2, format_func=lambda x: f"{x:,}", key="mc_n_sims")
    with mc_col2:
        mc_block = st.slider("Block size", 3, 63, 10, key="mc_block",
                              help="Block length for circular block bootstrap (match rebalance cadence)")
    with mc_col3:
        mc_ppyr = st.selectbox("Periods / year", [252, 52, 26, 12], key="mc_ppyr",
                                format_func=lambda x: {252: "Daily (252)", 52: "Weekly (52)",
                                                        26: "Biweekly (26)", 12: "Monthly (12)"}[x])
    with mc_col4:
        mc_capital = st.number_input("Initial capital ($)", min_value=1_000, max_value=10_000_000,
                                      value=100_000, step=1_000, key="mc_capital")

    mc_t_col1, mc_t_col2, mc_t_col3 = st.columns(3)
    mc_target_sharpe = mc_t_col1.number_input("Target Sharpe", value=1.0, step=0.1, format="%.1f", key="mc_t_sharpe")
    mc_target_calmar = mc_t_col2.number_input("Target Calmar", value=0.5, step=0.1, format="%.1f", key="mc_t_calmar")
    mc_target_dd     = mc_t_col3.number_input("Max DD target (e.g. −0.20)", value=-0.20, step=0.05, format="%.2f", key="mc_t_dd")

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
    mc_file = st.file_uploader(
        "Upload CSV (trade log or equity curve)",
        type=["csv"],
        key="mc_file",
        help=(
            "Accepted formats:\n"
            "• QC trade log — columns: Entry Price, Exit Price, P&L, Exit Time\n"
            "• Return series — column named 'return', 'returns', or 'ret'\n"
            "• Equity curve — column named 'equity', 'value', 'nav', etc."
        ),
    )
    ck_col1, ck_col2 = st.columns(2)
    mc_is_equity  = ck_col1.checkbox("CSV is an equity curve (not return series)", value=False, key="mc_is_equity")
    mc_pct_format = ck_col2.checkbox("Returns are in % format (÷ 100)", value=False, key="mc_pct_format",
                                      help="Check if your return column stores e.g. 5.0 for a 5% gain instead of 0.05.")

    mc_run = st.button("▶ Run Monte Carlo", type="primary", disabled=mc_file is None)
    _mc_key = f"mc_result_{mc_n_sims}_{mc_block}_{mc_ppyr}_{mc_capital}_{mc_is_equity}_{mc_pct_format}"

    if mc_run and mc_file is not None:
        with st.spinner(f"Running {mc_n_sims:,} bootstrap simulations…"):
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(mc_file.getvalue())
                    tmp_path = tmp.name
                rets = load_returns_csv(tmp_path, from_equity=mc_is_equity, initial_capital=float(mc_capital))
                os.unlink(tmp_path)
                if mc_pct_format:
                    rets = rets / 100.0
                cfg = MCConfig(
                    n_simulations   = mc_n_sims,
                    block_size      = mc_block,
                    initial_capital = float(mc_capital),
                    periods_per_yr  = float(mc_ppyr),
                    seed            = 42,
                    target_sharpe   = mc_target_sharpe,
                    target_calmar   = mc_target_calmar,
                    target_max_dd   = mc_target_dd,
                )
                st.session_state[_mc_key] = run_mc(rets, cfg)
                st.success("Monte Carlo complete — navigate to the other tabs to explore results.")
            except Exception as exc:
                st.error(f"Monte Carlo failed: {exc}")

    mc_result = st.session_state.get(_mc_key)
    if mc_result is None:
        st.info(
            "Upload a CSV then click **▶ Run Monte Carlo**."
            if mc_file is None
            else "File loaded — click **▶ Run Monte Carlo** to start the analysis."
        )
    else:
        r = mc_result
        st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
        rs1, rs2, rs3, rs4, rs5, rs6 = st.columns(6)
        rs1.markdown(kpi_card("Periods",    str(r.n_periods),  accent=COLORS["neutral"]), unsafe_allow_html=True)
        rs2.markdown(kpi_card("Mean Ret",   f"{r.ret_mean:.5f}", accent=COLORS["blue"]),  unsafe_allow_html=True)
        rs3.markdown(kpi_card("Std Dev",    f"{r.ret_std:.5f}",  accent=COLORS["neutral"]), unsafe_allow_html=True)
        rs4.markdown(kpi_card("Skew",       f"{r.ret_skew:.3f}", accent=COLORS["warning"]), unsafe_allow_html=True)
        rs5.markdown(kpi_card("Kurt",       f"{r.ret_kurt:.3f}", accent=COLORS["warning"]), unsafe_allow_html=True)
        rs6.markdown(kpi_card("Block Size", str(r.block_size),  accent=COLORS["neutral"]), unsafe_allow_html=True)


# Helper to pull result from session state
def _mc_result():
    _key = f"mc_result_{st.session_state.get('mc_n_sims', 10_000)}_{st.session_state.get('mc_block', 10)}_{st.session_state.get('mc_ppyr', 252)}_{st.session_state.get('mc_capital', 100_000)}_{st.session_state.get('mc_is_equity', False)}_{st.session_state.get('mc_pct_format', False)}"
    return st.session_state.get(_key)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fan Chart
# ══════════════════════════════════════════════════════════════════════════════

with tab_fan:
    r = _mc_result()
    if r is None:
        st.info("Upload a CSV and run the simulation in the **Upload & Run** tab first.")
    else:
        mc_target_sharpe = st.session_state.get("mc_t_sharpe", 1.0)
        mc_capital = st.session_state.get("mc_capital", 100_000)

        # Terminal wealth KPIs
        tw1, tw2, tw3, tw4, tw5 = st.columns(5)
        tw1.markdown(kpi_card("P(Losing Money)",  f"{r.p_loss:.1%}",      accent=COLORS["negative"]), unsafe_allow_html=True)
        tw2.markdown(kpi_card("P(Loss > 25%)",    f"{r.p_loss_25pct:.1%}", accent=COLORS["negative"]), unsafe_allow_html=True)
        tw3.markdown(kpi_card("P(Doubling)",      f"{r.p_double:.1%}",    accent=COLORS["positive"]), unsafe_allow_html=True)
        tw4.markdown(kpi_card("Median Final",     f"${r.terminal_percentiles[50]:,.0f}", accent=COLORS["blue"]), unsafe_allow_html=True)
        tw5.markdown(kpi_card("P5 Final",         f"${r.terminal_percentiles[5]:,.0f}",  accent=COLORS["neutral"]), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
        st.markdown(section_label("Equity Fan Chart"), unsafe_allow_html=True)

        def _band_polygon(fan_x, y_lower, y_upper, fillcolor, name):
            xs = list(fan_x) + list(fan_x[::-1])
            ys = list(y_upper) + list(y_lower[::-1])
            return go.Scatter(x=xs, y=ys, fill="toself", fillcolor=fillcolor,
                              line=dict(width=0), mode="lines", name=name, showlegend=True, hoverinfo="skip")

        fig_fan = go.Figure()
        _px, _py = [], []
        _terminals = np.array([p[-1] for p in r.sample_paths])
        _lo, _hi   = np.percentile(_terminals, [5, 95])
        for _path in r.sample_paths:
            if not (_lo <= _path[-1] <= _hi):
                continue
            _px.extend(r.fan_x); _px.append(None)
            _py.extend(_path);   _py.append(None)
        fig_fan.add_trace(go.Scatter(x=_px, y=_py, mode="lines",
                                      line=dict(color="rgba(160,160,160,0.22)", width=0.7),
                                      showlegend=False, hoverinfo="skip"))
        fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p5,  r.fan_p95, "rgba(65,130,200,0.13)", "5th–95th"))
        fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p10, r.fan_p90, "rgba(55,115,190,0.19)", "10th–90th"))
        fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p25, r.fan_p75, "rgba(45,100,180,0.30)", "25th–75th"))
        fig_fan.add_trace(go.Scatter(x=r.fan_x, y=r.fan_p50, mode="lines",
                                      line=dict(color=COLORS["blue"], width=2.5), name="Median",
                                      hovertemplate="Period %{x}: $%{y:,.0f}<extra>Median</extra>"))
        fig_fan.add_trace(go.Scatter(x=r.fan_x, y=r.orig_equity, mode="lines",
                                      line=dict(color=COLORS["warning"], width=1.8, dash="dash"), name="Original",
                                      hovertemplate="Period %{x}: $%{y:,.0f}<extra>Original</extra>"))
        _y_lo = min(r.fan_p5); _y_hi = max(r.fan_p95)
        _pad = (_y_hi - _y_lo) * 0.05
        apply_theme(fig_fan, legend_inside=False)
        fig_fan.update_layout(height=380,
                               yaxis=dict(tickprefix="$", tickformat=",.0f",
                                          range=[min(_y_lo, min(r.orig_equity)) - _pad,
                                                 max(_y_hi, max(r.orig_equity)) + _pad]),
                               xaxis=dict(title="Period"))
        st.plotly_chart(fig_fan, use_container_width=True, config=PLOTLY_CONFIG)

        # Terminal wealth histogram
        st.markdown(section_label("Terminal Wealth Distribution"), unsafe_allow_html=True)
        _tw_arr = np.array(r.terminal_values)
        _tw_med = float(np.median(_tw_arr))
        _tw_orig = float(r.orig_equity[-1]) if r.orig_equity else float(mc_capital)
        fig_tw = go.Figure()
        fig_tw.add_trace(go.Histogram(x=_tw_arr / 1_000, nbinsx=80,
                                       marker=dict(color=COLORS["blue"], line=dict(width=0)), opacity=0.65,
                                       histnorm="probability density",
                                       hovertemplate="$%{x:,.0f}K: %{y:.4f}<extra></extra>"))
        fig_tw.add_vline(x=mc_capital / 1_000, line=dict(color=COLORS["negative"], width=1.5, dash="dot"),
                          annotation_text="Initial", annotation_font_size=9, annotation_position="top right")
        fig_tw.add_vline(x=_tw_med / 1_000, line=dict(color=COLORS["text"], width=1.5, dash="dash"),
                          annotation_text=f"Med ${_tw_med/1_000:,.0f}K", annotation_font_size=9)
        fig_tw.add_vline(x=_tw_orig / 1_000, line=dict(color=COLORS["warning"], width=2),
                          annotation_text=f"Actual ${_tw_orig/1_000:,.0f}K", annotation_font_size=9,
                          annotation_position="top left")
        apply_theme(fig_tw)
        fig_tw.update_layout(height=260, xaxis=dict(title="Terminal Wealth ($K)"), bargap=0.02)
        st.plotly_chart(fig_tw, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Metric Distributions
# ══════════════════════════════════════════════════════════════════════════════

with tab_dist:
    r = _mc_result()
    if r is None:
        st.info("Upload a CSV and run the simulation in the **Upload & Run** tab first.")
    else:
        mc_target_sharpe = st.session_state.get("mc_t_sharpe", 1.0)
        mc_target_calmar = st.session_state.get("mc_t_calmar", 0.5)
        mc_target_dd     = st.session_state.get("mc_t_dd", -0.20)

        _mc_dist_panels = [
            ("Sharpe Ratio",    r.sharpe_dist,   r.orig_sharpe,  COLORS["teal"],     mc_target_sharpe, ">="),
            ("Calmar Ratio",    r.calmar_dist,   r.orig_calmar,  COLORS["blue"],     mc_target_calmar, ">="),
            ("Max Drawdown",    [v*100 for v in r.max_dd_dist], r.orig_max_dd*100, COLORS["negative"], mc_target_dd*100, ">="),
            ("Ann. Volatility", [v*100 for v in r.ann_vol_dist], r.orig_ann_vol*100, COLORS["neutral"], None, None),
        ]
        dist_row1 = st.columns(2)
        dist_row2 = st.columns(2)
        for panel_idx, (title, dist, orig_val, color, target_val, direction) in enumerate(_mc_dist_panels):
            col = (dist_row1 if panel_idx < 2 else dist_row2)[panel_idx % 2]
            with col:
                med = float(np.median(dist))
                fig_d = go.Figure()
                fig_d.add_trace(go.Histogram(x=dist, nbinsx=80,
                                              marker=dict(color=color, line=dict(width=0)), opacity=0.65,
                                              name="Bootstrap", histnorm="probability density",
                                              hovertemplate="%{x:.3f}: %{y:.4f}<extra></extra>"))
                fig_d.add_vline(x=med, line=dict(color=COLORS["text"], width=1.5, dash="dash"),
                                annotation_text=f"Med {med:.2f}", annotation_font_size=9)
                fig_d.add_vline(x=orig_val, line=dict(color=COLORS["warning"], width=2),
                                annotation_text=f"Actual {orig_val:.2f}", annotation_font_size=9,
                                annotation_position="top left")
                if target_val is not None:
                    fig_d.add_vline(x=target_val, line=dict(color=COLORS["neutral"], width=1, dash="dot"),
                                    annotation_text=f"Target {target_val:.1f}", annotation_font_size=8)
                    prob = float((np.array(dist) >= target_val if direction == ">=" else np.array(dist) <= target_val).mean())
                    title_full = f"{title}  (P = {prob:.1%})"
                else:
                    title_full = title
                apply_theme(fig_d, legend_inside=True)
                fig_d.update_layout(height=240, title=dict(text=title_full, font=dict(size=11)),
                                     showlegend=False, bargap=0.02)
                st.plotly_chart(fig_d, use_container_width=True, config=PLOTLY_CONFIG)

        # Sortino
        _sort_cols = st.columns(2)
        with _sort_cols[0]:
            _sor_med = float(np.median(r.sortino_dist))
            fig_sor = go.Figure()
            fig_sor.add_trace(go.Histogram(x=r.sortino_dist, nbinsx=80,
                                            marker=dict(color=COLORS["purple"], line=dict(width=0)), opacity=0.65,
                                            histnorm="probability density"))
            fig_sor.add_vline(x=_sor_med, line=dict(color=COLORS["text"], width=1.5, dash="dash"),
                               annotation_text=f"Med {_sor_med:.2f}", annotation_font_size=9)
            fig_sor.add_vline(x=r.orig_sortino, line=dict(color=COLORS["warning"], width=2),
                               annotation_text=f"Actual {r.orig_sortino:.2f}", annotation_font_size=9,
                               annotation_position="top left")
            apply_theme(fig_sor)
            fig_sor.update_layout(height=240, title=dict(text="Sortino Ratio", font=dict(size=11)),
                                   showlegend=False, bargap=0.02)
            st.plotly_chart(fig_sor, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        st.markdown(section_label("Joint Target Achievement"), unsafe_allow_html=True)
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.markdown(kpi_card(f"P(Sharpe ≥ {mc_target_sharpe:.1f})", f"{r.p_meets_sharpe:.1%}", accent=COLORS["teal"]),     unsafe_allow_html=True)
        tc2.markdown(kpi_card(f"P(Calmar ≥ {mc_target_calmar:.1f})", f"{r.p_meets_calmar:.1%}", accent=COLORS["blue"]),     unsafe_allow_html=True)
        tc3.markdown(kpi_card(f"P(Max DD ≥ {mc_target_dd:.0%})",     f"{r.p_meets_max_dd:.1%}", accent=COLORS["warning"]),  unsafe_allow_html=True)
        tc4.markdown(kpi_card("P(All Three)",                          f"{r.p_meets_all:.1%}",    accent=COLORS["positive"]), unsafe_allow_html=True)

        target_labels = [f"Sharpe ≥ {mc_target_sharpe:.1f}", f"Calmar ≥ {mc_target_calmar:.1f}",
                          f"Max DD ≥ {mc_target_dd:.0%}", "Any Two", "All Three"]
        target_probs  = [r.p_meets_sharpe*100, r.p_meets_calmar*100, r.p_meets_max_dd*100,
                          r.p_meets_any_two*100, r.p_meets_all*100]
        fig_target = go.Figure(go.Bar(
            x=target_probs, y=target_labels, orientation="h",
            marker=dict(color=[COLORS["teal"], COLORS["blue"], COLORS["warning"], COLORS["neutral"], COLORS["positive"]],
                        line=dict(width=0)), opacity=0.75,
            text=[f"{v:.1f} %" for v in target_probs], textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        ))
        apply_theme(fig_target)
        fig_target.update_layout(height=220, xaxis=dict(range=[0, 105], title="Probability (%)", showgrid=False),
                                  yaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig_target, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

with tab_diag:
    r = _mc_result()
    if r is None:
        st.info("Upload a CSV and run the simulation in the **Upload & Run** tab first.")
    else:
        mc_target_sharpe = st.session_state.get("mc_t_sharpe", 1.0)

        st.markdown(section_label("Return Autocorrelation (Serial Dependence)"), unsafe_allow_html=True)
        st.caption(f"Bars outside ±{r.acf_ci:.3f} (95% CI) indicate significant serial dependence. "
                   "Significant ACF(r²) = volatility clustering — consider increasing block size.")
        _acf_lags = list(range(1, len(r.acf_returns) + 1))
        _acf_col1, _acf_col2 = st.columns(2)
        with _acf_col1:
            fig_acf = go.Figure(go.Bar(
                x=_acf_lags, y=r.acf_returns,
                marker=dict(color=[COLORS["positive"] if abs(v) > r.acf_ci else COLORS["neutral"] for v in r.acf_returns],
                            line=dict(width=0)), opacity=0.8,
                hovertemplate="Lag %{x}: %{y:.3f}<extra>ACF(r)</extra>"))
            for _y in [r.acf_ci, -r.acf_ci]:
                fig_acf.add_hline(y=_y, line=dict(color=COLORS["border"], width=1, dash="dot"))
            fig_acf.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_acf, title="ACF of Returns")
            fig_acf.update_layout(height=240, showlegend=False,
                                   xaxis=dict(title="Lag (periods)", showgrid=False, tickmode="linear"),
                                   yaxis=dict(title="Autocorrelation"))
            st.plotly_chart(fig_acf, use_container_width=True, config=PLOTLY_CONFIG)
        with _acf_col2:
            fig_acf2 = go.Figure(go.Bar(
                x=_acf_lags, y=r.acf_squared,
                marker=dict(color=[COLORS["warning"] if abs(v) > r.acf_ci else COLORS["neutral"] for v in r.acf_squared],
                            line=dict(width=0)), opacity=0.8,
                hovertemplate="Lag %{x}: %{y:.3f}<extra>ACF(r²)</extra>"))
            for _y in [r.acf_ci, -r.acf_ci]:
                fig_acf2.add_hline(y=_y, line=dict(color=COLORS["border"], width=1, dash="dot"))
            fig_acf2.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_acf2, title="ACF of Squared Returns (Volatility Clustering)")
            fig_acf2.update_layout(height=240, showlegend=False,
                                    xaxis=dict(title="Lag (periods)", showgrid=False, tickmode="linear"),
                                    yaxis=dict(title="Autocorrelation"))
            st.plotly_chart(fig_acf2, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
        st.markdown(section_label(f"Rolling Sharpe Stability (window = {r.rolling_window} periods)"), unsafe_allow_html=True)
        fig_roll = go.Figure()
        fig_roll.add_traces([
            go.Scatter(x=r.rolling_x, y=r.rolling_p10, line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=r.rolling_x, y=r.rolling_p90, fill="tonexty", fillcolor="rgba(38,120,178,0.10)",
                       line=dict(width=0), name="10th–90th"),
            go.Scatter(x=r.rolling_x, y=r.rolling_p50, line=dict(color=COLORS["blue"], width=1.5), name="Median"),
            go.Scatter(x=r.rolling_x, y=r.rolling_orig, line=dict(color=COLORS["warning"], width=1.5, dash="dash"), name="Original"),
        ])
        fig_roll.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        fig_roll.add_hline(y=mc_target_sharpe, line=dict(color=COLORS["neutral"], width=1, dash="dot"),
                            annotation_text=f"Target {mc_target_sharpe:.1f}", annotation_font_size=8)
        apply_theme(fig_roll, legend_inside=True)
        fig_roll.update_layout(height=300, xaxis=dict(title="Period"), yaxis=dict(title="Annualised Sharpe"))
        st.plotly_chart(fig_roll, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
        st.markdown(section_label("Convergence Diagnostics"), unsafe_allow_html=True)
        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            fig_cv1 = go.Figure()
            fig_cv1.add_traces([
                go.Scatter(x=r.conv_n, y=r.conv_p5,  line=dict(color=COLORS["neutral"], width=1, dash="dot"),  name="P5"),
                go.Scatter(x=r.conv_n, y=r.conv_p50, line=dict(color=COLORS["blue"],    width=2),              name="P50"),
                go.Scatter(x=r.conv_n, y=r.conv_p95, line=dict(color=COLORS["neutral"], width=1, dash="dot"),  name="P95"),
            ])
            apply_theme(fig_cv1, legend_inside=True)
            fig_cv1.update_layout(height=260, title="Terminal Equity Percentiles vs Simulation Count",
                                   xaxis=dict(type="log", title="# Simulations"),
                                   yaxis=dict(tickprefix="$", tickformat=",.0f"))
            st.plotly_chart(fig_cv1, use_container_width=True, config=PLOTLY_CONFIG)
        with cv_col2:
            _conv_sharpe_p50 = [float(np.median(np.array(r.sharpe_dist[:n]))) for n in r.conv_n] if hasattr(r, "conv_n") else []
            if _conv_sharpe_p50:
                fig_cv2 = go.Figure(go.Scatter(
                    x=r.conv_n, y=_conv_sharpe_p50,
                    mode="lines", line=dict(color=COLORS["teal"], width=2), name="Median Sharpe",
                ))
                apply_theme(fig_cv2, legend_inside=True)
                fig_cv2.update_layout(height=260, title="Median Sharpe vs Simulation Count",
                                       xaxis=dict(type="log", title="# Simulations"),
                                       yaxis=dict(title="Median Sharpe"))
                st.plotly_chart(fig_cv2, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.caption("Convergence data for Sharpe not available.")
