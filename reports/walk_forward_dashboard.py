"""Walk-Forward Validation Dashboard — Strategy group, page 2.

Tabs:
  1. Configuration  — training/test window, top-N, cost controls
  2. Fold Results   — per-fold Sharpe, CAGR, drawdown table and bar charts
  3. OOS Equity     — combined out-of-sample equity curve with fold bands
  4. Stability      — Sharpe scatter across folds, trend line, IS/OOS analysis
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, range_selector,
    kpi_card, section_label, page_header,
)
from research.walk_forward_runner import WFVConfig, fold_summary, oos_equity_normalised, run as run_wfv
from storage.universe import universe_as_of_date
from features.compute import load_features
from storage.parquet_store import load_bars

st.markdown(CSS, unsafe_allow_html=True)

# ── Cached runner ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _features(symbols: tuple, start_str: str, end_str: str):
    import polars as pl
    try:
        df = load_features(list(symbols), date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        return None if df.is_empty() else df
    except Exception:
        return None


@st.cache_data(ttl=600)
def _walk_forward(symbols, start_str, end_str, train_years, test_months, top_n, cost_bps):
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


# ── Page header + universe ────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Walk-Forward Validation",
        "Expanding-window out-of-sample backtesting to detect overfitting and measure regime robustness.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

_ctrl, _spacer = st.columns([2, 6])
with _ctrl:
    lookback_years = st.select_slider(
        "Feature lookback",
        options=[1, 2, 3, 4, 5, 6, 7],
        value=6,
        format_func=lambda x: f"{x} yr",
        key="wfv_lookback",
    )

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

tab_cfg, tab_folds, tab_equity, tab_stability = st.tabs([
    "  Configuration  ",
    "  Fold Results  ",
    "  OOS Equity  ",
    "  Stability  ",
])

# ── Shared: run walk-forward (triggered from Configuration tab) ────────────────

with tab_cfg:
    st.markdown(section_label("Walk-Forward Parameters"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    wfv_train = c1.slider("Training window (years)", 2, 5, 3, key="wfv_train")
    wfv_test  = c2.slider("Test window (months)", 3, 24, 12, key="wfv_test")
    wfv_top_n = c3.slider("Top-N positions", 3, 10, 5, key="wfv_top_n")
    wfv_cost  = c4.number_input("Cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="wfv_cost")

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
        folds       = wfv_result.folds
        rows        = fold_summary(wfv_result)
        sharpe_vals = [r["OOS Sharpe"] for r in rows]
        _hit_rate   = sum(v > 0 for v in sharpe_vals) / max(len(sharpe_vals), 1)
        _worst_sh   = min(sharpe_vals) if sharpe_vals else 0.0
        _hit_accent = (COLORS["positive"] if _hit_rate >= 0.6 else
                       COLORS["warning"]  if _hit_rate >= 0.5 else COLORS["negative"])

        st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(kpi_card("OOS Sharpe", f"{wfv_result.combined_sharpe:.3f}",        accent=COLORS["teal"]),     unsafe_allow_html=True)
        c2.markdown(kpi_card("OOS CAGR",   f"{wfv_result.combined_cagr:.1%}",          accent=COLORS["blue"]),     unsafe_allow_html=True)
        c3.markdown(kpi_card("OOS Max DD",  f"{wfv_result.combined_max_drawdown:.1%}",  accent=COLORS["negative"]), unsafe_allow_html=True)
        c4.markdown(kpi_card("Folds",       str(len(folds)),                            accent=COLORS["neutral"]),  unsafe_allow_html=True)
        c5.markdown(kpi_card("Hit Rate",    f"{_hit_rate:.0%}",                         accent=_hit_accent),        unsafe_allow_html=True)
        c6.markdown(kpi_card("Worst Fold",  f"{_worst_sh:.3f}",
                              accent=COLORS["positive"] if _worst_sh >= 0 else COLORS["negative"]), unsafe_allow_html=True)
        st.success("Walk-forward complete — navigate to the other tabs to explore results.")


# Pull result for other tabs (may be None or error string)
_wfv_key = f"wfv_{_symbols}_{st.session_state.get('wfv_train', 3)}_{st.session_state.get('wfv_test', 12)}_{st.session_state.get('wfv_top_n', 5)}_{st.session_state.get('wfv_cost', 5.0)}_{lookback_years}"
wfv_result = st.session_state.get(_wfv_key)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fold Results
# ══════════════════════════════════════════════════════════════════════════════

with tab_folds:
    if wfv_result is None or isinstance(wfv_result, str):
        st.info("Run the Walk-Forward Validation in the **Configuration** tab first.")
    else:
        rows        = fold_summary(wfv_result)
        sharpe_vals = [r["OOS Sharpe"] for r in rows]
        cagr_vals   = [r["OOS CAGR"]   for r in rows]
        dd_vals     = [r["OOS Max DD"]  for r in rows]
        fold_labels = [f"Fold {r['Fold']}  {r['Test Start'][:7]}" for r in rows]

        fold_df = pd.DataFrame(rows)
        fold_df["OOS CAGR"]   = fold_df["OOS CAGR"].map("{:.1%}".format)
        fold_df["OOS Max DD"] = fold_df["OOS Max DD"].map("{:.1%}".format)
        fold_df["OOS Vol"]    = fold_df["OOS Vol"].map("{:.1%}".format)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        col_sharpe_bar, col_cagr_bar = st.columns(2)

        with col_sharpe_bar:
            fig_sb = go.Figure(go.Bar(
                x=fold_labels, y=sharpe_vals,
                marker=dict(color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in sharpe_vals],
                            line=dict(width=0)),
                text=[f"{v:.2f}" for v in sharpe_vals], textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{x}</b>: %{y:.3f}<extra>Sharpe</extra>",
            ))
            fig_sb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_sb)
            fig_sb.update_layout(height=260, title="OOS Sharpe by Fold",
                                  yaxis=dict(showgrid=False), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_sb, use_container_width=True, config=PLOTLY_CONFIG)

        with col_cagr_bar:
            fig_cb = go.Figure(go.Bar(
                x=fold_labels, y=cagr_vals,
                marker=dict(color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in cagr_vals],
                            line=dict(width=0)),
                text=[f"{v:.1%}" for v in cagr_vals], textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertemplate="<b>%{x}</b>: %{y:.1%}<extra>CAGR</extra>",
            ))
            fig_cb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
            apply_theme(fig_cb)
            fig_cb.update_layout(height=260, title="OOS CAGR by Fold",
                                  yaxis=dict(tickformat=".0%", showgrid=False), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_cb, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown(section_label("Per-Fold Max Drawdown"), unsafe_allow_html=True)
        fig_dd = go.Figure(go.Bar(
            x=fold_labels, y=[v * 100 for v in dd_vals],
            marker=dict(color=COLORS["negative"], line=dict(width=0)),
            text=[f"{v:.1%}" for v in dd_vals], textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b>: %{y:.2f}%<extra>Max DD</extra>",
        ))
        fig_dd.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        apply_theme(fig_dd)
        fig_dd.update_layout(height=240, yaxis=dict(title="Max Drawdown (%)", showgrid=False),
                               xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OOS Equity
# ══════════════════════════════════════════════════════════════════════════════

with tab_equity:
    if wfv_result is None or isinstance(wfv_result, str):
        st.info("Run the Walk-Forward Validation in the **Configuration** tab first.")
    else:
        folds  = wfv_result.folds
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
        fig_eq.update_layout(height=420, yaxis=dict(tickformat="$,.0f"),
                               xaxis=dict(rangeselector=range_selector()))
        st.plotly_chart(fig_eq, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("Alternating shaded bands = individual test folds. $10,000 start.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stability
# ══════════════════════════════════════════════════════════════════════════════

with tab_stability:
    if wfv_result is None or isinstance(wfv_result, str):
        st.info("Run the Walk-Forward Validation in the **Configuration** tab first.")
    else:
        rows        = fold_summary(wfv_result)
        sharpe_vals = [r["OOS Sharpe"] for r in rows]

        if len(sharpe_vals) >= 2:
            _fold_seq  = list(range(1, len(rows) + 1))
            _slope, _intercept = np.polyfit(_fold_seq, sharpe_vals, 1)
            _trend_y   = [_slope * x + _intercept for x in _fold_seq]
            _trend_dir = ("negative" if _slope < -0.05 else "positive" if _slope >  0.05 else "flat")
            _trend_msg = {
                "negative": "Sharpe declining across folds — possible overfitting or regime change.",
                "positive": "Sharpe improving across folds — strategy adapts well to new data.",
                "flat":     "Sharpe stable across folds — regime-robust signal.",
            }[_trend_dir]

            _stab_col, _stab_cap = st.columns([2, 1])
            with _stab_col:
                fig_stab = go.Figure()
                fig_stab.add_trace(go.Scatter(
                    x=_fold_seq, y=_trend_y, mode="lines",
                    line=dict(color=COLORS["warning"], width=1.5, dash="dash"),
                    name=f"Trend (slope {_slope:+.3f})", hoverinfo="skip",
                ))
                fig_stab.add_trace(go.Scatter(
                    x=_fold_seq, y=sharpe_vals, mode="markers+text",
                    marker=dict(color=COLORS["gold"], size=10, line=dict(color=COLORS["text"], width=1)),
                    text=[f"F{i}" for i in _fold_seq], textposition="top center",
                    textfont=dict(size=9, color=COLORS["text"]),
                    name="OOS Sharpe",
                    hovertemplate="Fold %{x}: Sharpe = %{y:.3f}<extra></extra>",
                ))
                fig_stab.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_stab, legend_inside=True)
                fig_stab.update_layout(height=300,
                                        xaxis=dict(title="Fold", showgrid=False, tickmode="linear"),
                                        yaxis=dict(title="OOS Sharpe"))
                st.plotly_chart(fig_stab, use_container_width=True, config=PLOTLY_CONFIG)

            with _stab_cap:
                st.markdown(
                    f'<div style="margin-top:60px;color:{COLORS["neutral"]};font-size:0.82rem;">'
                    f"<b>Trend:</b> {_trend_dir} ({_slope:+.3f} per fold)<br><br>"
                    f"{_trend_msg}<br><br>"
                    f'<span style="color:{COLORS["text_muted"]};font-size:0.74rem;">'
                    "Negative slope signals overfitting. Flat/positive = regime-robust."
                    "</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
            st.markdown(section_label("IS vs OOS Sharpe — Overfitting Radar"), unsafe_allow_html=True)
            st.caption(
                "Each dot is one fold. Points below the diagonal (IS > OOS) signal overfitting. "
                "Points on or above it indicate the strategy generalises out-of-sample."
            )
            _is_sharpes  = [getattr(f, "is_sharpe", 0.0) for f in wfv_result.folds]
            _oos_sharpes = sharpe_vals

            _all_sh = _is_sharpes + _oos_sharpes
            _mn = min(_all_sh) - 0.2
            _mx = max(_all_sh) + 0.2
            _overfit_pct = sum(1 for i, o in zip(_is_sharpes, _oos_sharpes) if i > o) / max(len(rows), 1)

            fig_scatter = go.Figure()
            # Diagonal reference (IS == OOS)
            fig_scatter.add_trace(go.Scatter(
                x=[_mn, _mx], y=[_mn, _mx], mode="lines",
                line=dict(color=COLORS["border"], dash="dot", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))
            # Colour dots: green = OOS >= IS (good), red = OOS < IS (overfit)
            dot_colors = [COLORS["positive"] if o >= i else COLORS["negative"]
                          for i, o in zip(_is_sharpes, _oos_sharpes)]
            fig_scatter.add_trace(go.Scatter(
                x=_is_sharpes, y=_oos_sharpes, mode="markers+text",
                marker=dict(color=dot_colors, size=13, line=dict(color=COLORS["text"], width=1)),
                text=[f"F{i+1}" for i in range(len(rows))], textposition="top center",
                textfont=dict(size=9, color=COLORS["text"]),
                hovertemplate="Fold %{text}<br>IS Sharpe: %{x:.3f}<br>OOS Sharpe: %{y:.3f}<extra></extra>",
                showlegend=False,
            ))
            apply_theme(fig_scatter, legend_inside=False)
            fig_scatter.update_layout(
                height=340,
                xaxis=dict(title="In-Sample Sharpe", showgrid=False),
                yaxis=dict(title="OOS Sharpe", showgrid=False),
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)
            overfit_label = "⚠ Possible overfitting" if _overfit_pct > 0.5 else "✓ Generalises well"
            st.caption(f"{int(_overfit_pct*100)}% of folds show IS > OOS Sharpe — {overfit_label}")
        else:
            st.info("Need at least 2 folds for stability analysis.")
