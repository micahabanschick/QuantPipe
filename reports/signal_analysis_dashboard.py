"""Signal Analysis Dashboard — Research group, page 2.

Tabs:
  1. IC Decay         — IC at horizons 1, 5, 21, 63, 126d with significance bands
  2. Regime IC        — bull vs bear conditioned IC
  3. Turnover & Cost  — rebalance turnover and transaction cost simulation
  4. Composite Signal — factor weight slider and signal preview
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from scipy.stats import t as _scipy_t
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, kpi_card, section_label, page_header,
)
from research.signal_scanner import ALL_FEATURES, FEATURE_LABELS
from research.factor_analysis import factor_pivot_from_features, price_pivot_from_bars
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date
from features.compute import load_features
from signals.composite import composite_signal
from signals.analysis import ic_decay, signal_turnover
from signals.momentum import cross_sectional_momentum, get_monthly_rebalance_dates

st.markdown(CSS, unsafe_allow_html=True)

# ── Cached data loaders ────────────────────────────────────────────────────────

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


# ── Page header + universe load ────────────────────────────────────────────────

st.markdown(
    page_header(
        "Signal Analysis",
        "Measure how long a factor's predictive power lasts, how it behaves across regimes, and how much turnover it generates.",
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
        key="sa_lookback",
    )

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

features_df: pl.DataFrame | None = None
if _symbols:
    with st.spinner("Loading features…"):
        features_df = _features(_symbols, str(_start), str(_end))

tab_decay, tab_regime, tab_to, tab_composite = st.tabs([
    "  IC Decay  ",
    "  Regime IC  ",
    "  Turnover & Cost  ",
    "  Composite Signal  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IC Decay
# ══════════════════════════════════════════════════════════════════════════════

with tab_decay:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        _sa_present  = [f for f in ALL_FEATURES if f in features_df.columns]
        _sa_all_syms = sorted(features_df["symbol"].unique().to_list())

        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            sa_factor = st.selectbox("Factor", _sa_present, key="sa_decay_factor",
                                     format_func=lambda x: FEATURE_LABELS.get(x, x))
        with col_a2:
            sa_syms = st.multiselect("Symbols", _sa_all_syms, default=_sa_all_syms, key="sa_decay_syms")

        if sa_syms and sa_factor:
            _sa_prices_pl = _prices(_symbols, str(_start), str(_end))
            if _sa_prices_pl is not None:
                _sa_fac_pivot   = factor_pivot_from_features(features_df, sa_syms, sa_factor)
                _sa_price_pivot = price_pivot_from_bars(_sa_prices_pl)
                _sa_decay       = ic_decay(_sa_fac_pivot, _sa_price_pivot, horizons=[1, 5, 21, 63, 126])

                _sa_horizons = [p.horizon_days for p in _sa_decay]
                _sa_mean_ics = [p.mean_ic       for p in _sa_decay]

                _sa_tstats, _sa_pvals, _sa_ci_hi, _sa_ci_lo = [], [], [], []
                for p in _sa_decay:
                    _n = max(p.n_obs, 2)
                    _t = p.icir * (_n ** 0.5) if p.icir != 0 else 0.0
                    _pv = float(2 * _scipy_t.sf(abs(_t), df=_n - 1))
                    _se = abs(p.mean_ic / (p.icir * (_n ** 0.5))) if p.icir != 0 and _n > 0 else 0.0
                    _sa_tstats.append(round(_t, 3))
                    _sa_pvals.append(round(_pv, 4))
                    _sa_ci_hi.append(p.mean_ic + 1.96 * _se)
                    _sa_ci_lo.append(p.mean_ic - 1.96 * _se)

                fig_decay = go.Figure()
                fig_decay.add_trace(go.Scatter(
                    x=_sa_horizons + _sa_horizons[::-1],
                    y=_sa_ci_hi + _sa_ci_lo[::-1],
                    fill="toself", fillcolor="rgba(0,230,118,0.08)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                bar_cols = [COLORS["positive"] if p < 0.05 else
                            COLORS["warning"]  if p < 0.10 else
                            COLORS["neutral"]   for p in _sa_pvals]
                fig_decay.add_trace(go.Bar(
                    x=_sa_horizons, y=_sa_mean_ics,
                    marker=dict(color=bar_cols, line=dict(width=0)), opacity=0.75,
                    name="Mean IC (green=p<0.05, gold=p<0.10, grey=NS)",
                    hovertemplate="Horizon %{x}d: IC=%{y:.4f}<extra>Full sample</extra>",
                ))
                fig_decay.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_decay, legend_inside=True)
                fig_decay.update_layout(height=320,
                                         xaxis=dict(title="Horizon (days)", showgrid=False),
                                         yaxis=dict(title="Mean IC (Spearman)"))
                st.plotly_chart(fig_decay, use_container_width=True, config=PLOTLY_CONFIG)

                _decay_tbl = pd.DataFrame({
                    "Horizon": [f"{h}d" for h in _sa_horizons],
                    "Mean IC":  [f"{v:+.4f}" for v in _sa_mean_ics],
                    "95% CI":   [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(_sa_ci_lo, _sa_ci_hi)],
                    "t-stat":   [f"{t:+.2f}" for t in _sa_tstats],
                    "p-value":  [f"{pv:.4f}" for pv in _sa_pvals],
                    "Sig":      ["***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else "NS"
                                 for pv in _sa_pvals],
                    "IC IR":    [f"{p.icir:.3f}" for p in _sa_decay],
                    "N Obs":    [p.n_obs for p in _sa_decay],
                })
                st.dataframe(_decay_tbl, use_container_width=True, hide_index=True)
            else:
                st.info("Price data unavailable for IC decay computation.")
        else:
            st.info("Select a factor and at least one symbol.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regime IC
# ══════════════════════════════════════════════════════════════════════════════

with tab_regime:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        _sa_present  = [f for f in ALL_FEATURES if f in features_df.columns]
        _sa_all_syms = sorted(features_df["symbol"].unique().to_list())

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            sa_factor_r = st.selectbox("Factor", _sa_present, key="sa_regime_factor",
                                       format_func=lambda x: FEATURE_LABELS.get(x, x))
        with col_r2:
            sa_syms_r = st.multiselect("Symbols", _sa_all_syms, default=_sa_all_syms, key="sa_regime_syms")

        if sa_syms_r and sa_factor_r:
            _rg_prices_pl = _prices(_symbols, str(_start), str(_end))
            if _rg_prices_pl is not None:
                _rg_fac_pivot   = factor_pivot_from_features(features_df, sa_syms_r, sa_factor_r)
                _rg_price_pivot = price_pivot_from_bars(_rg_prices_pl)

                _bull_decay, _bear_decay, _full_decay = [], [], []
                try:
                    _spy_ser = (_rg_prices_pl
                                .filter(pl.col("symbol") == "SPY").sort("date")
                                .to_pandas().set_index("date")
                                ["adj_close" if "adj_close" in _rg_prices_pl.columns else "close"])
                    _spy_ser.index = pd.to_datetime(_spy_ser.index)
                    _spy_252 = _spy_ser.pct_change(252)
                    _bull_dates = set(_spy_252[_spy_252 >  0].dropna().index.date)
                    _bear_dates = set(_spy_252[_spy_252 <= 0].dropna().index.date)
                    _fp_bull = _rg_fac_pivot[_rg_fac_pivot.index.map(
                        lambda d: d.date() if hasattr(d, "date") else d).isin(_bull_dates)]
                    _fp_bear = _rg_fac_pivot[_rg_fac_pivot.index.map(
                        lambda d: d.date() if hasattr(d, "date") else d).isin(_bear_dates)]
                    _full_decay = ic_decay(_rg_fac_pivot, _rg_price_pivot, horizons=[1, 5, 21, 63, 126])
                    if len(_fp_bull) > 20:
                        _bull_decay = ic_decay(_fp_bull, _rg_price_pivot, horizons=[1, 5, 21, 63, 126])
                    if len(_fp_bear) > 20:
                        _bear_decay = ic_decay(_fp_bear, _rg_price_pivot, horizons=[1, 5, 21, 63, 126])
                except Exception as _e:
                    st.warning(f"Regime split failed: {_e}")

                fig_rg = go.Figure()
                if _full_decay:
                    fig_rg.add_trace(go.Scatter(
                        x=[p.horizon_days for p in _full_decay],
                        y=[p.mean_ic       for p in _full_decay],
                        mode="lines+markers", line=dict(color=COLORS["neutral"], width=2),
                        marker=dict(size=8), name="Full sample",
                    ))
                if _bull_decay:
                    fig_rg.add_trace(go.Scatter(
                        x=[p.horizon_days for p in _bull_decay],
                        y=[p.mean_ic       for p in _bull_decay],
                        mode="lines+markers", line=dict(color=COLORS["positive"], width=2, dash="dot"),
                        marker=dict(size=7), name="Bull regime (SPY 252d > 0)",
                    ))
                if _bear_decay:
                    fig_rg.add_trace(go.Scatter(
                        x=[p.horizon_days for p in _bear_decay],
                        y=[p.mean_ic       for p in _bear_decay],
                        mode="lines+markers", line=dict(color=COLORS["negative"], width=2, dash="dot"),
                        marker=dict(size=7), name="Bear regime (SPY 252d ≤ 0)",
                    ))
                fig_rg.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_rg, legend_inside=True)
                fig_rg.update_layout(height=360,
                                      xaxis=dict(title="Horizon (days)", showgrid=False),
                                      yaxis=dict(title="Mean IC (Spearman)"),
                                      title="IC Decay by Market Regime")
                st.plotly_chart(fig_rg, use_container_width=True, config=PLOTLY_CONFIG)

                if _full_decay and (_bull_decay or _bear_decay):
                    _rg_rows = []
                    for _h_idx, _horizon in enumerate([1, 5, 21, 63, 126]):
                        _row = {"Horizon": f"{_horizon}d"}
                        if _full_decay and _h_idx < len(_full_decay):
                            _row["Full"] = f"{_full_decay[_h_idx].mean_ic:+.4f}"
                        if _bull_decay and _h_idx < len(_bull_decay):
                            _row["Bull"] = f"{_bull_decay[_h_idx].mean_ic:+.4f}"
                        if _bear_decay and _h_idx < len(_bear_decay):
                            _row["Bear"] = f"{_bear_decay[_h_idx].mean_ic:+.4f}"
                        _rg_rows.append(_row)
                    st.dataframe(pd.DataFrame(_rg_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Price data unavailable.")
        else:
            st.info("Select a factor and at least one symbol.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Turnover & Cost
# ══════════════════════════════════════════════════════════════════════════════

with tab_to:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        _to_prices_pl = _prices(_symbols, str(_start), str(_end))
        _to_sig_df: pl.DataFrame | None = None
        if _to_prices_pl is not None:
            try:
                _to_td = sorted(_to_prices_pl["date"].unique().to_list())
                _to_rd = get_monthly_rebalance_dates(_start, _end, _to_td)
                _to_sig_df = cross_sectional_momentum(features_df, _to_rd, top_n=5)
            except Exception:
                _to_sig_df = None

        if _to_sig_df is not None and not _to_sig_df.is_empty():
            _to_result = signal_turnover(_to_sig_df)

            col_to_kpi, col_to_chart = st.columns([1, 3])
            with col_to_kpi:
                st.markdown(kpi_card("Mean Turnover", f"{_to_result.mean_turnover:.1%}", accent=COLORS["blue"]), unsafe_allow_html=True)
            with col_to_chart:
                _to_dates_str = [str(d) for d in _to_result.rebal_dates]
                fig_to = go.Figure(go.Bar(
                    x=_to_dates_str, y=_to_result.turnover,
                    marker=dict(color=COLORS["blue"], line=dict(width=0)), opacity=0.8,
                    hovertemplate="%{x}: %{y:.1%}<extra>Turnover</extra>",
                ))
                fig_to.add_hline(y=_to_result.mean_turnover,
                                  line=dict(color=COLORS["warning"], width=2, dash="dot"),
                                  annotation_text=f"Mean {_to_result.mean_turnover:.1%}",
                                  annotation_font=dict(color=COLORS["warning"], size=10),
                                  annotation_position="top right")
                apply_theme(fig_to)
                fig_to.update_layout(height=260, yaxis=dict(tickformat=".0%", title="Turnover"),
                                      xaxis=dict(showgrid=False), showlegend=False)
                st.plotly_chart(fig_to, use_container_width=True, config=PLOTLY_CONFIG)

            st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
            st.markdown(section_label("Turnover Cost Simulation"), unsafe_allow_html=True)
            st.caption("Annual cost = 12 rebalances × mean_turnover × cost (one-way) × 2 (round-trip).")

            _tc1, _tc2 = st.columns([1, 2])
            with _tc1:
                _cost_bps = st.select_slider("Cost per trade (bps, one-way)",
                                              options=[1, 2, 5, 10, 15, 20, 30, 50], value=5, key="sa_cost_bps")
                _assumed_vol = st.number_input("Assumed annual vol (%)", min_value=5.0, max_value=50.0,
                                                value=15.0, step=1.0, key="sa_vol",
                                                help="Used to estimate Sharpe drag from costs.")
            _to_val = _to_result.mean_turnover
            _annual_cost_pct = 12 * _to_val * 2 * _cost_bps / 10_000
            _sharpe_drag     = _annual_cost_pct / (_assumed_vol / 100)
            with _tc2:
                _rows = []
                for _bps in [1, 2, 5, 10, 20, 30, 50]:
                    _ac = 12 * _to_val * 2 * _bps / 10_000
                    _sd = _ac / (_assumed_vol / 100)
                    _rows.append({"Cost (bps)": _bps, "Annual drag (%)": f"{_ac*100:.3f}%",
                                   "Sharpe reduction": f"{_sd:.3f}", "Selected": "<<" if _bps == _cost_bps else ""})
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

            _cc1, _cc2, _cc3 = st.columns(3)
            _cc1.markdown(kpi_card("Mean Turnover",    f"{_to_val:.1%}",              accent=COLORS["blue"]),     unsafe_allow_html=True)
            _cc2.markdown(kpi_card("Annual Cost Drag", f"{_annual_cost_pct*100:.3f}%",
                                    accent=COLORS["negative"] if _annual_cost_pct > 0.01 else COLORS["positive"]), unsafe_allow_html=True)
            _cc3.markdown(kpi_card("Sharpe Reduction", f"{_sharpe_drag:.3f}",         accent=COLORS["warning"]),  unsafe_allow_html=True)
        else:
            st.info("Could not compute signal turnover — price data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Composite Signal
# ══════════════════════════════════════════════════════════════════════════════

with tab_composite:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        _cs_present  = [f for f in ALL_FEATURES if f in features_df.columns]

        _FEATURE_DEFAULT_WEIGHTS = {
            "momentum_12m_1m":  1.0,
            "realized_vol_21d": 0.0,
            "log_return_1d":    0.0,
            "dollar_volume_63d":0.0,
            "reversal_5d":      0.0,
        }

        col_sliders, col_preview = st.columns([1, 2])
        _factor_weights: dict[str, float] = {}
        with col_sliders:
            for feat in _cs_present:
                fw = st.slider(
                    FEATURE_LABELS.get(feat, feat),
                    min_value=-1.0, max_value=1.0,
                    value=_FEATURE_DEFAULT_WEIGHTS.get(feat, 0.0),
                    step=0.1, key=f"cs_fw_{feat}",
                )
                _factor_weights[feat] = fw

            _cs_top_n    = st.slider("Top-N", min_value=3, max_value=10, value=5, key="cs_top_n")
            _preview_btn = st.button("Preview Composite", type="primary", key="cs_preview")

        with col_preview:
            if _preview_btn:
                _active_weights = {f: w for f, w in _factor_weights.items() if abs(w) > 1e-9}
                if not _active_weights:
                    st.warning("Set at least one factor weight > 0.")
                else:
                    _cs_prices_pl = _prices(_symbols, str(_start), str(_end))
                    if _cs_prices_pl is not None:
                        _cs_td  = sorted(_cs_prices_pl["date"].unique().to_list())
                        _cs_rd  = get_monthly_rebalance_dates(_start, _end, _cs_td)
                        _cs_sig = composite_signal(features_df, _cs_rd, _active_weights, top_n=_cs_top_n)

                        if not _cs_sig.is_empty():
                            _latest_rd  = _cs_sig["rebalance_date"].max()
                            _latest_day = _cs_sig.filter(pl.col("rebalance_date") == _latest_rd).sort("rank")
                            _cs_syms    = _latest_day["symbol"].to_list()
                            _cs_scores  = _latest_day["composite_score"].to_list()
                            _cs_sel     = _latest_day["selected"].to_list()

                            fig_cs = go.Figure(go.Bar(
                                x=_cs_scores, y=_cs_syms, orientation="h",
                                marker=dict(color=[COLORS["positive"] if s else COLORS["neutral"] for s in _cs_sel],
                                            line=dict(width=0)),
                                text=[f"{v:.3f}" for v in _cs_scores], textposition="outside",
                                textfont=dict(size=10, color=COLORS["text"]),
                                hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
                            ))
                            fig_cs.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
                            apply_theme(fig_cs)
                            fig_cs.update_layout(
                                height=max(280, 26 * len(_cs_syms)),
                                xaxis=dict(showgrid=False, title="Composite Score"),
                                yaxis=dict(showgrid=False, autorange="reversed"),
                                showlegend=False,
                                title=f"Composite Scores — {_latest_rd}",
                            )
                            st.plotly_chart(fig_cs, use_container_width=True, config=PLOTLY_CONFIG)

                            _sel_df = _latest_day.filter(pl.col("selected")).to_pandas()
                            if not _sel_df.empty:
                                st.dataframe(
                                    _sel_df[["symbol", "composite_score", "rank"]].rename(columns={
                                        "symbol": "Symbol", "composite_score": "Composite Score", "rank": "Rank",
                                    }),
                                    use_container_width=True, hide_index=True,
                                )
                        else:
                            st.info("Composite signal returned no data.")
                    else:
                        st.info("Price data unavailable.")
            else:
                st.info("Configure factor weights above and click **Preview Composite**.")
