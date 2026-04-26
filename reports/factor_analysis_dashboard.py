"""Factor Analysis Dashboard — Research group, page 1.

Tabs:
  1. Universe Snapshot  — current momentum ranking and factor z-scores
  2. Factor Time-Series — per-symbol factor line charts
  3. IC Analysis        — information coefficient, distribution, lookback stability, Hurst
  4. Factor Correlation — cross-factor correlation heatmap
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from scipy.stats import norm as _scipy_norm, t as _scipy_t
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
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date
from features.compute import load_features

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


@st.cache_data(ttl=600, show_spinner=False)
def _hurst_for_factor(symbols_t: tuple, factor: str,
                      start_s: str, end_s: str) -> dict[str, float]:
    from research.long_memory import hurst_rs
    feats = _features(symbols_t, start_s, end_s)
    if feats is None:
        return {}
    out: dict[str, float] = {}
    for sym in symbols_t:
        pv = factor_pivot_from_features(feats, [sym], factor)
        if sym in pv.columns:
            arr = pv[sym].dropna().values
            if len(arr) >= 30:
                out[sym] = hurst_rs(arr)
    return out


# ── Page header + controls ─────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Factor Analysis",
        "Examine factor quality: universe snapshot, time-series, IC, and long-memory regime.",
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
        key="fa_lookback",
    )

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

features_df: pl.DataFrame | None = None
if _symbols:
    with st.spinner("Loading features…"):
        features_df = _features(_symbols, str(_start), str(_end))

tab_snap, tab_ts, tab_ic, tab_corr = st.tabs([
    "  Universe Snapshot  ",
    "  Factor Time-Series  ",
    "  IC Analysis  ",
    "  Factor Correlation  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Universe Snapshot
# ══════════════════════════════════════════════════════════════════════════════

with tab_snap:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        snap = get_snapshot(features_df)

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.markdown(kpi_card("Universe Size",       str(snap.n_universe),       accent=COLORS["teal"]),     unsafe_allow_html=True)
        _sc2.markdown(kpi_card("Symbols w/ Momentum", str(snap.n_valid_momentum), accent=COLORS["blue"]),     unsafe_allow_html=True)
        _sc3.markdown(kpi_card("Current Top-1", snap.top5_momentum[0] if snap.top5_momentum else "—", accent=COLORS["positive"]), unsafe_allow_html=True)
        _sc4.markdown(kpi_card("As-of Date",          snap.latest_date,           accent=COLORS["neutral"]),  unsafe_allow_html=True)

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
        _col_rank, _col_zheat = st.columns(2)

        with _col_rank:
            st.markdown(section_label("12-1 Momentum Ranking"), unsafe_allow_html=True)
            _mom = momentum_ranked(snap.snap_pd)
            if not _mom.empty:
                _bar_c = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in _mom.values]
                _fig_rank = go.Figure(go.Bar(
                    x=_mom.values, y=_mom.index.tolist(), orientation="h",
                    marker=dict(color=_bar_c, line=dict(width=0)),
                    text=[f"{v:.1%}" for v in _mom.values], textposition="outside",
                    textfont=dict(size=10, color=COLORS["text"]),
                    hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
                ))
                _fig_rank.add_vline(x=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(_fig_rank)
                _fig_rank.update_layout(height=max(280, 26 * len(_mom)), xaxis=dict(tickformat=".0%", showgrid=False), yaxis=dict(showgrid=False), showlegend=False)
                st.plotly_chart(_fig_rank, use_container_width=True, config=PLOTLY_CONFIG)

        with _col_zheat:
            st.markdown(section_label("Factor Z-Scores"), unsafe_allow_html=True)
            _z = snap.z_scores
            _z_cols = [c for c in _z.columns if _z[c].notna().any()]
            if _z_cols:
                _z_sorted = _z[_z_cols].sort_values("momentum_12m_1m", ascending=False) if "momentum_12m_1m" in _z_cols else _z[_z_cols]
                _z_xlabels = [FEATURE_LABELS.get(c, c) for c in _z_cols]
                _z_text = [[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in _z_sorted.values]
                _fig_zheat = go.Figure(go.Heatmap(
                    z=_z_sorted.values, x=_z_xlabels, y=_z_sorted.index.tolist(),
                    text=_z_text, texttemplate="%{text}", textfont=dict(size=9),
                    colorscale="RdYlGn", zmid=0, showscale=True,
                    colorbar=dict(thickness=12, len=0.85),
                    hovertemplate="<b>%{y}</b> → %{x}: %{z:.2f}σ<extra></extra>",
                ))
                apply_theme(_fig_zheat)
                _fig_zheat.update_layout(height=max(280, 26 * len(_z_sorted)), xaxis=dict(side="top"))
                st.plotly_chart(_fig_zheat, use_container_width=True, config=PLOTLY_CONFIG)

        _pf = snap.present_features
        if _pf:
            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
            _display = snap.snap_pd[_pf].copy()
            _display.index.name = "Symbol"
            _display.columns = [FEATURE_LABELS.get(c, c) for c in _display.columns]
            if "12-1 Momentum" in _display.columns:
                _display = _display.sort_values("12-1 Momentum", ascending=False)
            _grad_cols = [FEATURE_LABELS.get(f, f) for f in _pf if f != "dollar_volume_63d" and FEATURE_LABELS.get(f, f) in _display.columns]
            _fmt = {FEATURE_LABELS.get(f, f): ("${:,.0f}" if f == "dollar_volume_63d" else "{:.3f}") for f in _pf}
            st.dataframe(_display.style.background_gradient(cmap="RdYlGn", axis=0, subset=_grad_cols).format(_fmt), use_container_width=True, height=min(500, 38 * (len(_display) + 1)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Factor Time-Series
# ══════════════════════════════════════════════════════════════════════════════

with tab_ts:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        present = [f for f in ALL_FEATURES if f in features_df.columns]
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            selected_factor = st.selectbox("Factor", present, format_func=lambda x: FEATURE_LABELS.get(x, x), key="fa_ts_factor")
        with col_ctrl2:
            all_syms = sorted(features_df["symbol"].unique().to_list())
            selected_syms = st.multiselect("Symbols", all_syms, default=all_syms[:8], max_selections=15, key="fa_ts_syms")

        if not selected_syms:
            st.info("Select at least one symbol.")
        else:
            fac_pivot = factor_pivot_from_features(features_df, selected_syms, selected_factor)
            is_pct    = selected_factor in PCT_FEATURES

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
                height=400,
                xaxis=dict(rangeselector=range_selector()),
                hovermode="x unified",
                yaxis=dict(tickformat=".1%" if is_pct else ",.2f"),
                title=FEATURE_LABELS.get(selected_factor, selected_factor),
            )
            st.plotly_chart(fig_ts, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IC Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_ic:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        present = [f for f in ALL_FEATURES if f in features_df.columns]
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 2, 1, 1])
        with col_ctrl1:
            selected_factor = st.selectbox("Factor", present, format_func=lambda x: FEATURE_LABELS.get(x, x), key="fa_ic_factor")
        with col_ctrl2:
            all_syms = sorted(features_df["symbol"].unique().to_list())
            selected_syms = st.multiselect("Symbols", all_syms, default=all_syms[:8], max_selections=15, key="fa_ic_syms")
        with col_ctrl3:
            ic_window = st.selectbox("IC window (days)", [21, 63, 126], index=0,
                                     format_func=lambda x: f"{x}d fwd", key="fa_ic_window")
        with col_ctrl4:
            spy_overlay = st.checkbox("SPY regime overlay", value=False,
                                      help="Shade bear-market periods on the IC chart", key="fa_spy_overlay")

        if not selected_syms:
            st.info("Select at least one symbol.")
        else:
            fac_pivot = factor_pivot_from_features(features_df, selected_syms, selected_factor)
            is_pct    = selected_factor in PCT_FEATURES

            # Distribution
            col_dist, col_ic = st.columns(2)
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
                fig_dist.update_layout(height=280, xaxis=dict(tickformat=".1%" if is_pct else ",.2f"),
                                        yaxis=dict(title="Density"), bargap=0.03)
                st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Mean",  f"{fstats.mean:.4f}")
                c_b.metric("Std",   f"{fstats.std:.4f}")
                c_c.metric("Skew",  f"{fstats.skew:.2f}")
                c_d.metric("Kurt",  f"{fstats.kurt:.2f}")

            # IC chart
            with col_ic:
                st.markdown(section_label(f"Information Coefficient ({ic_window}d forward)"), unsafe_allow_html=True)
                prices_pl = _prices(_symbols, str(_start), str(_end))
                ic_result = None
                if prices_pl is not None:
                    price_piv = price_pivot_from_bars(prices_pl)
                    ic_result = compute_ic(fac_pivot, price_piv, ic_window)

                if ic_result and ic_result.values:
                    _ic_arr   = np.array(ic_result.values)
                    _n_sym    = max(len(selected_syms), 3)
                    _n_obs    = len(_ic_arr)
                    _ic_se    = np.sqrt(np.clip(1 - _ic_arr**2, 0, 1) / max(_n_sym - 2, 1))
                    _ic_t     = np.where(_ic_se > 0, _ic_arr / _ic_se, 0)
                    _ic_pval  = 2 * _scipy_t.sf(np.abs(_ic_t), df=max(_n_sym - 2, 1))
                    _ic_ci_hi = _ic_arr + 1.96 * _ic_se
                    _ic_ci_lo = _ic_arr - 1.96 * _ic_se
                    bar_colors = [
                        COLORS["positive"] if (v >= 0 and p < 0.05) else
                        COLORS["negative"] if (v <  0 and p < 0.05) else
                        COLORS["gold_dim"]
                        for v, p in zip(_ic_arr, _ic_pval)
                    ]
                    _ic_std       = float(np.std(_ic_arr))
                    _ic_se_mean   = _ic_std / np.sqrt(_n_obs) if _n_obs > 1 else float("nan")
                    _ci_lo_mean   = ic_result.mean_ic - 1.96 * _ic_se_mean
                    _ci_hi_mean   = ic_result.mean_ic + 1.96 * _ic_se_mean
                    _pct_sig      = int(100 * np.mean(_ic_pval < 0.05))

                    fig_ic = go.Figure()
                    if spy_overlay and prices_pl is not None:
                        try:
                            _spy_px = prices_pl.filter(pl.col("symbol") == "SPY").sort("date").to_pandas().set_index("date")["close"]
                            _spy_px.index = pd.to_datetime(_spy_px.index)
                            _spy_roll = _spy_px.rolling(252).apply(lambda x: x[-1] / x[0] - 1, raw=True)
                            _bear, _in_bear, _bear_start = _spy_roll < 0, False, None
                            for _dt, _is_bear in _bear.items():
                                if _is_bear and not _in_bear:
                                    _bear_start, _in_bear = _dt, True
                                elif not _is_bear and _in_bear:
                                    fig_ic.add_vrect(x0=str(_bear_start.date()), x1=str(_dt.date()),
                                                     fillcolor="rgba(255,75,75,0.08)", layer="below", line_width=0)
                                    _in_bear = False
                            if _in_bear and _bear_start:
                                fig_ic.add_vrect(x0=str(_bear_start.date()), x1=str(ic_result.dates[-1]),
                                                 fillcolor="rgba(255,75,75,0.08)", layer="below", line_width=0)
                        except Exception:
                            pass
                    fig_ic.add_trace(go.Scatter(
                        x=ic_result.dates + ic_result.dates[::-1],
                        y=list(_ic_ci_hi) + list(_ic_ci_lo[::-1]),
                        fill="toself", fillcolor="rgba(201,162,39,0.08)",
                        line=dict(width=0), showlegend=True,
                        name="95% CI", hoverinfo="skip",
                    ))
                    fig_ic.add_trace(go.Bar(
                        x=ic_result.dates, y=ic_result.values,
                        marker=dict(color=bar_colors, line=dict(width=0)), opacity=0.65,
                        name="IC (gold=NS, green/red=p<0.05)",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>IC</extra>",
                    ))
                    fig_ic.add_trace(go.Scatter(
                        x=ic_result.dates, y=ic_result.rolling_mean,
                        line=dict(color=COLORS["teal"], width=2), name="6-period MA",
                    ))
                    fig_ic.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                    apply_theme(fig_ic, legend_inside=True)
                    fig_ic.update_layout(height=300, yaxis=dict(tickformat=".2f", title="Rank IC"))
                    st.plotly_chart(fig_ic, use_container_width=True, config=PLOTLY_CONFIG)
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Mean IC", f"{ic_result.mean_ic:.4f}",
                               delta=f"95% CI [{_ci_lo_mean:.4f}, {_ci_hi_mean:.4f}]", delta_color="off")
                    c_b.metric("IC IR", f"{ic_result.icir:.3f}")
                    c_c.metric("% Significant",  f"{_pct_sig}%",
                               delta=f"(p<0.05, n={_n_obs} periods)", delta_color="off")
                else:
                    st.info("Could not compute IC — price data unavailable.")

            # IC across lookback windows
            if ic_result and ic_result.values:
                st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
                st.markdown(section_label("IC across Lookback Windows"), unsafe_allow_html=True)
                st.caption("Shows whether the factor was consistently predictive across different historical periods.")
                _lb_rows = []
                for _lb in [1, 3, 5, lookback_years]:
                    if _lb > lookback_years:
                        continue
                    _lb_start = date(_end.year - _lb, _end.month, _end.day)
                    try:
                        _lb_feat = _features(_symbols, str(_lb_start), str(_end))
                        if _lb_feat is None:
                            continue
                        _lb_fp   = factor_pivot_from_features(_lb_feat, selected_syms, selected_factor)
                        _lb_ic   = compute_ic(_lb_fp, price_piv, ic_window) if prices_pl is not None else None
                        if _lb_ic and _lb_ic.values:
                            _lv = np.array(_lb_ic.values)
                            _ls = np.sqrt(np.clip(1 - _lv**2, 0, 1) / max(len(selected_syms) - 2, 1))
                            _lp = 2 * _scipy_t.sf(np.abs(np.where(_ls > 0, _lv / _ls, 0)), df=max(len(selected_syms) - 2, 1))
                            _lb_rows.append({
                                "Lookback": f"{_lb}yr",
                                "Mean IC":  round(_lb_ic.mean_ic, 4),
                                "IC IR":    round(_lb_ic.icir, 3),
                                "N periods": len(_lv),
                                "% Sig (p<0.05)": f"{int(100 * np.mean(_lp < 0.05))}%",
                            })
                    except Exception:
                        pass
                if _lb_rows:
                    st.dataframe(pd.DataFrame(_lb_rows).style.format({"Mean IC": "{:+.4f}", "IC IR": "{:.3f}"}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("Not enough data for lookback comparison.")

                # Feature contribution to current signal
                st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
                st.markdown(section_label("Feature Contribution to Current Signal"), unsafe_allow_html=True)
                st.caption("Weighted average factor z-score across current portfolio positions.")
                try:
                    from config.settings import DATA_DIR as _DATA_DIR
                    _tw_path = _DATA_DIR / "gold" / "equity" / "target_weights.parquet"
                    if _tw_path.exists():
                        _tw_df   = pl.read_parquet(_tw_path).to_pandas()
                        _port_syms = _tw_df["symbol"].unique().tolist()
                        _latest  = features_df.filter(pl.col("date") == features_df["date"].max()).to_pandas()
                        _latest  = _latest[_latest["symbol"].isin(_port_syms)].set_index("symbol")
                        _wt      = _tw_df.groupby("symbol")["weight"].last()
                        _contrib = {}
                        for _ff in present:
                            if _ff in _latest.columns:
                                _merged_c = _latest[[_ff]].join(_wt, how="inner")
                                _contrib[FEATURE_LABELS.get(_ff, _ff)] = float(
                                    (_merged_c[_ff] * _merged_c["weight"]).sum()
                                )
                        if _contrib:
                            _cdf = pd.DataFrame([{"Feature": k, "Weighted Z-Score": v}
                                                  for k, v in sorted(_contrib.items(), key=lambda x: abs(x[1]), reverse=True)])
                            _cc = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in _cdf["Weighted Z-Score"]]
                            _fig_contrib = go.Figure(go.Bar(
                                x=_cdf["Feature"], y=_cdf["Weighted Z-Score"],
                                marker=dict(color=_cc, line=dict(width=0)),
                                text=[f"{v:+.3f}" for v in _cdf["Weighted Z-Score"]],
                                textposition="outside",
                                hovertemplate="<b>%{x}</b>: %{y:+.3f}<extra></extra>",
                            ))
                            _fig_contrib.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                            apply_theme(_fig_contrib, title="Portfolio-weighted factor z-score", height=240)
                            _fig_contrib.update_layout(showlegend=False)
                            st.plotly_chart(_fig_contrib, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.caption("No overlap between portfolio symbols and feature data.")
                    else:
                        st.caption("No target weights found — run a rebalance first.")
                except Exception as _ce:
                    st.caption(f"Contribution analysis unavailable: {_ce}")

            # Hurst exponent
            st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
            st.markdown(section_label("Long Memory (Hurst Exponent)"), unsafe_allow_html=True)
            st.caption("H > 0.55 = persistent / trending.  H ~ 0.50 = random walk.  H < 0.45 = mean-reverting.")
            _hurst_map = _hurst_for_factor(tuple(sorted(selected_syms)), selected_factor, str(_start), str(_end))
            if _hurst_map:
                from research.long_memory import hurst_label as _hlabel
                _h_syms = sorted(_hurst_map, key=lambda s: _hurst_map[s], reverse=True)
                _h_vals = [_hurst_map[s] for s in _h_syms]
                _h_lbls = [_hlabel(v) for v in _h_vals]
                _h_cols = [COLORS["positive"] if v > 0.55 else COLORS["negative"] if v < 0.45 else COLORS["neutral"] for v in _h_vals]
                fig_hurst = go.Figure(go.Bar(
                    x=_h_syms, y=_h_vals,
                    marker=dict(color=_h_cols, line=dict(width=0)),
                    text=[f"{v:.3f}  {l}" for v, l in zip(_h_vals, _h_lbls)],
                    textposition="outside",
                    textfont=dict(size=9, color=COLORS["text"]),
                    hovertemplate="<b>%{x}</b>: H = %{y:.3f}<extra></extra>",
                ))
                fig_hurst.add_hline(y=0.5, line=dict(color=COLORS["border"], width=1.5, dash="dot"),
                                    annotation_text="H=0.50 random walk",
                                    annotation_font=dict(size=9, color=COLORS["text_muted"]),
                                    annotation_position="top right")
                apply_theme(fig_hurst)
                fig_hurst.update_layout(height=260, yaxis=dict(range=[0, 1.05], showgrid=False, title="Hurst exponent"),
                                         xaxis=dict(showgrid=False), showlegend=False)
                st.plotly_chart(fig_hurst, use_container_width=True, config=PLOTLY_CONFIG)
                st.dataframe(pd.DataFrame({"Symbol": _h_syms, "H (R/S)": [f"{v:.4f}" for v in _h_vals], "Regime": _h_lbls}),
                             hide_index=True, use_container_width=True)
            else:
                st.caption("Not enough data to compute Hurst exponents (need ≥ 30 observations per symbol).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Factor Correlation
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        present = [f for f in ALL_FEATURES if f in features_df.columns]
        all_syms = sorted(features_df["symbol"].unique().to_list())
        corr_syms = st.multiselect("Symbols for correlation", all_syms, default=all_syms, key="fa_corr_syms")

        if len(present) >= 2 and corr_syms:
            try:
                _fcorr_frames = []
                for _f in present:
                    _pv = factor_pivot_from_features(features_df, corr_syms, _f)
                    _series = _pv.stack().rename(_f)
                    _fcorr_frames.append(_series)
                _fcorr_df = pd.concat(_fcorr_frames, axis=1).dropna()
                _fcorr = _fcorr_df.corr()
                _flabels = [FEATURE_LABELS.get(f, f) for f in _fcorr.columns]
                _ftext = [[f"{v:.2f}" for v in row] for row in _fcorr.values]
                fig_fcorr = go.Figure(go.Heatmap(
                    z=_fcorr.values, x=_flabels, y=_flabels,
                    text=_ftext, texttemplate="%{text}", textfont=dict(size=11),
                    colorscale=[[0.0, COLORS["negative"]], [0.5, COLORS["card_bg"]], [1.0, COLORS["positive"]]],
                    zmin=-1, zmax=1, showscale=True,
                    colorbar=dict(tickvals=[-1, 0, 1], thickness=14, len=0.9),
                    hovertemplate="<b>%{y} / %{x}</b>: %{text}<extra></extra>",
                ))
                apply_theme(fig_fcorr)
                fig_fcorr.update_layout(height=max(300, 60 * len(_fcorr)), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fcorr, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption("Values are cross-asset, cross-time Spearman correlations between factor time-series.")
            except Exception as _e:
                st.error(f"Factor correlation unavailable: {_e}")
        elif len(present) < 2:
            st.info("Need at least 2 features in the gold layer to compute correlations.")
        else:
            st.info("Select at least one symbol.")
