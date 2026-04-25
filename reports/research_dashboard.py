"""Research dashboard - Streamlit page (Dashboard #4).

UI-only layer. All analytics logic lives in research/.

Tabs:
  Signal Scanner  - current factor rankings and universe snapshot
  Factor Analysis - factor time-series, distribution, and information coefficient
  Walk-Forward    - OOS validation with per-fold Sharpe breakdown
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
from research.walk_forward_runner import WFVConfig, fold_summary, oos_equity_normalised, run as run_wfv
from research.monte_carlo import MCConfig, load_returns_csv, run as run_mc
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date
from features.compute import load_features
from signals.composite import composite_signal
from signals.analysis import ic_decay, signal_turnover
from signals.momentum import cross_sectional_momentum, get_monthly_rebalance_dates
from research.kalman_filter import kalman_smooth_betas, KalmanResult
from risk.factor_model import (
    FACTOR_PROXIES, estimate_factor_returns, rolling_factor_betas,
)

st.markdown(CSS, unsafe_allow_html=True)

# ?? Cached data loaders ???????????????????????????????????????????????????????

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


@st.cache_data(ttl=300)
def _kalman_compute(
    sym: str, delta: float, ols_window: int,
    start_str: str, end_str: str, symbols_tuple: tuple,
) -> tuple[KalmanResult, pd.DataFrame]:
    """Return (KalmanResult, rolling_ols_df). All params explicit for cache key."""
    bars = _prices(symbols_tuple, start_str, end_str)
    if bars is None:
        return KalmanResult(), pd.DataFrame()
    fr = estimate_factor_returns(bars, FACTOR_PROXIES)
    if fr.returns.empty:
        return KalmanResult(), pd.DataFrame()
    price_col = "adj_close" if "adj_close" in bars.columns else "close"
    port_pd = (
        bars.filter(pl.col("symbol") == sym)
        .sort("date")
        .to_pandas()
        .set_index("date")[price_col]
        .pct_change()
        .dropna()
    )
    port_pd.index = pd.to_datetime(port_pd.index)
    kr  = kalman_smooth_betas(port_pd, fr.returns, delta=delta)
    ols = rolling_factor_betas(port_pd, fr, window=ols_window,
                               min_periods=max(ols_window // 2, 20))
    return kr, ols


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


# ?? Page header ???????????????????????????????????????????????????????????????

st.markdown(
    page_header(
        "Research",
        "Analyse factor quality and signal robustness before committing to a strategy.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# ?? Controls ??????????????????????????????????????????????????????????????????

_ctrl, _spacer = st.columns([2, 6])
with _ctrl:
    lookback_years = st.select_slider(
        "Feature lookback",
        options=[1, 2, 3, 4, 5, 6, 7],
        value=6,
        format_func=lambda x: f"{x} yr",
        key="research_lookback",
    )

# ?? Universe + feature load ???????????????????????????????????????????????????

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

features_df: pl.DataFrame | None = None
if _symbols:
    with st.spinner("Loading features?"):
        features_df = _features(_symbols, str(_start), str(_end))

tab_factor, tab_signal_analysis, tab_wfv, tab_mc, tab_kalman = st.tabs(
    ["  Factor Analysis  ", "  Signal Analysis  ", "  Walk-Forward  ", "  Monte Carlo  ", "  Kalman Filter  "]
)

# ???????????????????????????????????????????????????????????????????????????????
# TAB 3 - FACTOR ANALYSIS  (merges former Signal Scanner)
# ???????????????????????????????????????????????????????????????????????????????

with tab_factor:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        # ?? Universe Snapshot (formerly Signal Scanner) ???????????????????????
        snap = get_snapshot(features_df)
        with st.expander(f"Universe Snapshot ? {snap.latest_date}", expanded=False):
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            _sc1.markdown(kpi_card("Universe Size",       str(snap.n_universe),       accent=COLORS["teal"]),     unsafe_allow_html=True)
            _sc2.markdown(kpi_card("Symbols w/ Momentum", str(snap.n_valid_momentum), accent=COLORS["blue"]),     unsafe_allow_html=True)
            _sc3.markdown(kpi_card("Current Top-1", snap.top5_momentum[0] if snap.top5_momentum else "-", accent=COLORS["positive"]), unsafe_allow_html=True)
            _sc4.markdown(kpi_card("As-of Date",          snap.latest_date,           accent=COLORS["neutral"]), unsafe_allow_html=True)
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
                    st.plotly_chart(_fig_rank, width="stretch", config=PLOTLY_CONFIG)
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
                        hovertemplate="<b>%{y}</b> ? %{x}: %{z:.2f}sigma<extra></extra>",
                    ))
                    apply_theme(_fig_zheat)
                    _fig_zheat.update_layout(height=max(280, 26 * len(_z_sorted)), xaxis=dict(side="top"))
                    st.plotly_chart(_fig_zheat, width="stretch", config=PLOTLY_CONFIG)
            # Full snapshot table
            _pf = snap.present_features
            if _pf:
                _display = snap.snap_pd[_pf].copy()
                _display.index.name = "Symbol"
                _display.columns = [FEATURE_LABELS.get(c, c) for c in _display.columns]
                if "12-1 Momentum" in _display.columns:
                    _display = _display.sort_values("12-1 Momentum", ascending=False)
                _grad_cols = [FEATURE_LABELS.get(f, f) for f in _pf if f != "dollar_volume_63d" and FEATURE_LABELS.get(f, f) in _display.columns]
                _fmt = {FEATURE_LABELS.get(f, f): ("${:,.0f}" if f == "dollar_volume_63d" else "{:.3f}") for f in _pf}
                st.dataframe(_display.style.background_gradient(cmap="RdYlGn", axis=0, subset=_grad_cols).format(_fmt), width="stretch", height=min(500, 38 * (len(_display) + 1)))
        st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
        present = [f for f in ALL_FEATURES if f in features_df.columns]

        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 2, 1, 1])
        with col_ctrl1:
            selected_factor = st.selectbox("Factor", present, format_func=lambda x: FEATURE_LABELS.get(x, x))
        with col_ctrl2:
            all_syms = sorted(features_df["symbol"].unique().to_list())
            selected_syms = st.multiselect("Symbols", all_syms, default=all_syms[:8], max_selections=15)
        with col_ctrl3:
            ic_window = st.selectbox("IC window (days)", [21, 63, 126], index=0,
                                     format_func=lambda x: f"{x}d fwd")
        with col_ctrl4:
            spy_overlay = st.checkbox("SPY regime overlay", value=False, help="Shade bear-market periods on the IC chart")

        if not selected_syms:
            st.info("Select at least one symbol.")
        else:
            fac_pivot = factor_pivot_from_features(features_df, selected_syms, selected_factor)
            is_pct    = selected_factor in PCT_FEATURES

            # ?? Time series ???????????????????????????????????????????????????
            st.markdown(section_label(f"{FEATURE_LABELS.get(selected_factor, selected_factor)} - Time Series"), unsafe_allow_html=True)
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

            # ?? Distribution ??????????????????????????????????????????????????
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

            # ?? Information Coefficient ???????????????????????????????????????
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

                    # Per-observation SE and p-value (t-test for Spearman correlation)
                    _ic_se    = np.sqrt(np.clip(1 - _ic_arr**2, 0, 1) / max(_n_sym - 2, 1))
                    _ic_t     = np.where(_ic_se > 0, _ic_arr / _ic_se, 0)
                    _ic_pval  = 2 * _scipy_t.sf(np.abs(_ic_t), df=max(_n_sym - 2, 1))
                    _ic_ci_hi = _ic_arr + 1.96 * _ic_se
                    _ic_ci_lo = _ic_arr - 1.96 * _ic_se

                    # Colour by significance: green=sig positive, red=sig negative, gold=NS
                    bar_colors = [
                        COLORS["positive"] if (v >= 0 and p < 0.05) else
                        COLORS["negative"] if (v <  0 and p < 0.05) else
                        COLORS["gold_dim"]
                        for v, p in zip(_ic_arr, _ic_pval)
                    ]

                    # Mean IC confidence interval (analytical)
                    _ic_std       = float(np.std(_ic_arr))
                    _ic_se_mean   = _ic_std / np.sqrt(_n_obs) if _n_obs > 1 else float("nan")
                    _ci_lo_mean   = ic_result.mean_ic - 1.96 * _ic_se_mean
                    _ci_hi_mean   = ic_result.mean_ic + 1.96 * _ic_se_mean
                    _pct_sig      = int(100 * np.mean(_ic_pval < 0.05))

                    fig_ic = go.Figure()

                    # SPY bear regime shading
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

                    # 95 % CI band
                    fig_ic.add_trace(go.Scatter(
                        x=ic_result.dates + ic_result.dates[::-1],
                        y=list(_ic_ci_hi) + list(_ic_ci_lo[::-1]),
                        fill="toself",
                        fillcolor="rgba(201,162,39,0.08)",
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
                    st.plotly_chart(fig_ic, width="stretch", config=PLOTLY_CONFIG)

                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Mean IC",
                               f"{ic_result.mean_ic:.4f}",
                               delta=f"95% CI [{_ci_lo_mean:.4f}, {_ci_hi_mean:.4f}]",
                               delta_color="off")
                    c_b.metric("IC IR", f"{ic_result.icir:.3f}")
                    c_c.metric("% Significant",  f"{_pct_sig}%",
                               delta=f"(p<0.05, n={_n_obs} periods)",
                               delta_color="off")

                    # ?? IC across lookback windows ??????????????????????????????
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
                            _lb_fp   = factor_pivot_from_features(_lb_feat, selected_syms, selected_feature)
                            _lb_ic   = compute_ic(_lb_fp, price_piv, ic_window) if price_piv is not None else None
                            if _lb_ic and _lb_ic.values:
                                _lv = np.array(_lb_ic.values)
                                _ls = np.sqrt(np.clip(1 - _lv**2, 0, 1) / max(_n_sym - 2, 1))
                                _lp = 2 * _scipy_t.sf(np.abs(np.where(_ls > 0, _lv / _ls, 0)), df=max(_n_sym - 2, 1))
                                _lb_rows.append({
                                    "Lookback": f"{_lb}yr",
                                    "Mean IC":  round(_lb_ic.mean_ic, 4),
                                    "IC IR":    round(_lb_ic.icir, 3),
                                    "N periods":len(_lv),
                                    "% Sig (p<0.05)": f"{int(100 * np.mean(_lp < 0.05))}%",
                                })
                        except Exception:
                            pass
                    if _lb_rows:
                        _lb_df = pd.DataFrame(_lb_rows)
                        st.dataframe(
                            _lb_df.style.format({"Mean IC": "{:+.4f}", "IC IR": "{:.3f}"}),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.caption("Not enough data for lookback comparison.")

                    # ?? Feature contribution decomposition ??????????????????????
                    st.markdown(section_label("Feature Contribution to Current Signal"), unsafe_allow_html=True)
                    st.caption("Weighted average factor z-score across the current portfolio positions.")
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
                                _cdf = pd.DataFrame([
                                    {"Feature": k, "Weighted Z-Score": v}
                                    for k, v in sorted(_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
                                ])
                                _cc = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                                       for v in _cdf["Weighted Z-Score"]]
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
                                st.plotly_chart(_fig_contrib, width="stretch", config=PLOTLY_CONFIG)
                            else:
                                st.caption("No overlap between portfolio symbols and feature data.")
                        else:
                            st.caption("No target weights found - run a rebalance first.")
                    except Exception as _ce:
                        st.caption(f"Contribution analysis unavailable: {_ce}")

                else:
                    st.info("Could not compute IC - price data unavailable.")

            # ?? Factor Correlation Heatmap ?????????????????????????????????????
            if len(present) >= 2 and selected_syms:
                st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
                st.markdown(section_label("Factor Correlation"), unsafe_allow_html=True)
                try:
                    _fcorr_frames = []
                    for _f in present:
                        _pv = factor_pivot_from_features(features_df, selected_syms, _f)
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
                    fig_fcorr.update_layout(height=max(220, 50 * len(_fcorr)), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_fcorr, width="stretch", config=PLOTLY_CONFIG)
                except Exception:
                    st.info("Factor correlation unavailable.")


# ???????????????????????????????????????????????????????????????????????????????
# TAB 4 - SIGNAL ANALYSIS
# ???????????????????????????????????????????????????????????????????????????????

with tab_signal_analysis:
    if features_df is None:
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        _sa_present = [f for f in ALL_FEATURES if f in features_df.columns]
        _sa_all_syms = sorted(features_df["symbol"].unique().to_list())

        # ?? Section A: IC Decay ???????????????????????????????????????????????
        st.markdown(section_label("IC Decay Curve"), unsafe_allow_html=True)

        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            sa_factor = st.selectbox(
                "Factor", _sa_present,
                key="sa_factor",
                format_func=lambda x: FEATURE_LABELS.get(x, x),
            )
        with col_a2:
            sa_syms = st.multiselect(
                "Symbols", _sa_all_syms, default=_sa_all_syms,
                key="sa_syms",
            )

        if sa_syms and sa_factor:
            _sa_prices_pl = _prices(_symbols, str(_start), str(_end))
            if _sa_prices_pl is not None:
                _sa_fac_pivot   = factor_pivot_from_features(features_df, sa_syms, sa_factor)
                _sa_price_pivot = price_pivot_from_bars(_sa_prices_pl)
                _sa_decay = ic_decay(_sa_fac_pivot, _sa_price_pivot, horizons=[1, 5, 21, 63, 126])

                _sa_horizons  = [p.horizon_days for p in _sa_decay]
                _sa_mean_ics  = [p.mean_ic for p in _sa_decay]
                _sa_icirs     = [p.icir for p in _sa_decay]

                fig_decay = go.Figure()
                fig_decay.add_trace(go.Scatter(
                    x=_sa_horizons, y=_sa_mean_ics,
                    mode="lines+markers",
                    line=dict(color=COLORS["teal"], width=2.5),
                    marker=dict(size=8, color=COLORS["teal"]),
                    name="Mean IC",
                    hovertemplate="Horizon %{x}d: IC=%{y:.4f}<extra></extra>",
                ))
                fig_decay.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_decay)
                fig_decay.update_layout(
                    height=260,
                    xaxis=dict(title="Horizon (days)", showgrid=False),
                    yaxis=dict(title="Mean IC (Spearman)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_decay, width="stretch", config=PLOTLY_CONFIG)

                # Table
                _decay_tbl = pd.DataFrame({
                    "Horizon (days)": _sa_horizons,
                    "Mean IC":        [f"{v:.4f}" for v in _sa_mean_ics],
                    "IC IR":          [f"{p.icir:.3f}" for p in _sa_decay],
                    "N Obs":          [p.n_obs for p in _sa_decay],
                })
                st.dataframe(_decay_tbl, width="stretch", hide_index=True)
            else:
                st.info("Price data unavailable for IC decay computation.")

        st.markdown("<div style='height:18px'/>", unsafe_allow_html=True)

        # ?? Section B: Signal Turnover ????????????????????????????????????????
        st.markdown(section_label("Signal Turnover"), unsafe_allow_html=True)

        _sb_prices_pl = _prices(_symbols, str(_start), str(_end))
        _sb_sig_df: pl.DataFrame | None = None
        if _sb_prices_pl is not None:
            try:
                _sb_td = sorted(_sb_prices_pl["date"].unique().to_list())
                _sb_rd = get_monthly_rebalance_dates(_start, _end, _sb_td)
                _sb_sig_df = cross_sectional_momentum(features_df, _sb_rd, top_n=5)
            except Exception:
                _sb_sig_df = None

        if _sb_sig_df is not None and not _sb_sig_df.is_empty():
            _sb_to = signal_turnover(_sb_sig_df)

            col_to_kpi, col_to_chart = st.columns([1, 3])
            with col_to_kpi:
                st.markdown(
                    kpi_card("Mean Turnover", f"{_sb_to.mean_turnover:.1%}", accent=COLORS["blue"]),
                    unsafe_allow_html=True,
                )

            with col_to_chart:
                _sb_dates_str = [str(d) for d in _sb_to.rebal_dates]
                _sb_to_vals   = _sb_to.turnover
                fig_to = go.Figure(go.Bar(
                    x=_sb_dates_str,
                    y=_sb_to_vals,
                    marker=dict(color=COLORS["blue"], line=dict(width=0)),
                    opacity=0.8,
                    name="Turnover",
                    hovertemplate="%{x}: %{y:.1%}<extra>Turnover</extra>",
                ))
                fig_to.add_hline(
                    y=_sb_to.mean_turnover,
                    line=dict(color=COLORS["warning"], width=2, dash="dot"),
                    annotation_text=f"Mean {_sb_to.mean_turnover:.1%}",
                    annotation_font=dict(color=COLORS["warning"], size=10),
                    annotation_position="top right",
                )
                apply_theme(fig_to)
                fig_to.update_layout(
                    height=240,
                    yaxis=dict(tickformat=".0%", title="Turnover"),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_to, width="stretch", config=PLOTLY_CONFIG)
        else:
            st.info("Could not compute signal turnover - price data unavailable.")

        st.markdown("<div style='height:18px'/>", unsafe_allow_html=True)

        # ?? Section C: Composite Signal Builder ???????????????????????????????
        st.markdown(section_label("Composite Signal Builder"), unsafe_allow_html=True)

        _FEATURE_DEFAULT_WEIGHTS = {
            "momentum_12m_1m": 1.0,
            "realized_vol_21d": 0.0,
            "log_return_1d": 0.0,
            "dollar_volume_63d": 0.0,
            "reversal_5d": 0.0,
        }

        col_sliders, col_preview = st.columns([1, 2])
        _factor_weights: dict[str, float] = {}
        with col_sliders:
            for feat in _sa_present:
                default_val = _FEATURE_DEFAULT_WEIGHTS.get(feat, 0.0)
                fw = st.slider(
                    FEATURE_LABELS.get(feat, feat),
                    min_value=-1.0, max_value=1.0,
                    value=default_val, step=0.1,
                    key=f"fw_{feat}",
                )
                _factor_weights[feat] = fw

            _cs_top_n = st.slider("Top-N", min_value=3, max_value=10, value=5, key="cs_top_n")
            _preview_btn = st.button("Preview Composite", type="primary", key="cs_preview")

        with col_preview:
            if _preview_btn:
                _active_weights = {f: w for f, w in _factor_weights.items() if abs(w) > 1e-9}
                if not _active_weights:
                    st.warning("Set at least one factor weight > 0.")
                else:
                    _cs_prices_pl = _prices(_symbols, str(_start), str(_end))
                    if _cs_prices_pl is not None:
                        _cs_td = sorted(_cs_prices_pl["date"].unique().to_list())
                        _cs_rd = get_monthly_rebalance_dates(_start, _end, _cs_td)
                        _cs_sig = composite_signal(
                            features_df, _cs_rd, _active_weights, top_n=_cs_top_n
                        )

                        if not _cs_sig.is_empty():
                            _latest_rd = _cs_sig["rebalance_date"].max()
                            _latest_day = _cs_sig.filter(
                                pl.col("rebalance_date") == _latest_rd
                            ).sort("rank")

                            _cs_syms  = _latest_day["symbol"].to_list()
                            _cs_scores = _latest_day["composite_score"].to_list()
                            _cs_sel    = _latest_day["selected"].to_list()

                            bar_colors = [
                                COLORS["positive"] if s else COLORS["neutral"]
                                for s in _cs_sel
                            ]
                            fig_cs = go.Figure(go.Bar(
                                x=_cs_scores,
                                y=_cs_syms,
                                orientation="h",
                                marker=dict(color=bar_colors, line=dict(width=0)),
                                text=[f"{v:.3f}" for v in _cs_scores],
                                textposition="outside",
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
                                title=f"Composite Scores - {_latest_rd}",
                            )
                            st.plotly_chart(fig_cs, width="stretch", config=PLOTLY_CONFIG)

                            # Selected symbols table
                            _sel_df = _latest_day.filter(pl.col("selected")).to_pandas()
                            if not _sel_df.empty:
                                st.dataframe(
                                    _sel_df[["symbol", "composite_score", "rank"]].rename(columns={
                                        "symbol": "Symbol",
                                        "composite_score": "Composite Score",
                                        "rank": "Rank",
                                    }),
                                    width="stretch",
                                    hide_index=True,
                                )
                        else:
                            st.info("Composite signal returned no data.")
                    else:
                        st.info("Price data unavailable.")
            else:
                st.info("Configure factor weights above and click **Preview Composite**.")


# ???????????????????????????????????????????????????????????????????????????????
# TAB 4 - WALK-FORWARD
# ???????????????????????????????????????????????????????????????????????????????

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

    run_btn = st.button("? Run Walk-Forward Validation", type="primary")
    _key    = f"wfv_{_symbols}_{wfv_train}_{wfv_test}_{wfv_top_n}_{wfv_cost}_{lookback_years}"

    if run_btn:
        with st.spinner("Running walk-forward validation - this may take 30-90 seconds?"):
            st.session_state[_key] = _walk_forward(
                _symbols, str(_start), str(_end),
                wfv_train, wfv_test, wfv_top_n, float(wfv_cost),
            )

    wfv_result = st.session_state.get(_key)

    if wfv_result is None:
        st.info("Configure parameters above and click **? Run Walk-Forward Validation** to start.")
    elif isinstance(wfv_result, str):
        st.error(f"Walk-forward failed: {wfv_result}")
    else:
        folds = wfv_result.folds

        # ?? Summary KPIs ??????????????????????????????????????????????????????
        st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("OOS Sharpe", f"{wfv_result.combined_sharpe:.3f}",        accent=COLORS["teal"]),     unsafe_allow_html=True)
        c2.markdown(kpi_card("OOS CAGR",   f"{wfv_result.combined_cagr:.1%}",          accent=COLORS["blue"]),     unsafe_allow_html=True)
        c3.markdown(kpi_card("OOS Max DD",  f"{wfv_result.combined_max_drawdown:.1%}",  accent=COLORS["negative"]), unsafe_allow_html=True)
        c4.markdown(kpi_card("Folds",       str(len(folds)),                            accent=COLORS["neutral"]),  unsafe_allow_html=True)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

        # ?? OOS equity curve ??????????????????????????????????????????????????
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

        # ?? Per-fold table ????????????????????????????????????????????????????
        st.markdown(section_label("Per-Fold Performance"), unsafe_allow_html=True)
        rows = fold_summary(wfv_result)
        fold_df = pd.DataFrame(rows)
        fold_df["OOS CAGR"]   = fold_df["OOS CAGR"].map("{:.1%}".format)
        fold_df["OOS Max DD"] = fold_df["OOS Max DD"].map("{:.1%}".format)
        fold_df["OOS Vol"]    = fold_df["OOS Vol"].map("{:.1%}".format)
        st.dataframe(fold_df, width="stretch", hide_index=True)

        # ?? Per-fold bar charts ???????????????????????????????????????????????
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

# ???????????????????????????????????????????????????????????????????????????????
# TAB 5 - MONTE CARLO BOOTSTRAP
# ???????????????????????????????????????????????????????????????????????????????

with tab_mc:
    st.markdown(section_label("Monte Carlo Block-Bootstrap Analysis"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{COLORS["neutral"]};font-size:0.83rem;margin-bottom:14px;">'
        "Upload a CSV of trade returns or an equity curve to run circular block-bootstrap "
        "analysis across 10 000+ simulated paths. Covers: equity fan chart ? metric "
        "distributions ? terminal wealth ? convergence diagnostics ? rolling Sharpe stability."
        "</div>",
        unsafe_allow_html=True,
    )

    # ?? Config ????????????????????????????????????????????????????????????????
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    with mc_col1:
        mc_n_sims   = st.selectbox("Simulations", [1_000, 5_000, 10_000, 25_000],
                                    index=2, format_func=lambda x: f"{x:,}")
    with mc_col2:
        mc_block    = st.slider("Block size", 3, 63, 10,
                                help="Block length for circular block bootstrap (match rebalance cadence)")
    with mc_col3:
        mc_ppyr     = st.selectbox("Periods / year",
                                    [252, 52, 26, 12],
                                    format_func=lambda x: {252:"Daily (252)",52:"Weekly (52)",
                                                            26:"Biweekly (26)",12:"Monthly (12)"}[x])
    with mc_col4:
        mc_capital  = st.number_input("Initial capital ($)", min_value=1_000,
                                       max_value=10_000_000, value=100_000, step=1_000)

    mc_t_col1, mc_t_col2, mc_t_col3 = st.columns(3)
    mc_target_sharpe = mc_t_col1.number_input("Target Sharpe", value=1.0, step=0.1, format="%.1f")
    mc_target_calmar = mc_t_col2.number_input("Target Calmar", value=0.5, step=0.1, format="%.1f")
    mc_target_dd     = mc_t_col3.number_input("Max DD target (e.g. ?0.20)", value=-0.20, step=0.05, format="%.2f")

    # ?? File upload ???????????????????????????????????????????????????????????
    mc_file      = st.file_uploader(
        "Upload CSV (trade log or equity curve)",
        type=["csv"],
        help=(
            "Accepted formats:\n"
            "? QC trade log - columns: Entry Price, Exit Price, P&L, Exit Time\n"
            "? Return series - column named 'return', 'returns', or 'ret'\n"
            "? Equity curve - column named 'equity', 'value', 'nav', etc."
        ),
    )
    ck_col1, ck_col2 = st.columns(2)
    mc_is_equity  = ck_col1.checkbox("CSV is an equity curve (not return series)", value=False)
    mc_pct_format = ck_col2.checkbox(
        "Returns are in % format (? 100)",
        value=False,
        help="Check if your return column stores e.g. 5.0 for a 5 % gain instead of 0.05.",
    )

    mc_run = st.button("? Run Monte Carlo", type="primary", disabled=mc_file is None)
    _mc_key = f"mc_result_{mc_n_sims}_{mc_block}_{mc_ppyr}_{mc_capital}_{mc_is_equity}_{mc_pct_format}"

    if mc_run and mc_file is not None:
        with st.spinner(f"Running {mc_n_sims:,} bootstrap simulations?"):
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
                mc_result = run_mc(rets, cfg)
                st.session_state[_mc_key] = mc_result
            except Exception as exc:
                st.error(f"Monte Carlo failed: {exc}")
                mc_result = None
    else:
        mc_result = st.session_state.get(_mc_key)

    if mc_result is None:
        st.info(
            "Upload a CSV then click **? Run Monte Carlo**."
            if mc_file is None
            else "File loaded - click **? Run Monte Carlo** to start the analysis."
        )
        st.stop()

    r = mc_result

    # ?? Return series stats ???????????????????????????????????????????????????
    st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
    rs1, rs2, rs3, rs4, rs5, rs6 = st.columns(6)
    rs1.markdown(kpi_card("Periods",    str(r.n_periods),          accent=COLORS["neutral"]),  unsafe_allow_html=True)
    rs2.markdown(kpi_card("Mean Ret",   f"{r.ret_mean:.5f}",       accent=COLORS["blue"]),     unsafe_allow_html=True)
    rs3.markdown(kpi_card("Std Dev",    f"{r.ret_std:.5f}",        accent=COLORS["neutral"]),  unsafe_allow_html=True)
    rs4.markdown(kpi_card("Skew",       f"{r.ret_skew:.3f}",       accent=COLORS["warning"]),  unsafe_allow_html=True)
    rs5.markdown(kpi_card("Kurt",       f"{r.ret_kurt:.3f}",       accent=COLORS["warning"]),  unsafe_allow_html=True)
    rs6.markdown(kpi_card("Block Size", str(r.block_size),         accent=COLORS["neutral"]),  unsafe_allow_html=True)

    st.markdown("<div style='height:18px'/>", unsafe_allow_html=True)

    # ?? Section A: Equity Fan Chart ????????????????????????????????????????????
    st.markdown(section_label("Equity Fan Chart"), unsafe_allow_html=True)

    def _band_polygon(fan_x, y_lower, y_upper, fillcolor, name):
        """Closed-polygon band - immune to fill='tonexty' ordering issues."""
        xs = list(fan_x) + list(fan_x[::-1])
        ys = list(y_upper) + list(y_lower[::-1])
        return go.Scatter(
            x=xs, y=ys,
            fill="toself", fillcolor=fillcolor,
            line=dict(width=0),
            mode="lines", name=name,
            showlegend=True, hoverinfo="skip",
        )

    fig_fan = go.Figure()

    # Layer 1: all sampled simulation paths in faint gray (single trace via None separators)
    _px: list = []
    _py: list = []
    for _path in r.sample_paths:
        _px.extend(r.fan_x)
        _px.append(None)
        _py.extend(_path)
        _py.append(None)
    fig_fan.add_trace(go.Scatter(
        x=_px, y=_py,
        mode="lines",
        line=dict(color="rgba(160,160,160,0.22)", width=0.7),
        showlegend=False, hoverinfo="skip", name="_sim_paths",
    ))

    # Layer 2: percentile bands (outermost first so inner bands render on top)
    fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p5,  r.fan_p95, "rgba(65,130,200,0.13)", "5th-95th"))
    fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p10, r.fan_p90, "rgba(55,115,190,0.19)", "10th-90th"))
    fig_fan.add_trace(_band_polygon(r.fan_x, r.fan_p25, r.fan_p75, "rgba(45,100,180,0.30)", "25th-75th"))

    # Layer 3: median line + original path on top
    fig_fan.add_trace(go.Scatter(
        x=r.fan_x, y=r.fan_p50,
        mode="lines", line=dict(color=COLORS["blue"], width=2.5),
        name="Median",
        hovertemplate="Period %{x}: $%{y:,.0f}<extra>Median</extra>",
    ))
    fig_fan.add_trace(go.Scatter(
        x=r.fan_x, y=r.orig_equity,
        mode="lines", line=dict(color=COLORS["warning"], width=1.8, dash="dash"),
        name="Original",
        hovertemplate="Period %{x}: $%{y:,.0f}<extra>Original</extra>",
    ))

    # Clamp y-axis to P5-P95 range with 5% padding so outlier paths don't
    # distort the scale. The original path is allowed to exceed this window.
    _y_lo = min(r.fan_p5)
    _y_hi = max(r.fan_p95)
    _orig_min = min(r.orig_equity)
    _orig_max = max(r.orig_equity)
    _pad = (_y_hi - _y_lo) * 0.05
    _range_lo = min(_y_lo, _orig_min) - _pad
    _range_hi = max(_y_hi, _orig_max) + _pad

    apply_theme(fig_fan, legend_inside=False)
    fig_fan.update_layout(
        height=360,
        yaxis=dict(tickprefix="$", tickformat=",.0f", range=[_range_lo, _range_hi]),
        xaxis=dict(title="Period"),
    )
    st.plotly_chart(fig_fan, width="stretch", config=PLOTLY_CONFIG)

    # ?? Section B: Terminal Wealth + Metric Distributions ?????????????????????
    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Terminal Wealth & Metric Distributions"), unsafe_allow_html=True)

    # KPIs
    tw1, tw2, tw3, tw4, tw5 = st.columns(5)
    tw1.markdown(kpi_card("P(Losing Money)",   f"{r.p_loss:.1%}",      accent=COLORS["negative"]), unsafe_allow_html=True)
    tw2.markdown(kpi_card("P(Loss > 25 %)",    f"{r.p_loss_25pct:.1%}", accent=COLORS["negative"]), unsafe_allow_html=True)
    tw3.markdown(kpi_card("P(Doubling)",        f"{r.p_double:.1%}",    accent=COLORS["positive"]), unsafe_allow_html=True)
    tw4.markdown(kpi_card("Median Final",       f"${r.terminal_percentiles[50]:,.0f}", accent=COLORS["blue"]), unsafe_allow_html=True)
    tw5.markdown(kpi_card("P5 Final",           f"${r.terminal_percentiles[5]:,.0f}",  accent=COLORS["neutral"]), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)

    # Metric distribution charts (2 x 2)
    _mc_dist_panels = [
        ("Sharpe Ratio",    r.sharpe_dist,   r.orig_sharpe,  COLORS["teal"],     mc_target_sharpe, ">="),
        ("Calmar Ratio",    r.calmar_dist,   r.orig_calmar,  COLORS["blue"],     mc_target_calmar, ">="),
        ("Max Drawdown",    [v*100 for v in r.max_dd_dist], r.orig_max_dd*100, COLORS["negative"], mc_target_dd*100, ">="),
        ("Ann. Volatility", [v*100 for v in r.ann_vol_dist], r.orig_ann_vol*100, COLORS["neutral"], None, None),
    ]
    dist_row1, dist_row2 = st.columns(2), st.columns(2)
    for panel_idx, (title, dist, orig_val, color, target_val, direction) in enumerate(_mc_dist_panels):
        col = (dist_row1 if panel_idx < 2 else dist_row2)[panel_idx % 2]
        with col:
            med = float(np.median(dist))
            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(
                x=dist, nbinsx=80,
                marker=dict(color=color, line=dict(width=0)), opacity=0.65,
                name="Bootstrap", histnorm="probability density",
                hovertemplate="%{x:.3f}: %{y:.4f}<extra></extra>",
            ))
            fig_d.add_vline(x=med, line=dict(color=COLORS["text"], width=1.5, dash="dash"),
                            annotation_text=f"Med {med:.2f}", annotation_font_size=9)
            fig_d.add_vline(x=orig_val, line=dict(color=COLORS["warning"], width=2),
                            annotation_text=f"Actual {orig_val:.2f}", annotation_font_size=9,
                            annotation_position="top left")
            if target_val is not None:
                fig_d.add_vline(x=target_val, line=dict(color=COLORS["neutral"], width=1, dash="dot"),
                                annotation_text=f"Target {target_val:.1f}", annotation_font_size=8)
                if direction == ">=":
                    prob = float((np.array(dist) >= target_val).mean())
                else:
                    prob = float((np.array(dist) <= target_val).mean())
                title_full = f"{title}  (P = {prob:.1%})"
            else:
                title_full = title
            apply_theme(fig_d, legend_inside=True)
            fig_d.update_layout(
                height=240, title=dict(text=title_full, font=dict(size=11)),
                showlegend=False, bargap=0.02,
            )
            st.plotly_chart(fig_d, width="stretch", config=PLOTLY_CONFIG)

    # ?? Section C: Target Achievement ?????????????????????????????????????????
    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Joint Target Achievement"), unsafe_allow_html=True)

    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.markdown(kpi_card(f"P(Sharpe >= {mc_target_sharpe:.1f})",    f"{r.p_meets_sharpe:.1%}",  accent=COLORS["teal"]),     unsafe_allow_html=True)
    tc2.markdown(kpi_card(f"P(Calmar >= {mc_target_calmar:.1f})",    f"{r.p_meets_calmar:.1%}",  accent=COLORS["blue"]),     unsafe_allow_html=True)
    tc3.markdown(kpi_card(f"P(Max DD >= {mc_target_dd:.0%})",        f"{r.p_meets_max_dd:.1%}",  accent=COLORS["warning"]),  unsafe_allow_html=True)
    tc4.markdown(kpi_card("P(All Three)",                             f"{r.p_meets_all:.1%}",     accent=COLORS["positive"]), unsafe_allow_html=True)

    # Horizontal bar chart
    target_labels = [
        f"Sharpe >= {mc_target_sharpe:.1f}",
        f"Calmar >= {mc_target_calmar:.1f}",
        f"Max DD >= {mc_target_dd:.0%}",
        "Any Two",
        "All Three",
    ]
    target_probs = [
        r.p_meets_sharpe * 100,
        r.p_meets_calmar * 100,
        r.p_meets_max_dd * 100,
        r.p_meets_any_two * 100,
        r.p_meets_all * 100,
    ]
    bar_colors_t = [
        COLORS["teal"], COLORS["blue"], COLORS["warning"], COLORS["neutral"], COLORS["positive"]
    ]
    fig_target = go.Figure(go.Bar(
        x=target_probs, y=target_labels, orientation="h",
        marker=dict(color=bar_colors_t, line=dict(width=0)), opacity=0.75,
        text=[f"{v:.1f} %" for v in target_probs], textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))
    apply_theme(fig_target)
    fig_target.update_layout(
        height=220, xaxis=dict(range=[0, 105], title="Probability (%)", showgrid=False),
        yaxis=dict(showgrid=False), showlegend=False,
    )
    st.plotly_chart(fig_target, width="stretch", config=PLOTLY_CONFIG)

    # ?? Section D: Rolling Sharpe Stability ???????????????????????????????????
    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
    st.markdown(section_label(f"Rolling Sharpe Stability (window = {r.rolling_window} periods)"), unsafe_allow_html=True)

    fig_roll = go.Figure()
    fig_roll.add_traces([
        go.Scatter(x=r.rolling_x, y=r.rolling_p10, line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=r.rolling_x, y=r.rolling_p90, fill="tonexty", fillcolor="rgba(38,120,178,0.10)",
                   line=dict(width=0), name="10th-90th"),
        go.Scatter(x=r.rolling_x, y=r.rolling_p50, line=dict(color=COLORS["blue"], width=1.5), name="Median"),
        go.Scatter(x=r.rolling_x, y=r.rolling_orig, line=dict(color=COLORS["warning"], width=1.5, dash="dash"),
                   name="Original"),
    ])
    fig_roll.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    fig_roll.add_hline(y=mc_target_sharpe, line=dict(color=COLORS["neutral"], width=1, dash="dot"),
                       annotation_text=f"Target {mc_target_sharpe:.1f}", annotation_font_size=8)
    apply_theme(fig_roll, legend_inside=True)
    fig_roll.update_layout(height=280, xaxis=dict(title="Period"), yaxis=dict(title="Annualized Sharpe"))
    st.plotly_chart(fig_roll, width="stretch", config=PLOTLY_CONFIG)

    # ?? Section E: Convergence ????????????????????????????????????????????????
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
        fig_cv1.update_layout(
            height=240, title="Terminal Equity Percentiles vs Simulation Count",
            xaxis=dict(type="log", title="# Simulations"),
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig_cv1, width="stretch", config=PLOTLY_CONFIG)

    with cv_col2:
        fig_cv2 = go.Figure(go.Scatter(
            x=r.conv_n, y=r.conv_sharpe_p50,
            line=dict(color=COLORS["teal"], width=2), name="Sharpe P50",
        ))
        apply_theme(fig_cv2, legend_inside=True)
        fig_cv2.update_layout(
            height=240, title="Median Sharpe vs Simulation Count",
            xaxis=dict(type="log", title="# Simulations"),
            yaxis=dict(title="Sharpe (median)"),
        )
        st.plotly_chart(fig_cv2, width="stretch", config=PLOTLY_CONFIG)

    # ?? Section F: Summary Table ??????????????????????????????????????????????
    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)
    st.markdown(section_label("Summary Table"), unsafe_allow_html=True)

    _sum_df = pd.DataFrame(r.summary_rows)
    _sum_display = _sum_df.copy()
    for c in ["sharpe", "calmar", "sortino"]:
        if c in _sum_display.columns:
            _sum_display[c] = _sum_display[c].apply(lambda v: f"{v:.3f}")
    for c in ["max_dd", "ann_vol"]:
        if c in _sum_display.columns:
            _sum_display[c] = _sum_display[c].apply(lambda v: f"{v:.2%}")
    if "final_equity" in _sum_display.columns:
        _sum_display["final_equity"] = _sum_display["final_equity"].apply(lambda v: f"${v:,.0f}")
    _sum_display.columns = [
        {"sharpe": "Sharpe", "calmar": "Calmar", "max_dd": "Max DD",
         "sortino": "Sortino", "ann_vol": "Ann. Vol", "final_equity": "Final Equity",
         "Percentile": "Percentile"}.get(c, c)
        for c in _sum_display.columns
    ]
    st.dataframe(_sum_display, width="stretch", hide_index=True)

st.caption("QuantPipe - for research and paper trading only. Not investment advice.")


# ???????????????????????????????????????????????????????????????????????????????
# TAB 5 - KALMAN FILTER  (dynamic beta / TVP regression)
# ???????????????????????????????????????????????????????????????????????????????

with tab_kalman:
    st.write("MINIMAL TEST - tab mechanism works")
    st.success("If you see this, the tab is working.")
