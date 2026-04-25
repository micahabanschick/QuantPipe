"""Kalman Filter dashboard - dynamic beta / TVP regression.

Standalone page (not a tab) to avoid Streamlit tab rendering limits.
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, kpi_card, section_label, page_header,
)
from research.kalman_filter import kalman_smooth_betas, KalmanResult
from risk.factor_model import (
    FACTOR_PROXIES, estimate_factor_returns, rolling_factor_betas,
)
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    page_header(
        "Kalman Filter",
        "Time-varying parameter regression - dynamic factor betas updated daily.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_prices(symbols: tuple, start_str: str, end_str: str):
    try:
        df = load_bars(list(symbols), date.fromisoformat(start_str),
                       date.fromisoformat(end_str), "equity")
        return None if df.is_empty() else df
    except Exception:
        return None


@st.cache_data(ttl=300)
def _run_kalman(sym: str, delta: float, ols_window: int,
                start_str: str, end_str: str,
                symbols_tuple: tuple):
    bars = _load_prices(symbols_tuple, start_str, end_str)
    if bars is None:
        return KalmanResult(), pd.DataFrame()
    fr = estimate_factor_returns(bars, FACTOR_PROXIES)
    if fr.returns.empty:
        return KalmanResult(), pd.DataFrame()
    price_col = "adj_close" if "adj_close" in bars.columns else "close"
    port = (
        bars.filter(pl.col("symbol") == sym)
        .sort("date")
        .to_pandas()
        .set_index("date")[price_col]
        .pct_change()
        .dropna()
    )
    port.index = pd.to_datetime(port.index)
    kr  = kalman_smooth_betas(port, fr.returns, delta=delta)
    ols = rolling_factor_betas(port, fr, window=ols_window,
                               min_periods=max(ols_window // 2, 20))
    return kr, ols


# ── Universe ───────────────────────────────────────────────────────────────────

_end      = date.today()
_sym_list = universe_as_of_date("equity", _end, require_data=True)
_symbols  = tuple(sorted(_sym_list)) if _sym_list else ()

# ── Controls ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 2])
with c1:
    lookback_years = st.select_slider(
        "Lookback (years)",
        options=[1, 2, 3, 4, 5, 6, 7], value=6,
        format_func=lambda x: f"{x} yr",
        key="kf_lb",
    )
with c2:
    sym = st.selectbox(
        "Asset",
        options=sorted(_sym_list) if _sym_list else ["SPY"],
        index=(sorted(_sym_list).index("SPY")
               if _sym_list and "SPY" in _sym_list else 0),
        key="kf_sym",
    )
with c3:
    delta = st.select_slider(
        "delta (process noise)",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}",
        key="kf_delta",
        help="delta/(1-delta) = Q/R ratio. Higher = betas adapt faster.",
    )
with c4:
    ols_window = st.select_slider(
        "OLS comparison window",
        options=[63, 126, 252], value=126,
        format_func=lambda x: f"{x}d",
        key="kf_ols",
    )

_start = date(_end.year - lookback_years, _end.month, _end.day)

# ── Run ────────────────────────────────────────────────────────────────────────

with st.spinner("Running Kalman filter..."):
    kr, ols_df = _run_kalman(sym, delta, ols_window,
                             str(_start), str(_end), _symbols)

if not kr.dates or kr.filtered_betas.size == 0:
    st.warning("Not enough price data. Try extending the lookback period.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────

K = kr.filtered_betas.shape[1]
beta_names = ["Alpha"] + kr.factor_names
mkt_idx = 1 if K > 1 else 0
cur_beta = float(kr.filtered_betas[-1, mkt_idx])
cur_std  = float(np.sqrt(kr.filtered_vars[-1, mkt_idx, mkt_idx]))

st.markdown(section_label("Filter Summary"), unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("Log-Likelihood",  f"{kr.log_likelihood:,.1f}",
                      accent=COLORS["blue"]), unsafe_allow_html=True)
k2.markdown(kpi_card("Mean Innovation", f"{float(np.mean(kr.innovations)):+.5f}",
                      delta="~0 if well-calibrated",
                      accent=COLORS["gold"]), unsafe_allow_html=True)
k3.markdown(kpi_card(f"Current {beta_names[mkt_idx]} beta",
                      f"{cur_beta:+.4f}",
                      accent=COLORS["positive"] if cur_beta >= 0 else COLORS["negative"]),
            unsafe_allow_html=True)
k4.markdown(kpi_card("Beta uncertainty (1-sigma)", f"+/-{cur_std:.4f}",
                      accent=COLORS["warning"]), unsafe_allow_html=True)

# ── Rolling beta comparison ────────────────────────────────────────────────────

st.markdown(section_label("Kalman vs OLS Rolling Betas"), unsafe_allow_html=True)
st.caption("Solid = Kalman filtered betas with +/-1-sigma confidence band. "
           "Dashed = OLS rolling betas.")

dates_pd = pd.to_datetime(kr.dates)
fig_beta = go.Figure()
_colors      = [COLORS["gold"], COLORS["green"], COLORS["purple"], COLORS["blue"]]
_fill_colors = ["rgba(201,162,39,0.10)", "rgba(0,230,118,0.10)",
                "rgba(123,94,167,0.10)", "rgba(74,144,217,0.10)"]

for fi in range(1, K):
    fname = beta_names[fi]
    betas = kr.filtered_betas[:, fi]
    stds  = np.sqrt(np.clip(kr.filtered_vars[:, fi, fi], 0, None))
    col   = _colors[(fi - 1) % len(_colors)]

    fig_beta.add_trace(go.Scatter(
        x=list(dates_pd) + list(dates_pd[::-1]),
        y=list(betas + stds) + list((betas - stds)[::-1]),
        fill="toself", fillcolor=_fill_colors[(fi - 1) % len(_fill_colors)],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_beta.add_trace(go.Scatter(
        x=dates_pd, y=betas, mode="lines",
        line=dict(color=col, width=2),
        name=f"{fname} (Kalman)",
        hovertemplate=f"<b>{fname}</b>: %{{y:.4f}}<extra>Kalman</extra>",
    ))
    if not ols_df.empty and fname in ols_df.columns:
        fig_beta.add_trace(go.Scatter(
            x=ols_df.index, y=ols_df[fname], mode="lines",
            line=dict(color=col, width=1.5, dash="dot"),
            name=f"{fname} (OLS {ols_window}d)",
            hovertemplate=f"<b>{fname}</b>: %{{y:.4f}}<extra>OLS</extra>",
        ))

fig_beta.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
apply_theme(fig_beta, legend_inside=True)
fig_beta.update_layout(height=340, yaxis_title="Beta")
st.plotly_chart(fig_beta, use_container_width=True, config=PLOTLY_CONFIG)

# ── Diagnostics ────────────────────────────────────────────────────────────────

st.markdown(section_label("Innovations and Beta Uncertainty"), unsafe_allow_html=True)
d1, d2 = st.columns(2)

with d1:
    fig_i = go.Figure(go.Bar(
        x=dates_pd, y=kr.innovations,
        marker=dict(
            color=[COLORS["positive"] if v >= 0 else COLORS["negative"]
                   for v in kr.innovations],
            line=dict(width=0)),
        opacity=0.6, name="Innovation",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.5f}<extra></extra>",
    ))
    fig_i.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    apply_theme(fig_i, title="One-step-ahead innovations")
    fig_i.update_layout(height=260, showlegend=False)
    st.plotly_chart(fig_i, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Innovations should be near zero with no autocorrelation "
               "if the model is well-specified.")

with d2:
    fig_v = go.Figure()
    for fi in range(1, K):
        stds = np.sqrt(np.clip(kr.filtered_vars[:, fi, fi], 0, None))
        fig_v.add_trace(go.Scatter(
            x=dates_pd, y=stds, mode="lines",
            line=dict(color=_colors[(fi-1) % len(_colors)], width=1.8),
            name=beta_names[fi],
            hovertemplate=f"<b>{beta_names[fi]}</b>: %{{y:.4f}}<extra></extra>",
        ))
    apply_theme(fig_v, title="Posterior std (estimation uncertainty)",
                legend_inside=True)
    fig_v.update_layout(height=260, yaxis_title="Posterior std")
    st.plotly_chart(fig_v, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Higher values = Kalman less confident about current beta. "
               "Spikes often coincide with regime changes.")

# ── Factor loading heatmap ─────────────────────────────────────────────────────

if K > 1:
    st.markdown(section_label("Factor Loading Heatmap"), unsafe_allow_html=True)
    step = max(1, len(dates_pd) // 60)
    idx  = range(0, len(dates_pd), step)
    heat_dates  = [str(dates_pd[i].date()) for i in idx]
    heat_z      = [[float(kr.filtered_betas[i, fi]) for i in idx]
                   for fi in range(1, K)]

    fig_h = go.Figure(go.Heatmap(
        z=heat_z, x=heat_dates, y=beta_names[1:],
        colorscale=[[0.0, COLORS["negative"]],
                    [0.5, COLORS["card_bg"]],
                    [1.0, COLORS["positive"]]],
        zmid=0, showscale=True,
        colorbar=dict(thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> on %{x}: beta=%{z:.4f}<extra></extra>",
    ))
    apply_theme(fig_h, title="Kalman-filtered betas over time")
    fig_h.update_layout(
        height=max(180, 55 * len(beta_names[1:])),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_h, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Colour shows direction and magnitude of each factor's contribution. "
               "Horizontal colour transitions indicate regime shifts.")
