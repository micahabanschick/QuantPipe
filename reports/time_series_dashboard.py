"""Time Series Analysis — standalone page.

Sections:
  1. Power Spectral Density (Welch) + dominant cycles
  2. FFT Frequency Filter Decomposition (trend vs cycle)
  3. Haar Wavelet Decomposition
  4. Geometric Brownian Motion Simulation
  5. Autocorrelation (returns + squared returns / ARCH)
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, kpi_card, section_label, page_header,
)
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    page_header(
        "Time Series Analysis",
        "Spectral decomposition, frequency filtering, wavelet analysis, GBM simulation, and autocorrelation.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

# -- Data loader ---------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_prices(sym: str, start_str: str, end_str: str) -> pd.Series | None:
    try:
        bars = load_bars([sym], date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        if bars.is_empty():
            return None
        price_col = "adj_close" if "adj_close" in bars.columns else "close"
        s = bars.sort("date").to_pandas().set_index("date")[price_col]
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception:
        return None

# -- Universe + controls -------------------------------------------------------

_end      = date.today()
_sym_list = universe_as_of_date("equity", _end, require_data=True)

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    lookback_years = st.select_slider(
        "Lookback (years)", options=[1, 2, 3, 4, 5, 6, 7], value=6,
        format_func=lambda x: f"{x} yr", key="ts_lookback",
    )
with c2:
    ts_sym = st.selectbox(
        "Asset",
        sorted(_sym_list) if _sym_list else ["SPY"],
        index=(sorted(_sym_list).index("SPY") if _sym_list and "SPY" in _sym_list else 0),
        key="ts_sym",
    )
with c3:
    ts_cutoff = st.select_slider(
        "Trend cutoff (days)", options=[5, 10, 20, 40, 63], value=20,
        key="ts_cutoff",
        help="FFT low-pass cutoff: cycles longer than this are classified as trend.",
    )

_start = date(_end.year - lookback_years, _end.month, _end.day)

with st.spinner("Loading prices..."):
    ts_s = _load_prices(ts_sym, str(_start), str(_end))

if ts_s is None or len(ts_s) < 30:
    st.warning("Not enough price data. Try a different asset or extend the lookback.")
    st.stop()

# Pre-compute shared series
ts_log    = np.log(ts_s.values)
ts_log_dm = ts_log - ts_log.mean()
ts_rets   = ts_s.pct_change().dropna().values
ts_dates  = ts_s.index

from research.spectral import (
    compute_psd, dominant_cycles, fft_filter,
    haar_wavelet_1d, estimate_gbm_params, gbm_paths,
    acf as _spec_acf,
)

# ==============================================================================
# Section 1 — Power Spectral Density
# ==============================================================================

st.markdown(section_label("Power Spectral Density"), unsafe_allow_html=True)
st.caption(
    "Welch PSD of demeaned log-prices. Peaks reveal dominant cycles. "
    "x-axis shows cycle period in trading days (log scale)."
)

freqs, power = compute_psd(ts_log_dm)
cycles = dominant_cycles(freqs, power, top_n=5)

if len(freqs) > 0:
    period_axis = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (period_axis >= 2) & (period_axis <= len(ts_s))
    fig_psd = go.Figure(go.Scatter(
        x=period_axis[mask], y=power[mask],
        fill="tozeroy", fillcolor="rgba(74,144,217,0.12)",
        line=dict(color=COLORS["blue"], width=2),
        hovertemplate="Period: %{x:.0f}d<br>Power: %{y:.4f}<extra></extra>",
    ))
    for cyc in cycles:
        fig_psd.add_vline(
            x=cyc["period_days"],
            line=dict(color=COLORS["gold"], width=1.5, dash="dot"),
            annotation_text=f"{cyc['period_days']:.0f}d ({cyc['label']})",
            annotation_font=dict(size=9, color=COLORS["gold"]),
            annotation_position="top left",
        )
    apply_theme(fig_psd)
    fig_psd.update_layout(
        height=280,
        xaxis=dict(title="Cycle period (trading days)", type="log", showgrid=False),
        yaxis=dict(title="Power", showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig_psd, use_container_width=True, config=PLOTLY_CONFIG)

    if cycles:
        cyc_cols = st.columns(len(cycles))
        for ci, (cyc, col) in enumerate(zip(cycles, cyc_cols)):
            col.markdown(
                kpi_card(
                    cyc["label"],
                    f"{cyc['period_days']:.0f}d",
                    accent=COLORS["series"][ci % len(COLORS["series"])],
                    delta=f"{cyc['rel_power_pct']:.1f}% of power",
                ),
                unsafe_allow_html=True,
            )

# ==============================================================================
# Section 2 — FFT Frequency Filter
# ==============================================================================

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
st.markdown(section_label("Frequency Filter Decomposition"), unsafe_allow_html=True)
st.caption(
    f"Brick-wall FFT filter — cutoff = {ts_cutoff}d. "
    "Trend = low-frequency component. Cycle = residual (faster than the cutoff)."
)

cutoff_frac = 1.0 / ts_cutoff
trend = fft_filter(ts_log_dm, cutoff_frac, mode="low")
cycle = ts_log_dm - trend

fft_c1, fft_c2 = st.columns(2)
with fft_c1:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=ts_dates, y=ts_log_dm,
        line=dict(color=COLORS["neutral"], width=1, dash="dot"), opacity=0.5,
        name="Original (log, demeaned)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=ts_dates, y=trend,
        line=dict(color=COLORS["positive"], width=2),
        name=f"Trend (>{ts_cutoff}d)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>Trend</extra>",
    ))
    apply_theme(fig_trend, legend_inside=True)
    fig_trend.update_layout(
        height=260, title="Trend vs Original",
        yaxis=dict(showgrid=False), xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)

with fft_c2:
    cyc_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in cycle]
    fig_cycle = go.Figure(go.Bar(
        x=ts_dates, y=cycle,
        marker=dict(color=cyc_colors, line=dict(width=0)), opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>Cycle</extra>",
    ))
    fig_cycle.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    apply_theme(fig_cycle)
    fig_cycle.update_layout(
        height=260, title=f"Cyclical Component (<{ts_cutoff}d)",
        yaxis=dict(showgrid=False), xaxis=dict(showgrid=False), showlegend=False,
    )
    st.plotly_chart(fig_cycle, use_container_width=True, config=PLOTLY_CONFIG)

# ==============================================================================
# Section 3 — Haar Wavelet Decomposition
# ==============================================================================

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
st.markdown(section_label("Haar Wavelet Decomposition"), unsafe_allow_html=True)
st.caption(
    "3-level Haar decomposition of daily returns. "
    "Approx = low-frequency trend. Detail L1 = finest scale (1-2d). "
    "Percentage shows fraction of total signal energy at each scale."
)

wav_levels  = haar_wavelet_1d(ts_rets, levels=3)
wav_labels  = ["Approximation (trend)", "Detail L1 (fine)", "Detail L2 (mid)", "Detail L3 (coarse)"]
wav_colors  = [COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["gold"]]
wav_energies = [float(np.sum(lv**2)) for lv in wav_levels]
wav_total   = max(sum(wav_energies), 1e-12)

wav_cols = st.columns(min(len(wav_levels), 4))
for wi, (wlv, wlbl, wcl) in enumerate(zip(wav_levels, wav_labels, wav_colors)):
    pct = wav_energies[wi] / wav_total * 100
    fig_w = go.Figure(go.Bar(
        y=wlv, x=list(range(len(wlv))),
        marker=dict(color=[wcl if v >= 0 else COLORS["negative"] for v in wlv], line=dict(width=0)),
        opacity=0.75,
        hovertemplate="Coeff %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_w.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    apply_theme(fig_w, title=f"{wlbl} — {pct:.1f}% energy")
    fig_w.update_layout(
        height=200, showlegend=False,
        yaxis=dict(showgrid=False), xaxis=dict(showgrid=False, title="Coefficient"),
    )
    wav_cols[wi].plotly_chart(fig_w, use_container_width=True, config=PLOTLY_CONFIG)

# ==============================================================================
# Section 4 — Geometric Brownian Motion Simulation
# ==============================================================================

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
st.markdown(section_label("Geometric Brownian Motion Simulation"), unsafe_allow_html=True)
st.caption(
    "500 GBM paths calibrated to historical drift (mu) and volatility (sigma). "
    "Actual price shown in gold. Persistent deviation outside the fan indicates "
    "the GBM null hypothesis is violated (trending or mean-reverting behaviour)."
)

mu_ann, sig_ann = estimate_gbm_params(ts_s.values)
T      = len(ts_s) - 1
S0     = float(ts_s.iloc[0])
paths  = gbm_paths(S0, mu_ann, sig_ann, T, n_paths=500, seed=42)
pcts   = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
t_ax   = list(range(T + 1))

g1, g2, g3, g4 = st.columns(4)
g1.markdown(kpi_card("Est. Drift (ann.)", f"{mu_ann:.1%}",
                      accent=COLORS["positive"] if mu_ann >= 0 else COLORS["negative"]),
            unsafe_allow_html=True)
g2.markdown(kpi_card("Est. Vol (ann.)", f"{sig_ann:.1%}", accent=COLORS["warning"]),
            unsafe_allow_html=True)
g3.markdown(kpi_card("P95 Terminal", f"${pcts[4, -1]:,.0f}", accent=COLORS["blue"]),
            unsafe_allow_html=True)
g4.markdown(kpi_card("P5 Terminal",  f"${pcts[0, -1]:,.0f}", accent=COLORS["neutral"]),
            unsafe_allow_html=True)

_terminal = paths[:, -1]
_lo, _hi  = np.percentile(_terminal, [5, 95])
fig_gbm = go.Figure()
_px, _py = [], []
for _path in paths:
    if not (_lo <= _path[-1] <= _hi):
        continue
    _px.extend(t_ax); _px.append(None)
    _py.extend(_path); _py.append(None)
fig_gbm.add_trace(go.Scatter(
    x=_px, y=_py, mode="lines",
    line=dict(color="rgba(160,160,160,0.15)", width=0.5),
    showlegend=False, hoverinfo="skip",
))
fig_gbm.add_trace(go.Scatter(
    x=t_ax + t_ax[::-1], y=list(pcts[4]) + list(pcts[0][::-1]),
    fill="toself", fillcolor="rgba(65,130,200,0.10)", line=dict(width=0),
    name="5th–95th", hoverinfo="skip",
))
fig_gbm.add_trace(go.Scatter(
    x=t_ax + t_ax[::-1], y=list(pcts[3]) + list(pcts[1][::-1]),
    fill="toself", fillcolor="rgba(65,130,200,0.22)", line=dict(width=0),
    name="25th–75th", hoverinfo="skip",
))
fig_gbm.add_trace(go.Scatter(
    x=t_ax, y=pcts[2],
    line=dict(color=COLORS["blue"], width=2), name="Median GBM",
    hovertemplate="Day %{x}: $%{y:,.0f}<extra>Median</extra>",
))
fig_gbm.add_trace(go.Scatter(
    x=t_ax, y=ts_s.values,
    line=dict(color=COLORS["gold"], width=2.5, dash="dash"), name="Actual",
    hovertemplate="Day %{x}: $%{y:,.0f}<extra>Actual</extra>",
))
apply_theme(fig_gbm, legend_inside=False)
fig_gbm.update_layout(
    height=360,
    xaxis=dict(title="Trading day", showgrid=False),
    yaxis=dict(title="Price ($)", tickprefix="$", tickformat=",.0f"),
)
st.plotly_chart(fig_gbm, use_container_width=True, config=PLOTLY_CONFIG)

# ==============================================================================
# Section 5 — Autocorrelation
# ==============================================================================

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
st.markdown(section_label("Autocorrelation (Returns & Volatility Clustering)"), unsafe_allow_html=True)
st.caption(
    "ACF of returns tests for serial correlation (momentum / mean-reversion). "
    "ACF of squared returns tests for volatility clustering (ARCH effects). "
    "Bars outside dashed lines exceed the 95% CI (+/- 1.96 / sqrt(n))."
)

acf_lags = list(range(1, 21))
acf_r    = _spec_acf(ts_rets,      max_lag=20)
acf_sq   = _spec_acf(ts_rets ** 2, max_lag=20)
acf_ci   = 1.96 / np.sqrt(len(ts_rets))

acf_c1, acf_c2 = st.columns(2)
for col, vals, title, sig_col in [
    (acf_c1, acf_r,  "ACF of Returns",               COLORS["positive"]),
    (acf_c2, acf_sq, "ACF of Squared Returns (ARCH)", COLORS["warning"]),
]:
    with col:
        bar_cols = [sig_col if abs(v) > acf_ci else COLORS["neutral"] for v in vals]
        fig_acf  = go.Figure(go.Bar(
            x=acf_lags, y=list(vals),
            marker=dict(color=bar_cols, line=dict(width=0)), opacity=0.8,
            hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>",
        ))
        fig_acf.add_hline(y=acf_ci,  line=dict(color=COLORS["border"], width=1, dash="dot"))
        fig_acf.add_hline(y=-acf_ci, line=dict(color=COLORS["border"], width=1, dash="dot"))
        fig_acf.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
        apply_theme(fig_acf, title=title)
        fig_acf.update_layout(
            height=260, showlegend=False,
            xaxis=dict(title="Lag (days)", showgrid=False, tickmode="linear"),
            yaxis=dict(title="Autocorrelation"),
        )
        st.plotly_chart(fig_acf, use_container_width=True, config=PLOTLY_CONFIG)

# ==============================================================================
# Section 6 — Rolling Hurst Exponent
# ==============================================================================

st.markdown("<div style='height:16px'/>", unsafe_allow_html=True)
st.markdown(section_label("Rolling Hurst Exponent (Long Memory)"), unsafe_allow_html=True)
st.caption(
    "H > 0.55 = persistent / trending.  H ~ 0.50 = random walk.  H < 0.45 = mean-reverting. "
    "Computed on log-prices via R/S analysis in a rolling window."
)

from research.long_memory import rolling_hurst, hurst_rs, hurst_label as _hurst_label

hurst_window = st.select_slider(
    "Rolling window (days)", options=[63, 126, 252], value=126, key="ts_hurst_win",
)

with st.spinner("Computing Hurst exponents…"):
    _h_full = hurst_rs(ts_log)
    _h_lbl  = _hurst_label(_h_full)
    _h_idxs, _h_vals = rolling_hurst(ts_log, window=hurst_window)

_h_dates = [ts_dates[i] for i in _h_idxs] if len(_h_idxs) > 0 else []

hk1, hk2, hk3, hk4 = st.columns(4)
hk1.markdown(kpi_card("Full-Sample H (R/S)", f"{_h_full:.4f}",
                        accent=COLORS["positive"] if _h_full > 0.55
                        else COLORS["negative"] if _h_full < 0.45 else COLORS["neutral"]),
             unsafe_allow_html=True)
hk2.markdown(kpi_card("Regime", _h_lbl,
                        accent=COLORS["positive"] if "Persistent" in _h_lbl
                        else COLORS["negative"] if "Mean" in _h_lbl else COLORS["neutral"]),
             unsafe_allow_html=True)
hk3.markdown(kpi_card("Rolling Window", f"{hurst_window}d", accent=COLORS["blue"]),
             unsafe_allow_html=True)
if _h_vals is not None and len(_h_vals) > 0:
    hk4.markdown(kpi_card("Current H (rolling)", f"{float(_h_vals[-1]):.4f}",
                            accent=COLORS["gold"]), unsafe_allow_html=True)

if _h_dates and len(_h_dates) > 1:
    _h_colors = [
        COLORS["positive"] if v > 0.55 else
        COLORS["negative"] if v < 0.45 else
        COLORS["neutral"]
        for v in _h_vals
    ]
    fig_hurst = go.Figure()
    # Regime bands
    fig_hurst.add_hrect(y0=0.55, y1=1.0,  fillcolor="rgba(0,212,170,0.05)", line_width=0)
    fig_hurst.add_hrect(y0=0.0,  y1=0.45, fillcolor="rgba(255,77,77,0.05)",  line_width=0)
    fig_hurst.add_trace(go.Scatter(
        x=_h_dates, y=_h_vals, mode="lines",
        line=dict(color=COLORS["gold"], width=2),
        name=f"Rolling {hurst_window}d H",
        hovertemplate="%{x|%Y-%m-%d}: H=%{y:.4f}<extra></extra>",
    ))
    fig_hurst.add_hline(y=0.5,  line=dict(color=COLORS["border"],   width=1.5, dash="dot"),
                         annotation_text="H=0.50 random walk",
                         annotation_font=dict(size=9, color=COLORS["text_muted"]),
                         annotation_position="right")
    fig_hurst.add_hline(y=0.55, line=dict(color=COLORS["positive"], width=1,   dash="dot"),
                         annotation_text="0.55 persistent",
                         annotation_font=dict(size=9, color=COLORS["positive"]),
                         annotation_position="right")
    fig_hurst.add_hline(y=0.45, line=dict(color=COLORS["negative"], width=1,   dash="dot"),
                         annotation_text="0.45 mean-reverting",
                         annotation_font=dict(size=9, color=COLORS["negative"]),
                         annotation_position="right")
    apply_theme(fig_hurst, legend_inside=True)
    fig_hurst.update_layout(
        height=280,
        yaxis=dict(range=[0.2, 0.85], title="Hurst exponent H", showgrid=False),
        xaxis=dict(showgrid=False),
        showlegend=True,
    )
    st.plotly_chart(fig_hurst, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(
        "Green band = persistent / trending regime (H > 0.55). "
        "Red band = mean-reverting regime (H < 0.45). "
        "Centre = random walk. Use this to decide whether to apply momentum or mean-reversion signals."
    )
else:
    st.info(f"Need at least {hurst_window} observations to compute rolling Hurst. Try reducing the window.")

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
