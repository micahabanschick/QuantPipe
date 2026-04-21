"""Research dashboard — Streamlit page (Dashboard #4).

Tabs:
  Signal Scanner  — current factor rankings and universe snapshot
  Factor Analysis — factor time-series, distribution, and information coefficient
  Walk-Forward    — OOS validation with per-fold Sharpe breakdown
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
import streamlit as st

from config.settings import DATA_DIR
from features.compute import load_features
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, apply_subplot_theme, range_selector,
    kpi_card, section_label, page_header,
)
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date

st.markdown(CSS, unsafe_allow_html=True)

FEATURE_LABELS = {
    "momentum_12m_1m":  "12-1 Momentum",
    "realized_vol_21d": "Realized Vol (21d)",
    "log_return_1d":    "1d Log Return",
    "dollar_volume_63d":"Dollar Volume (63d)",
    "reversal_5d":      "5d Reversal",
}
ALL_FEATURES = list(FEATURE_LABELS.keys())

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_features_cached(
    symbols: tuple[str, ...],
    start_str: str,
    end_str: str,
) -> pl.DataFrame | None:
    try:
        return load_features(
            list(symbols),
            date.fromisoformat(start_str),
            date.fromisoformat(end_str),
            "equity",
        )
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_prices_cached(
    symbols: tuple[str, ...],
    start_str: str,
    end_str: str,
) -> pl.DataFrame | None:
    try:
        prices = load_bars(list(symbols), date.fromisoformat(start_str), date.fromisoformat(end_str), "equity")
        return None if prices.is_empty() else prices
    except Exception:
        return None


@st.cache_data(ttl=600)
def _run_walk_forward(
    symbols: tuple[str, ...],
    start_str: str,
    end_str: str,
    train_years: int,
    test_months: int,
    top_n: int,
    cost_bps: float,
):
    from backtest.walk_forward import walk_forward
    from signals.momentum import cross_sectional_momentum, momentum_weights

    prices = _load_prices_cached(symbols, start_str, end_str)
    features = _load_features_cached(symbols, start_str, end_str)
    if prices is None or features is None:
        return None
    try:
        return walk_forward(
            prices, features,
            signal_fn=cross_sectional_momentum,
            weight_fn=momentum_weights,
            train_years=train_years,
            test_months=test_months,
            cost_bps=cost_bps,
            top_n=top_n,
        )
    except Exception as e:
        return str(e)


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

# ── Load universe & features ──────────────────────────────────────────────────

_end   = date.today()
_start = date(_end.year - lookback_years, _end.month, _end.day)

_symbols_list = universe_as_of_date("equity", _end, require_data=True)
_symbols       = tuple(sorted(_symbols_list)) if _symbols_list else ()

features_df: pl.DataFrame | None = None
if _symbols:
    with st.spinner("Loading features…"):
        features_df = _load_features_cached(_symbols, str(_start), str(_end))

tab_scanner, tab_factor, tab_wfv = st.tabs(
    ["  Signal Scanner  ", "  Factor Analysis  ", "  Walk-Forward  "]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scanner:
    if features_df is None or features_df.is_empty():
        st.warning("No features available. Run: `uv run python features/compute.py`")
    else:
        latest_date = features_df["date"].max()
        snap = features_df.filter(pl.col("date") == latest_date)

        present_features = [f for f in ALL_FEATURES if f in snap.columns]
        snap_pd = snap.select(["symbol"] + present_features).to_pandas().set_index("symbol")

        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown(
            page_header("", f"Universe snapshot · {latest_date}", ""),
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        n_valid_mom = int(snap_pd["momentum_12m_1m"].notna().sum()) if "momentum_12m_1m" in snap_pd.columns else 0
        top5 = (
            snap_pd["momentum_12m_1m"].dropna().nlargest(5).index.tolist()
            if "momentum_12m_1m" in snap_pd.columns else []
        )
        c1.markdown(kpi_card("Universe Size", str(len(snap_pd)), accent=COLORS["teal"]), unsafe_allow_html=True)
        c2.markdown(kpi_card("Symbols w/ Momentum", str(n_valid_mom), accent=COLORS["blue"]), unsafe_allow_html=True)
        c3.markdown(kpi_card("Current Top-1", top5[0] if top5 else "—", accent=COLORS["positive"]), unsafe_allow_html=True)
        c4.markdown(kpi_card("As-of Date", str(latest_date), accent=COLORS["neutral"]), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

        col_rank, col_heat = st.columns([1, 1])

        # ── Momentum ranking bar ──────────────────────────────────────────────
        with col_rank:
            st.markdown(section_label("12-1 Momentum Ranking"), unsafe_allow_html=True)
            if "momentum_12m_1m" in snap_pd.columns:
                mom = snap_pd["momentum_12m_1m"].dropna().sort_values()
                colors_bar = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in mom.values]
                fig_rank = go.Figure(go.Bar(
                    x=mom.values,
                    y=mom.index.tolist(),
                    orientation="h",
                    marker=dict(color=colors_bar, line=dict(width=0)),
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
            z_cols = [f for f in present_features if snap_pd[f].notna().any()]
            if z_cols:
                z_df = snap_pd[z_cols].copy()
                for col in z_cols:
                    mu, sigma = z_df[col].mean(), z_df[col].std()
                    if sigma > 1e-10:
                        z_df[col] = (z_df[col] - mu) / sigma

                z_sorted = z_df.sort_values("momentum_12m_1m", ascending=False) if "momentum_12m_1m" in z_df.columns else z_df

                col_labels = [FEATURE_LABELS.get(c, c) for c in z_cols]
                text_ann = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in z_sorted.values]

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

        # ── Current top-5 table ───────────────────────────────────────────────
        st.markdown(section_label("Full Factor Snapshot"), unsafe_allow_html=True)
        if present_features:
            display_df = snap_pd[present_features].copy()
            display_df.index.name = "Symbol"
            fmt = {}
            for f in present_features:
                if f == "dollar_volume_63d":
                    fmt[f] = "${:,.0f}"
                else:
                    fmt[f] = "{:.3f}"
            display_df.columns = [FEATURE_LABELS.get(c, c) for c in display_df.columns]

            if "12-1 Momentum" in display_df.columns:
                display_df = display_df.sort_values("12-1 Momentum", ascending=False)

            st.dataframe(
                display_df.style.background_gradient(
                    cmap="RdYlGn", axis=0, subset=[FEATURE_LABELS.get(f, f)
                                                    for f in present_features
                                                    if f != "dollar_volume_63d"
                                                    and FEATURE_LABELS.get(f, f) in display_df.columns]
                ).format({FEATURE_LABELS.get(f, f): (
                    "${:,.0f}" if f == "dollar_volume_63d" else "{:.3f}"
                ) for f in present_features}),
                width="stretch",
                height=min(600, 38 * (len(display_df) + 1)),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_factor:
    if features_df is None or features_df.is_empty():
        st.warning("No features available.")
    else:
        present_factors = [f for f in ALL_FEATURES if f in features_df.columns]

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
        with col_ctrl1:
            selected_factor = st.selectbox(
                "Factor",
                present_factors,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
            )
        with col_ctrl2:
            all_syms = sorted(features_df["symbol"].unique().to_list())
            selected_syms = st.multiselect(
                "Symbols",
                all_syms,
                default=all_syms[:8],
                max_selections=15,
            )
        with col_ctrl3:
            ic_window = st.selectbox("IC window (days)", [21, 63, 126], index=0,
                                     format_func=lambda x: f"{x}d fwd")

        if not selected_syms:
            st.info("Select at least one symbol.")
        else:
            fac_df = (
                features_df
                .filter(pl.col("symbol").is_in(selected_syms))
                .select(["date", "symbol", selected_factor])
                .to_pandas()
            )
            fac_df["date"] = pd.to_datetime(fac_df["date"])
            fac_pivot = fac_df.pivot(index="date", columns="symbol", values=selected_factor)

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
                yaxis=dict(
                    tickformat=".1%" if selected_factor in ("momentum_12m_1m", "reversal_5d", "realized_vol_21d", "log_return_1d") else ",.0f"
                ),
            )
            st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CONFIG)

            col_dist, col_ic = st.columns([1, 1])

            # ── Distribution ──────────────────────────────────────────────────
            with col_dist:
                st.markdown(section_label("Factor Distribution"), unsafe_allow_html=True)
                all_vals = fac_df[selected_factor].dropna().values
                from scipy.stats import norm as _norm
                x_grid = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 200)
                pdf_fit = _norm.pdf(x_grid, all_vals.mean(), all_vals.std())

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=all_vals,
                    histnorm="probability density",
                    nbinsx=60,
                    marker=dict(color=COLORS["blue"], line=dict(width=0)),
                    opacity=0.65,
                    name="Observed",
                ))
                fig_dist.add_trace(go.Scatter(
                    x=x_grid, y=pdf_fit,
                    line=dict(color=COLORS["warning"], width=2),
                    name="Normal fit",
                ))
                apply_theme(fig_dist, legend_inside=True)
                fig_dist.update_layout(
                    height=280,
                    xaxis=dict(
                        tickformat=".1%" if selected_factor in ("momentum_12m_1m", "reversal_5d", "realized_vol_21d", "log_return_1d") else ",.0f"
                    ),
                    yaxis=dict(title="Density"),
                    bargap=0.03,
                    showlegend=True,
                )
                st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)

                # Stats
                skew = float(pd.Series(all_vals).skew())
                kurt = float(pd.Series(all_vals).kurtosis())
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Mean",  f"{all_vals.mean():.4f}")
                c_b.metric("Std",   f"{all_vals.std():.4f}")
                c_c.metric("Skew",  f"{skew:.2f}")
                c_d.metric("Kurt",  f"{kurt:.2f}")

            # ── Information Coefficient ───────────────────────────────────────
            with col_ic:
                st.markdown(section_label(f"Information Coefficient ({ic_window}d forward)"), unsafe_allow_html=True)

                prices_df = _load_prices_cached(_symbols, str(_start), str(_end))
                ic_dates, ic_vals = [], []

                if prices_df is not None:
                    try:
                        prices_pd = (
                            prices_df.select(["date", "symbol", "adj_close"])
                            .to_pandas()
                        )
                        prices_pd["date"] = pd.to_datetime(prices_pd["date"])
                        price_pivot = prices_pd.pivot(index="date", columns="symbol", values="adj_close").sort_index()

                        fwd_ret = price_pivot.pct_change(ic_window).shift(-ic_window)
                        all_dates = sorted(fac_pivot.index)
                        step = max(1, ic_window // 3)

                        for dt in all_dates[::step]:
                            if dt not in fac_pivot.index or dt not in fwd_ret.index:
                                continue
                            factor_row = fac_pivot.loc[dt].dropna()
                            fwd_row    = fwd_ret.loc[dt].reindex(factor_row.index).dropna()
                            shared = factor_row.index.intersection(fwd_row.index)
                            if len(shared) < 5:
                                continue
                            rho, _ = spearmanr(factor_row[shared], fwd_row[shared])
                            if not np.isnan(rho):
                                ic_dates.append(dt)
                                ic_vals.append(rho)
                    except Exception:
                        pass

                if ic_vals:
                    ic_series = pd.Series(ic_vals, index=ic_dates)
                    ic_roll   = ic_series.rolling(6, min_periods=3).mean()
                    mean_ic   = float(ic_series.mean())
                    icir      = mean_ic / ic_series.std() if ic_series.std() > 1e-10 else 0.0

                    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in ic_vals]
                    fig_ic = go.Figure()
                    fig_ic.add_trace(go.Bar(
                        x=ic_dates, y=ic_vals,
                        marker=dict(color=bar_colors, line=dict(width=0)),
                        opacity=0.55,
                        name="IC",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>IC</extra>",
                    ))
                    fig_ic.add_trace(go.Scatter(
                        x=list(ic_roll.index), y=ic_roll.values,
                        line=dict(color=COLORS["teal"], width=2),
                        name="6-period MA",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>MA</extra>",
                    ))
                    fig_ic.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                    apply_theme(fig_ic, legend_inside=True)
                    fig_ic.update_layout(
                        height=280,
                        yaxis=dict(tickformat=".2f", title="Rank IC"),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_ic, width="stretch", config=PLOTLY_CONFIG)

                    c_a, c_b = st.columns(2)
                    c_a.metric("Mean IC",  f"{mean_ic:.4f}")
                    c_b.metric("IC IR",    f"{icir:.3f}")
                else:
                    st.info("Could not compute IC — price data unavailable.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_wfv:
    st.markdown(section_label("Walk-Forward Validation Configuration"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wfv_train  = st.slider("Training window (years)", 2, 5, 3)
    with c2:
        wfv_test   = st.slider("Test window (months)", 3, 24, 12)
    with c3:
        wfv_top_n  = st.slider("Top-N positions", 3, 10, 5)
    with c4:
        wfv_cost   = st.number_input("Cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

    required_years = wfv_train + wfv_test / 12
    if lookback_years < required_years:
        st.warning(f"Increase Feature lookback to at least {required_years:.1f} years for {len([None])} fold(s).")

    run_wfv = st.button("▶ Run Walk-Forward Validation", type="primary", use_container_width=False)

    wfv_result_key = f"wfv_{_symbols}_{wfv_train}_{wfv_test}_{wfv_top_n}_{wfv_cost}_{lookback_years}"

    if run_wfv:
        with st.spinner("Running walk-forward validation — this may take 30–90 seconds…"):
            result = _run_walk_forward(
                _symbols, str(_start), str(_end),
                wfv_train, wfv_test, wfv_top_n, float(wfv_cost),
            )
        st.session_state[wfv_result_key] = result

    wfv_result = st.session_state.get(wfv_result_key)

    if wfv_result is None:
        st.info("Configure parameters above and click **▶ Run Walk-Forward Validation** to start.")
    elif isinstance(wfv_result, str):
        st.error(f"Walk-forward failed: {wfv_result}")
    else:
        folds = wfv_result.folds
        if not folds:
            st.warning("No complete folds — extend lookback or reduce training/test windows.")
        else:
            # ── Summary KPIs ──────────────────────────────────────────────────
            st.markdown("<div style='height:12px'/>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(kpi_card("OOS Sharpe",    f"{wfv_result.combined_sharpe:.3f}",    accent=COLORS["teal"]),   unsafe_allow_html=True)
            c2.markdown(kpi_card("OOS CAGR",      f"{wfv_result.combined_cagr:.1%}",      accent=COLORS["blue"]),   unsafe_allow_html=True)
            c3.markdown(kpi_card("OOS Max DD",     f"{wfv_result.combined_max_drawdown:.1%}", accent=COLORS["negative"]), unsafe_allow_html=True)
            c4.markdown(kpi_card("Folds",          str(len(folds)),                        accent=COLORS["neutral"]), unsafe_allow_html=True)

            st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

            # ── OOS combined equity curve ──────────────────────────────────────
            st.markdown(section_label("OOS Combined Equity Curve"), unsafe_allow_html=True)
            eq = wfv_result.combined_equity
            eq_norm = 10_000 * eq / eq.iloc[0]
            fig_eq = go.Figure(go.Scatter(
                x=eq_norm.index, y=eq_norm.values,
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.07)",
                line=dict(color=COLORS["positive"], width=2),
                hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra>OOS</extra>",
            ))

            # Shade each fold a different background
            palette = [COLORS["card_bg"], COLORS["bg"]]
            for i, fold in enumerate(folds):
                fig_eq.add_vrect(
                    x0=str(fold.test_start), x1=str(fold.test_end),
                    fillcolor=palette[i % 2], opacity=0.35, layer="below",
                    line_width=0,
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

            # ── Per-fold breakdown ─────────────────────────────────────────────
            st.markdown(section_label("Per-Fold Performance"), unsafe_allow_html=True)

            from backtest.tearsheet import tearsheet_dict
            fold_rows = []
            for fold in folds:
                td = tearsheet_dict(fold.result)
                fold_rows.append({
                    "Fold":       fold.fold + 1,
                    "Test Start": str(fold.test_start),
                    "Test End":   str(fold.test_end),
                    "OOS Sharpe": round(td.get("sharpe", 0), 3),
                    "OOS CAGR":   f"{td.get('cagr', 0):.1%}",
                    "OOS Max DD": f"{td.get('max_drawdown', 0):.1%}",
                    "OOS Vol":    f"{td.get('vol', 0):.1%}",
                    "Calmar":     round(td.get('calmar', 0), 3),
                })

            fold_df = pd.DataFrame(fold_rows)
            st.dataframe(fold_df, width="stretch", hide_index=True)

            # ── Sharpe per fold bar ────────────────────────────────────────────
            col_sharpe_bar, col_cagr_bar = st.columns([1, 1])

            sharpe_vals = [r["OOS Sharpe"] for r in fold_rows]
            cagr_vals   = [float(r["OOS CAGR"].rstrip("%")) / 100 for r in fold_rows]
            fold_labels = [f"Fold {r['Fold']}\n{r['Test Start'][:7]}" for r in fold_rows]

            with col_sharpe_bar:
                fig_sb = go.Figure(go.Bar(
                    x=fold_labels, y=sharpe_vals,
                    marker=dict(
                        color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in sharpe_vals],
                        line=dict(width=0),
                    ),
                    text=[f"{v:.2f}" for v in sharpe_vals],
                    textposition="outside",
                    textfont=dict(size=11, color=COLORS["text"]),
                    hovertemplate="<b>%{x}</b>: %{y:.3f}<extra>Sharpe</extra>",
                ))
                fig_sb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_sb)
                fig_sb.update_layout(
                    height=240, title="OOS Sharpe by Fold",
                    yaxis=dict(showgrid=False),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_sb, width="stretch", config=PLOTLY_CONFIG)

            with col_cagr_bar:
                fig_cb = go.Figure(go.Bar(
                    x=fold_labels, y=cagr_vals,
                    marker=dict(
                        color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in cagr_vals],
                        line=dict(width=0),
                    ),
                    text=[f"{v:.1%}" for v in cagr_vals],
                    textposition="outside",
                    textfont=dict(size=11, color=COLORS["text"]),
                    hovertemplate="<b>%{x}</b>: %{y:.1%}<extra>CAGR</extra>",
                ))
                fig_cb.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
                apply_theme(fig_cb)
                fig_cb.update_layout(
                    height=240, title="OOS CAGR by Fold",
                    yaxis=dict(tickformat=".0%", showgrid=False),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_cb, width="stretch", config=PLOTLY_CONFIG)

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
