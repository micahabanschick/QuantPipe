"""Performance dashboard — Streamlit app (Dashboard #2).

Shows: daily P&L, equity curve, drawdown, rolling Sharpe, sector exposures,
current portfolio positions, VaR trend, and stress test results.

Reads from:
  - data/gold/equity/portfolio_log.parquet    (daily risk snapshots)
  - data/gold/equity/target_weights.parquet   (latest target weights)
  - data/bronze/equity/                        (prices, for equity curve)

Run with: streamlit run reports/performance_dashboard.py
"""

from datetime import date

import numpy as np
import polars as pl
import plotly.graph_objects as go
import streamlit as st

from config.settings import DATA_DIR
from reports._theme import CSS, COLORS, apply_theme, badge
from risk.engine import compute_exposures
from risk.scenarios import run_all_scenarios
from storage.parquet_store import load_bars

st.set_page_config(page_title="QuantPipe — Performance", layout="wide", page_icon="📈")
st.markdown(CSS, unsafe_allow_html=True)
st.title("QuantPipe — Performance")
st.caption(f"As of {date.today()}")

PORTFOLIO_LOG_PATH = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"
TARGET_WEIGHTS_PATH = DATA_DIR / "gold" / "equity" / "target_weights.parquet"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_portfolio_log() -> pl.DataFrame | None:
    if not PORTFOLIO_LOG_PATH.exists():
        return None
    return pl.read_parquet(PORTFOLIO_LOG_PATH).sort("date")


@st.cache_data(ttl=300)
def _load_target_weights() -> pl.DataFrame | None:
    if not TARGET_WEIGHTS_PATH.exists():
        return None
    return pl.read_parquet(TARGET_WEIGHTS_PATH).sort(["date", "symbol"])


@st.cache_data(ttl=300)
def _load_backtest_equity_curve(lookback_years: int = 6) -> pl.DataFrame | None:
    """Run a lightweight canary backtest and return the equity curve."""
    try:
        from backtest.engine import run_backtest
        from features.compute import load_features
        from signals.momentum import (
            cross_sectional_momentum,
            get_monthly_rebalance_dates,
            momentum_weights,
        )
        from storage.universe import universe_as_of_date

        end = date.today()
        start = date(end.year - lookback_years, end.month, end.day)
        symbols = universe_as_of_date("equity", end, require_data=True)
        if not symbols:
            return None

        prices = load_bars(symbols, start, end, "equity")
        if prices.is_empty():
            return None

        features = load_features(symbols, start, end, "equity",
                                 feature_list=["momentum_12m_1m", "realized_vol_21d"])
        if features.is_empty():
            return None

        trading_dates = sorted(prices["date"].unique().to_list())
        rebal_dates = get_monthly_rebalance_dates(start, end, trading_dates)
        signal = cross_sectional_momentum(features, rebal_dates, top_n=5)
        weights = momentum_weights(signal, weight_scheme="equal")

        result = run_backtest(prices, weights, cost_bps=5.0)
        equity = result.equity_curve
        df = pl.from_pandas(equity.reset_index())
        df.columns = ["date", "portfolio_value"]
        return df.with_columns(pl.col("date").cast(pl.Date))
    except Exception:
        return None


def _rolling_sharpe(returns: np.ndarray, window: int = 63) -> np.ndarray:
    if len(returns) < window:
        return np.full(len(returns), np.nan)
    out = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        chunk = returns[i - window + 1: i + 1]
        std = chunk.std()
        out[i] = (chunk.mean() / std * np.sqrt(252)) if std > 1e-10 else np.nan
    return out


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    lookback_years = st.slider("Backtest lookback (years)", 1, 7, 6)
    rolling_window = st.selectbox("Rolling Sharpe window", [21, 63, 126, 252], index=1)
    show_backtest = st.checkbox("Show backtest equity curve", value=True)

# ── Load data ─────────────────────────────────────────────────────────────────

portfolio_log = _load_portfolio_log()
target_weights_df = _load_target_weights()

# ── Section 1: Current portfolio ──────────────────────────────────────────────

st.subheader("Current Portfolio")

if target_weights_df is not None and not target_weights_df.is_empty():
    latest_date = target_weights_df["date"].max()
    latest_weights_df = target_weights_df.filter(pl.col("date") == latest_date)
    current_weights = dict(zip(latest_weights_df["symbol"], latest_weights_df["weight"]))
    exposures = compute_exposures(current_weights)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positions", exposures.n_positions)
    col2.metric("Gross Exposure", f"{exposures.gross_exposure:.1%}")
    col3.metric("Top-5 Concentration", f"{exposures.top_5_concentration:.1%}")
    col4.metric("Largest Name",
                f"{exposures.largest_position[0]} ({exposures.largest_position[1]:.1%})")

    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        weights_display = (
            latest_weights_df
            .with_columns(pl.col("weight").map_elements(lambda w: f"{w:.1%}", return_dtype=pl.Utf8))
            .select(["symbol", "weight", "rebalance_date"])
        )
        st.dataframe(weights_display.to_pandas(), use_container_width=True, hide_index=True)

    with col_chart:
        if exposures.sector_exposures:
            sectors = list(exposures.sector_exposures.keys())
            values = list(exposures.sector_exposures.values())
            fig = go.Figure(go.Bar(
                x=values,
                y=sectors,
                orientation="h",
                marker_color=COLORS["positive"],
                marker_line_width=0,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                textfont=dict(color=COLORS["neutral"], size=11),
            ))
            apply_theme(fig, "Sector Exposures")
            fig.update_layout(height=250, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No target weights found. Run: `uv run python orchestration/generate_signals.py`")
    current_weights = {}

st.divider()

# ── Section 2: Risk snapshot ──────────────────────────────────────────────────

st.subheader("Risk Snapshot")

if portfolio_log is not None and not portfolio_log.is_empty():
    snap = portfolio_log.tail(1).to_dicts()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1-day VaR 95%", f"{snap['var_1d_95']:.2%}")
    col2.metric("1-day VaR 99%", f"{snap['var_1d_99']:.2%}")

    passed = snap["pre_trade_passed"]
    with col3:
        st.metric("Pre-trade", "")
        st.markdown(badge("PASS", "positive") if passed else badge("FAIL", "negative"),
                    unsafe_allow_html=True)

    col4.metric("Worst Stress",
                f"{snap['worst_stress_scenario']} {snap['worst_stress_pnl']:.1%}")

    if len(portfolio_log) > 5:
        var_pd = portfolio_log.select(["date", "var_1d_95", "var_1d_99"]).to_pandas()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=var_pd["date"], y=var_pd["var_1d_95"],
            name="VaR 95%", line=dict(color=COLORS["negative"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=var_pd["date"], y=var_pd["var_1d_99"],
            name="VaR 99%", line=dict(color=COLORS["warning"], width=2, dash="dot"),
        ))
        apply_theme(fig, "VaR Trend")
        fig.update_layout(height=240, yaxis=dict(tickformat=".2%"))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No risk snapshots yet. Run the signal generator to populate.")

# Stress scenarios
if current_weights:
    st.subheader("Stress Scenarios")
    stress = run_all_scenarios(current_weights)
    cols = st.columns(len(stress))
    for col, (scenario, pnl) in zip(cols, stress.items()):
        variant = "negative" if pnl < -0.05 else "warning" if pnl < 0 else "positive"
        col.metric(scenario, f"{pnl:.1%}")

st.divider()

# ── Section 3: Equity curve, drawdown, rolling Sharpe ────────────────────────

st.subheader("Backtest Equity Curve (Canary Strategy)")

equity_df = None
if show_backtest:
    with st.spinner("Running canary backtest..."):
        equity_df = _load_backtest_equity_curve(lookback_years)

    if equity_df is not None and not equity_df.is_empty():
        eq_pd = equity_df.to_pandas().set_index("date")
        eq_values = eq_pd["portfolio_value"].values
        eq_pd["Drawdown"] = _drawdown_series(eq_values)

        daily_ret = eq_pd["portfolio_value"].pct_change().dropna()
        total_return = eq_values[-1] / eq_values[0] - 1
        n_years = len(eq_pd) / 252
        cagr = (eq_values[-1] / eq_values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)
        max_dd = float(eq_pd["Drawdown"].min())

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Return", f"{total_return:.1%}")
        sc2.metric("CAGR", f"{cagr:.1%}")
        sc3.metric("Sharpe", f"{sharpe:.2f}")
        sc4.metric("Max Drawdown", f"{max_dd:.1%}")

        # Equity curve with range selector
        fig_eq = go.Figure(go.Scatter(
            x=eq_pd.index,
            y=eq_pd["portfolio_value"],
            fill="tozeroy",
            fillcolor=f"rgba(0,212,170,0.08)",
            line=dict(color=COLORS["positive"], width=2),
            name="Portfolio Value",
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        apply_theme(fig_eq, "Portfolio Value ($)")
        fig_eq.update_layout(
            height=340,
            yaxis=dict(tickformat="$,.0f"),
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor=COLORS["card_bg"],
                    activecolor=COLORS["positive"],
                    font=dict(color=COLORS["text"], size=11),
                    bordercolor=COLORS["border"],
                ),
            ),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown
        fig_dd = go.Figure(go.Scatter(
            x=eq_pd.index,
            y=eq_pd["Drawdown"],
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.15)",
            line=dict(color=COLORS["negative"], width=1.5),
            name="Drawdown",
            hovertemplate="<b>%{x}</b><br>%{y:.2%}<extra></extra>",
        ))
        apply_theme(fig_dd, "Drawdown")
        fig_dd.update_layout(height=220, yaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Rolling Sharpe
        daily_ret_arr = eq_pd["portfolio_value"].pct_change().fillna(0).values
        rs = _rolling_sharpe(daily_ret_arr, window=rolling_window)
        fig_rs = go.Figure(go.Scatter(
            x=eq_pd.index,
            y=rs,
            line=dict(color=COLORS["blue"], width=1.5),
            name=f"Rolling {rolling_window}d Sharpe",
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
        ))
        fig_rs.add_hline(y=0, line=dict(color=COLORS["neutral"], dash="dot", width=1))
        fig_rs.add_hline(y=1, line=dict(color=COLORS["positive"], dash="dot", width=1),
                         annotation_text="1.0", annotation_font_color=COLORS["positive"])
        apply_theme(fig_rs, f"Rolling {rolling_window}-day Sharpe (annualised)")
        fig_rs.update_layout(height=220)
        st.plotly_chart(fig_rs, use_container_width=True)

    else:
        st.warning("Could not generate equity curve. Ensure price data and features are available.")

st.divider()

# ── Section 4: Monthly returns heatmap ───────────────────────────────────────

if show_backtest and equity_df is not None and not equity_df.is_empty():
    st.subheader("Monthly Returns")
    eq_pd_monthly = equity_df.to_pandas().set_index("date")
    monthly = eq_pd_monthly["portfolio_value"].resample("ME").last().pct_change().dropna()
    monthly_pivot = monthly.to_frame()
    monthly_pivot["Year"] = monthly_pivot.index.year
    monthly_pivot["Month"] = monthly_pivot.index.strftime("%b")
    pivot = monthly_pivot.pivot_table(values="portfolio_value", index="Year", columns="Month")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    z = pivot.values
    text = [[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in z]
    fig_hm = go.Figure(go.Heatmap(
        z=z,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="RdYlGn",
        zmid=0,
        showscale=True,
        colorbar=dict(tickformat=".0%", len=0.8, thickness=14),
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
    ))
    apply_theme(fig_hm, "Monthly Returns")
    fig_hm.update_layout(height=max(200, 40 * len(pivot)), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True)

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
