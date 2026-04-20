"""Performance dashboard — Streamlit app (Dashboard #2).

Shows: daily P&L, equity curve, drawdown, rolling Sharpe, sector exposures,
current portfolio positions, VaR trend, and stress test results.

Reads from:
  - data/gold/equity/portfolio_log.parquet    (daily risk snapshots)
  - data/gold/equity/target_weights.parquet   (latest target weights)
  - data/bronze/equity/                        (prices, for equity curve)

Run with: streamlit run reports/performance_dashboard.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

from config.settings import DATA_DIR
from risk.engine import EQUITY_SECTOR_MAP, compute_exposures
from risk.scenarios import SCENARIOS, run_all_scenarios
from storage.parquet_store import load_bars

st.set_page_config(page_title="QuantPipe — Performance", layout="wide", page_icon="📈")
st.title("QuantPipe — Performance Dashboard")
st.caption(f"As of {date.today()}")

PORTFOLIO_LOG_PATH = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"
TARGET_WEIGHTS_PATH = DATA_DIR / "gold" / "equity" / "target_weights.parquet"


# ── Data loading helpers ──────────────────────────────────────────────────────

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
        equity = result.equity_curve  # pandas Series (VectorBT)
        df = pl.from_pandas(equity.reset_index())
        df.columns = ["date", "portfolio_value"]
        return df.with_columns(pl.col("date").cast(pl.Date))
    except Exception:
        return None


def _rolling_sharpe(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """Annualised rolling Sharpe ratio."""
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


# ── Sidebar controls ──────────────────────────────────────────────────────────

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

    col1, col2, col3, col4 = st.columns(4)
    exposures = compute_exposures(current_weights)

    col1.metric("Positions", exposures.n_positions)
    col2.metric("Gross Exposure", f"{exposures.gross_exposure:.1%}")
    col3.metric("Top-5 Concentration", f"{exposures.top_5_concentration:.1%}")
    col4.metric("Largest Name",
                f"{exposures.largest_position[0]} ({exposures.largest_position[1]:.1%})")

    # Weights table
    weights_display = (
        latest_weights_df
        .with_columns(pl.col("weight").map_elements(lambda w: f"{w:.1%}", return_dtype=pl.Utf8))
        .select(["symbol", "weight", "rebalance_date"])
    )
    st.dataframe(weights_display.to_pandas(), use_container_width=True, hide_index=True)

    # Sector breakdown
    if exposures.sector_exposures:
        sector_df = pl.DataFrame({
            "Sector": list(exposures.sector_exposures.keys()),
            "Exposure": list(exposures.sector_exposures.values()),
        })
        st.bar_chart(sector_df.to_pandas().set_index("Sector")["Exposure"])

else:
    st.info("No target weights found. Run: `uv run python orchestration/generate_signals.py`")
    current_weights = {}

st.divider()

# ── Section 2: Risk metrics ───────────────────────────────────────────────────

st.subheader("Risk Snapshot")

if portfolio_log is not None and not portfolio_log.is_empty():
    latest_snapshot = portfolio_log.tail(1).to_dicts()[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1-day VaR 95%", f"{latest_snapshot['var_1d_95']:.2%}")
    col2.metric("1-day VaR 99%", f"{latest_snapshot['var_1d_99']:.2%}")
    col3.metric("Pre-trade", "PASS" if latest_snapshot["pre_trade_passed"] else "FAIL",
                delta=None,
                delta_color="normal" if latest_snapshot["pre_trade_passed"] else "inverse")
    col4.metric("Worst Stress",
                f"{latest_snapshot['worst_stress_scenario']} "
                f"{latest_snapshot['worst_stress_pnl']:.1%}")

    # VaR trend
    if len(portfolio_log) > 5:
        var_trend = portfolio_log.select(["date", "var_1d_95", "var_1d_99"]).to_pandas().set_index("date")
        st.line_chart(var_trend, color=["#e74c3c", "#c0392b"])
else:
    st.info("No risk snapshots yet. Run the signal generator to populate.")

# Stress scenarios
if current_weights:
    st.subheader("Stress Scenarios (current weights)")
    stress = run_all_scenarios(current_weights)
    stress_df = pl.DataFrame({
        "Scenario": list(stress.keys()),
        "Estimated P&L": [f"{v:.1%}" for v in stress.values()],
    })
    st.dataframe(stress_df.to_pandas(), use_container_width=True, hide_index=True)

st.divider()

# ── Section 3: Equity curve & drawdown ───────────────────────────────────────

st.subheader("Backtest Equity Curve (Canary Strategy)")

if show_backtest:
    with st.spinner("Running canary backtest..."):
        equity_df = _load_backtest_equity_curve(lookback_years)

    if equity_df is not None and not equity_df.is_empty():
        eq_pd = equity_df.to_pandas().set_index("date")
        eq_values = eq_pd["portfolio_value"].values
        eq_pd["Drawdown"] = _drawdown_series(eq_values)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.caption("Portfolio value ($)")
            st.line_chart(eq_pd[["portfolio_value"]])

        with col2:
            # Summary stats
            daily_ret = eq_pd["portfolio_value"].pct_change().dropna()
            total_return = eq_values[-1] / eq_values[0] - 1
            n_years = len(eq_pd) / 252
            cagr = (eq_values[-1] / eq_values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
            sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                      if daily_ret.std() > 0 else 0)
            max_dd = float(eq_pd["Drawdown"].min())

            st.metric("Total Return", f"{total_return:.1%}")
            st.metric("CAGR", f"{cagr:.1%}")
            st.metric("Sharpe", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{max_dd:.1%}")

        st.caption("Drawdown (%)")
        st.area_chart(eq_pd[["Drawdown"]], color=["#e74c3c"])

        # Rolling Sharpe
        st.caption(f"Rolling {rolling_window}-day Sharpe (annualised)")
        daily_ret_arr = eq_pd["portfolio_value"].pct_change().fillna(0).values
        rs = _rolling_sharpe(daily_ret_arr, window=rolling_window)
        rs_df = eq_pd.copy()
        rs_df["Rolling Sharpe"] = rs
        st.line_chart(rs_df[["Rolling Sharpe"]])
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
    st.dataframe(
        pivot.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True,
    )

st.caption("QuantPipe — for research and paper trading only. Not investment advice.")
