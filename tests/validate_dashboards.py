"""One-off validation script — confirms all dashboard data sources load correctly."""
import sys, traceback
from datetime import date

def main():
    import polars as pl
    from config.settings import DATA_DIR
    from risk.engine import compute_exposures
    from storage.parquet_store import load_bars

    # 1. target_weights
    tw = pl.read_parquet(DATA_DIR / "gold" / "equity" / "target_weights.parquet")
    latest = tw["date"].max()
    positions = tw.filter(pl.col("date") == latest)["symbol"].to_list()
    print(f"target_weights : {len(tw)} rows, latest={latest}, positions={positions}")

    # 2. portfolio_log
    plog = pl.read_parquet(DATA_DIR / "gold" / "equity" / "portfolio_log.parquet")
    snap = plog.tail(1).to_dicts()[0]
    print(f"portfolio_log  : {len(plog)} rows, VaR95={snap['var_1d_95']:.2%}, "
          f"gross={snap['gross_exposure']:.1%}, passed={snap['pre_trade_passed']}")

    # 3. equity curve (what performance dashboard renders)
    from backtest.engine import run_backtest
    from features.compute import load_features
    from signals.momentum import (cross_sectional_momentum,
                                   get_monthly_rebalance_dates, momentum_weights)
    from storage.universe import universe_as_of_date

    end = date.today()
    start = date(end.year - 6, end.month, end.day)
    symbols = universe_as_of_date("equity", end, require_data=True)
    prices = load_bars(symbols, start, end, "equity")
    features = load_features(symbols, start, end, "equity",
                             feature_list=["momentum_12m_1m", "realized_vol_21d"])
    td = sorted(prices["date"].unique().to_list())
    rd = get_monthly_rebalance_dates(start, end, td)
    sig = cross_sectional_momentum(features, rd, top_n=5)
    wts = momentum_weights(sig, weight_scheme="equal")
    result = run_backtest(prices, wts, cost_bps=5.0)
    eq = result.equity_curve
    print(f"equity curve   : {len(eq)} days, final=${eq.iloc[-1]:,.0f}, "
          f"Sharpe={result.sharpe:.3f}")

    # 4. health dashboard storage reads
    from storage.parquet_store import list_symbols
    eq_syms = list_symbols("equity")
    cr_syms = list_symbols("crypto")
    print(f"storage symbols: {len(eq_syms)} equity, {len(cr_syms)} crypto")

    print("\nALL DASHBOARD DATA OK")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
