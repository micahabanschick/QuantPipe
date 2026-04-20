"""Canary backtest — the pipeline sanity check.

Strategy: cross-sectional 12-1 momentum on the 9-box + sector ETF universe,
monthly rebalance, equal-weight top-5 long only, 5bps round-trip cost.

Expected result: Sharpe 0.5-1.5 over available history.
If Sharpe is outside this range, there is likely a bug in the pipeline.

Usage:
    uv run python backtest/canary.py
    uv run python backtest/canary.py --walk-forward
"""

import argparse
import sys
from datetime import date

import polars as pl

from backtest.engine import run_backtest
from backtest.tearsheet import print_tearsheet, tearsheet_dict
from backtest.walk_forward import walk_forward
from features.compute import load_features
from signals.momentum import (
    cross_sectional_momentum,
    get_monthly_rebalance_dates,
    momentum_weights,
)
from storage.parquet_store import load_bars
from storage.universe import universe_as_of_date

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Canary parameters ─────────────────────────────────────────────────────────
ASSET_CLASS = "equity"
TOP_N = 5
COST_BPS = 5.0
WEIGHT_SCHEME = "equal"

# Sharpe sanity bounds — fail loudly if outside range
SHARPE_MIN = 0.3
SHARPE_MAX = 2.5


def run_canary(
    start: date | None = None,
    end: date | None = None,
    walk_fwd: bool = False,
) -> dict:
    end = end or date.today()
    # Need at least 252 + 21 days of warm-up before usable signals
    start = start or date(end.year - 6, end.month, end.day)

    print(f"\nCanary backtest | {start} to {end} | top-{TOP_N} momentum | {COST_BPS}bps")

    # --- 1. Load universe and prices ---
    symbols = universe_as_of_date(ASSET_CLASS, end, require_data=True)
    print(f"Universe: {len(symbols)} symbols")

    prices = load_bars(symbols, start, end, ASSET_CLASS)
    if prices.is_empty():
        raise RuntimeError("No price data found. Run the historical backfill first.")
    print(f"Prices loaded: {len(prices)} rows")

    # --- 2. Load features from gold layer (or compute on-the-fly) ---
    features = load_features(symbols, start, end, ASSET_CLASS,
                             feature_list=["momentum_12m_1m", "realized_vol_21d"])
    if features.is_empty():
        raise RuntimeError("No features found. Run: uv run python features/compute.py")
    print(f"Features loaded: {len(features)} rows")

    if walk_fwd:
        print("\nRunning walk-forward validation (3yr train / 1yr OOS)...")
        wf_result = walk_forward(
            prices=prices,
            features=features,
            signal_fn=cross_sectional_momentum,
            weight_fn=momentum_weights,
            train_years=3,
            test_months=12,
            cost_bps=COST_BPS,
            top_n=TOP_N,
        )
        print(f"\nWalk-forward complete: {len(wf_result.folds)} folds")
        print(f"  Combined Sharpe  : {wf_result.combined_sharpe:.3f}")
        print(f"  Combined CAGR    : {wf_result.combined_cagr:.1%}")
        print(f"  Combined Max DD  : {wf_result.combined_max_drawdown:.1%}")
        for row in wf_result.summary():
            print(f"  Fold {row['fold']}: {row['test_start']} -> {row['test_end']} "
                  f"Sharpe={row['sharpe']:.2f} CAGR={row['cagr']:.1%}")
        return {"walk_forward": wf_result.summary(), "combined_sharpe": wf_result.combined_sharpe}

    # --- 3. Generate signal on full history ---
    trading_dates = sorted(prices["date"].unique().to_list())
    rebal_dates = get_monthly_rebalance_dates(start, end, trading_dates)
    print(f"Rebalance dates: {len(rebal_dates)} (monthly)")

    signal = cross_sectional_momentum(features, rebal_dates, top_n=TOP_N)
    weights = momentum_weights(signal, weight_scheme=WEIGHT_SCHEME)
    print(f"Signal generated: {len(signal)} rows across {len(rebal_dates)} rebalances")

    # --- 4. Run backtest ---
    result = run_backtest(prices, weights, cost_bps=COST_BPS)

    # --- 5. Tearsheet ---
    print_tearsheet(result, title=f"Canary: 12-1 Momentum Top-{TOP_N} ETF (equal weight)")

    # --- 6. Sanity check ---
    metrics = tearsheet_dict(result)
    if not (SHARPE_MIN <= result.sharpe <= SHARPE_MAX):
        print(f"WARNING: Sharpe {result.sharpe:.3f} is outside expected range "
              f"[{SHARPE_MIN}, {SHARPE_MAX}]. Check the pipeline for bugs.")
    else:
        print(f"Sanity check PASSED: Sharpe {result.sharpe:.3f} in [{SHARPE_MIN}, {SHARPE_MAX}]")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canary momentum backtest")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation instead of full-period backtest")
    args = parser.parse_args()
    run_canary(args.start, args.end, args.walk_forward)


if __name__ == "__main__":
    main()
