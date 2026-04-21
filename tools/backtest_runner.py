"""Backtest runner — invoked as a subprocess by the Strategy Lab dashboard.

Outputs a single JSON object to stdout:
    {
      "ok": true,
      "metrics": { ... },
      "equity": { "dates": [...], "values": [...] },
      "benchmark": { "dates": [...], "values": [...] }   // SPY buy-and-hold
    }

On error:
    { "ok": false, "error": "..." }

Progress lines are printed to stderr so the dashboard can stream them.
"""

import json
import sys
from datetime import date

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(
    lookback_years: int = 6,
    top_n: int = 5,
    cost_bps: float = 5.0,
    weight_scheme: str = "equal",
) -> None:
    try:
        from backtest.canary import ASSET_CLASS
        from backtest.engine import run_backtest
        from backtest.tearsheet import tearsheet_dict
        from features.compute import load_features
        from signals.momentum import (
            cross_sectional_momentum,
            get_monthly_rebalance_dates,
            momentum_weights,
        )
        from storage.parquet_store import load_bars
        from storage.universe import universe_as_of_date

        end = date.today()
        start = date(end.year - lookback_years, end.month, end.day)

        _progress(f"Loading universe ({ASSET_CLASS})...")
        symbols = universe_as_of_date(ASSET_CLASS, end, require_data=True)
        _progress(f"Universe: {len(symbols)} symbols")

        _progress("Loading prices...")
        prices = load_bars(symbols, start, end, ASSET_CLASS)
        if prices.is_empty():
            raise RuntimeError("No price data found. Run the historical backfill first.")
        _progress(f"Prices: {len(prices)} rows")

        _progress("Loading features...")
        features = load_features(
            symbols, start, end, ASSET_CLASS,
            feature_list=["momentum_12m_1m", "realized_vol_21d"],
        )
        if features.is_empty():
            raise RuntimeError("No features found. Run: uv run python features/compute.py")
        _progress(f"Features: {len(features)} rows")

        _progress("Generating signals...")
        trading_dates = sorted(prices["date"].unique().to_list())
        rebal_dates = get_monthly_rebalance_dates(start, end, trading_dates)
        _progress(f"Rebalance dates: {len(rebal_dates)}")

        signal = cross_sectional_momentum(features, rebal_dates, top_n=top_n)
        weights = momentum_weights(signal, weight_scheme=weight_scheme)

        _progress("Running backtest...")
        result = run_backtest(prices, weights, cost_bps=cost_bps)

        _progress("Computing benchmark (SPY buy-and-hold)...")
        benchmark_dates = []
        benchmark_values = []
        try:
            spy_prices = load_bars(["SPY"], start, end, ASSET_CLASS)
            if not spy_prices.is_empty():
                import pandas as pd
                spy_pd = spy_prices.to_pandas().sort_values("date")
                spy_pd = spy_pd[spy_pd["symbol"] == "SPY"]
                if not spy_pd.empty:
                    closes = spy_pd.set_index("date")["close"]
                    spy_norm = 10_000 * closes / closes.iloc[0]
                    benchmark_dates = [str(d) for d in spy_norm.index]
                    benchmark_values = spy_norm.tolist()
        except Exception:
            pass

        metrics = tearsheet_dict(result)
        eq = result.equity_curve

        payload = {
            "ok": True,
            "metrics": metrics,
            "equity": {
                "dates": [str(d.date()) for d in eq.index],
                "values": eq.tolist(),
            },
            "benchmark": {
                "dates": benchmark_dates,
                "values": benchmark_values,
            },
            "params": {
                "lookback_years": lookback_years,
                "top_n": top_n,
                "cost_bps": cost_bps,
                "weight_scheme": weight_scheme,
            },
        }
        _progress("Done.")
        print(json.dumps(payload))

    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-years", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--weight-scheme", default="equal")
    args = parser.parse_args()

    run(
        lookback_years=args.lookback_years,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        weight_scheme=args.weight_scheme,
    )
