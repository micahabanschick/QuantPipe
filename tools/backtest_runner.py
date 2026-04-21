"""Backtest runner — invoked as a subprocess by the Strategy Lab dashboard.

Loads a strategy from strategies/<file>.py via importlib, runs the full
backtest pipeline, and emits a single JSON payload to stdout.

stdout:
    { "ok": true,  "metrics": {...}, "equity": {...}, "benchmark": {...}, "params": {...} }
    { "ok": false, "error": "..." }

stderr: progress lines (streamed to the dashboard console expander).
"""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_strategy(strategy_path: str):
    """Import a strategy module from an absolute or ROOT-relative path."""
    p = Path(strategy_path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Strategy file not found: {p}")
    spec = importlib.util.spec_from_file_location("_strategy", p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("get_signal", "get_weights"):
        if not hasattr(mod, attr):
            raise AttributeError(f"Strategy {p.name!r} is missing required function '{attr}'")
    return mod


def run(
    strategy_path: str,
    lookback_years: int = 6,
    top_n: int = 5,
    cost_bps: float = 5.0,
    weight_scheme: str = "equal",
) -> None:
    try:
        from backtest.engine import run_backtest
        from backtest.tearsheet import tearsheet_dict
        from features.compute import load_features
        from signals.momentum import get_monthly_rebalance_dates
        from storage.parquet_store import load_bars
        from storage.universe import universe_as_of_date

        _progress(f"Loading strategy: {Path(strategy_path).name}")
        strategy = _load_strategy(strategy_path)
        defaults = getattr(strategy, "DEFAULT_PARAMS", {})

        _top_n         = top_n         if top_n         else defaults.get("top_n", 5)
        _cost_bps      = cost_bps      if cost_bps >= 0 else defaults.get("cost_bps", 5.0)
        _weight_scheme = weight_scheme if weight_scheme  else defaults.get("weight_scheme", "equal")
        _lookback      = lookback_years if lookback_years else defaults.get("lookback_years", 6)

        end   = date.today()
        start = date(end.year - _lookback, end.month, end.day)

        _progress(f"Universe: equity | {start} → {end}")
        symbols = universe_as_of_date("equity", end, require_data=True)
        _progress(f"  {len(symbols)} symbols")

        _progress("Loading prices…")
        prices = load_bars(symbols, start, end, "equity")
        if prices.is_empty():
            raise RuntimeError("No price data — run the historical backfill first.")
        _progress(f"  {len(prices)} rows")

        _progress("Loading features…")
        features = load_features(
            symbols, start, end, "equity",
            feature_list=["momentum_12m_1m", "realized_vol_21d"],
        )
        if features.is_empty():
            raise RuntimeError("No features — run: uv run python features/compute.py")
        _progress(f"  {len(features)} rows")

        _progress("Generating signals…")
        trading_dates = sorted(prices["date"].unique().to_list())
        rebal_dates   = get_monthly_rebalance_dates(start, end, trading_dates)
        _progress(f"  {len(rebal_dates)} rebalance dates")

        # Pass prices_df so strategies can use it directly instead of loading from storage.
        # This decouples get_signal() from the storage layer and makes it unit-testable.
        signal  = strategy.get_signal(features, rebal_dates, top_n=_top_n, prices_df=prices)
        weights = strategy.get_weights(signal, weight_scheme=_weight_scheme)

        _progress("Running backtest engine…")
        result = run_backtest(prices, weights, cost_bps=_cost_bps)

        _progress("Computing SPY benchmark…")
        benchmark_dates:  list[str]   = []
        benchmark_values: list[float] = []
        try:
            spy_prices = load_bars(["SPY"], start, end, "equity")
            if not spy_prices.is_empty():
                import pandas as pd
                spy_pd = (
                    spy_prices.to_pandas()
                    .sort_values("date")
                    .pipe(lambda df: df[df["symbol"] == "SPY"])
                )
                if not spy_pd.empty:
                    closes = spy_pd.set_index("date")["close"]
                    spy_norm = 10_000 * closes / closes.iloc[0]
                    benchmark_dates  = [str(d) for d in spy_norm.index]
                    benchmark_values = spy_norm.tolist()
        except Exception:
            pass

        metrics = tearsheet_dict(result)
        eq      = result.equity_curve

        payload = {
            "ok": True,
            "strategy_name": getattr(strategy, "NAME", Path(strategy_path).stem),
            "metrics": metrics,
            "equity": {
                "dates":  [str(d.date()) for d in eq.index],
                "values": eq.tolist(),
            },
            "benchmark": {
                "dates":  benchmark_dates,
                "values": benchmark_values,
            },
            "params": {
                "lookback_years": _lookback,
                "top_n":          _top_n,
                "cost_bps":       _cost_bps,
                "weight_scheme":  _weight_scheme,
            },
        }
        _progress("Done.")
        print(json.dumps(payload))

    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",       required=True,  help="Path to strategy .py file")
    parser.add_argument("--lookback-years", type=int,   default=6)
    parser.add_argument("--top-n",          type=int,   default=0, help="0 = use strategy default")
    parser.add_argument("--cost-bps",       type=float, default=-1, help="-1 = use strategy default")
    parser.add_argument("--weight-scheme",  default="",    help="empty = use strategy default")
    args = parser.parse_args()

    run(
        strategy_path  = args.strategy,
        lookback_years = args.lookback_years,
        top_n          = args.top_n,
        cost_bps       = args.cost_bps,
        weight_scheme  = args.weight_scheme,
    )
