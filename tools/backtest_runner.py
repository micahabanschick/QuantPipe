"""Backtest runner -- invoked as a subprocess by the Strategy Lab.

stdout: one JSON line per run (ok=true or ok=false).
stderr: progress lines shown in the dashboard console expander.
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
    p = Path(strategy_path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Strategy file not found: {p}")
    spec = importlib.util.spec_from_file_location("_strategy", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("get_signal", "get_weights"):
        if not hasattr(mod, attr):
            raise AttributeError(f"Strategy {p.name!r} missing {attr!r}")
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
        from backtest.tearsheet import (
            alpha_beta, drawdown_series, information_ratio,
            monthly_returns_matrix, rolling_sharpe_series,
            tearsheet_dict, tracking_error,
        )
        from features.compute import load_features
        from signals.momentum import get_monthly_rebalance_dates
        from storage.parquet_store import load_bars
        from storage.universe import universe_as_of_date

        _progress(f"Loading strategy: {Path(strategy_path).name}")
        strategy = _load_strategy(strategy_path)
        defaults = getattr(strategy, "DEFAULT_PARAMS", {})

        _top_n    = top_n         if top_n > 0    else defaults.get("top_n", 5)
        _cost_bps = cost_bps      if cost_bps >= 0 else defaults.get("cost_bps", 5.0)
        _scheme   = weight_scheme if weight_scheme  else defaults.get("weight_scheme", "equal")
        _lb       = lookback_years if lookback_years else defaults.get("lookback_years", 6)

        end   = date.today()
        start = date(end.year - _lb, end.month, end.day)

        _progress(f"Universe: equity | {start} -> {end}")
        symbols = universe_as_of_date("equity", end, require_data=True)
        _progress(f"  {len(symbols)} symbols")

        _progress("Loading prices...")
        prices = load_bars(symbols, start, end, "equity")
        if prices.is_empty():
            raise RuntimeError("No price data -- run the historical backfill first.")
        _progress(f"  {len(prices)} rows")

        _progress("Loading features...")
        features = load_features(
            symbols, start, end, "equity",
            feature_list=["momentum_12m_1m", "realized_vol_21d"],
        )
        if features.is_empty():
            raise RuntimeError("No features -- run: uv run python features/compute.py")
        _progress(f"  {len(features)} rows")

        _progress("Generating signals...")
        trading_dates = sorted(prices["date"].unique().to_list())
        rebal_dates = get_monthly_rebalance_dates(start, end, trading_dates)
        _progress(f"  {len(rebal_dates)} rebalance dates")

        signal  = strategy.get_signal(features, rebal_dates, top_n=_top_n, prices_df=prices)
        weights = strategy.get_weights(signal, weight_scheme=_scheme)

        _progress("Running backtest engine...")
        result = run_backtest(prices, weights, cost_bps=_cost_bps)

        _progress("Computing SPY benchmark...")
        bench_dates, bench_values, bench_returns = [], [], None
        try:
            spy_prices = load_bars(["SPY"], start, end, "equity")
            if not spy_prices.is_empty():
                import pandas as pd
                spy_pd = (
                    spy_prices.to_pandas().sort_values("date")
                    .pipe(lambda df: df[df["symbol"] == "SPY"])
                )
                if not spy_pd.empty:
                    closes = spy_pd.set_index("date")["close"]
                    spy_norm = 10_000 * closes / closes.iloc[0]
                    bench_dates  = [str(d) for d in spy_norm.index]
                    bench_values = [round(v, 2) for v in spy_norm.tolist()]
                    bench_returns = spy_norm.pct_change().fillna(0.0)
        except Exception as e:
            _progress(f"  SPY benchmark failed: {e}")

        _progress("Computing analytics...")
        metrics   = tearsheet_dict(result)
        eq        = result.equity_curve
        strat_ret = result.returns

        roll_win    = min(252, max(20, len(eq) // 4))
        roll_sharpe = rolling_sharpe_series(strat_ret, window=roll_win)
        rolling_sharpe_data = {
            "dates":  [str(d.date()) for d in roll_sharpe.index],
            "values": [round(v, 3) for v in roll_sharpe.tolist()],
            "window": roll_win,
        }

        dd = drawdown_series(eq)
        drawdown_data = {
            "dates":  [str(d.date()) for d in dd.index],
            "values": [round(v, 3) for v in dd.tolist()],
        }

        monthly_mat  = monthly_returns_matrix(eq)
        monthly_data = {}
        if not monthly_mat.empty:
            for yr, row in monthly_mat.iterrows():
                monthly_data[str(int(yr))] = {
                    m: (round(float(v), 4) if v == v else None)
                    for m, v in row.items()
                }

        trades = result.trades.copy()
        if not trades.empty:
            trades["date"] = trades["date"].astype(str)
        trade_log = trades.to_dict(orient="records")

        alpha_val, beta_val, te_val, ir_val = 0.0, 1.0, 0.0, 0.0
        if bench_returns is not None and len(bench_returns) > 20:
            import pandas as pd
            bench_ser = pd.Series(
                bench_returns.values,
                index=pd.to_datetime([str(d) for d in bench_returns.index]),
            )
            alpha_val, beta_val = alpha_beta(strat_ret, bench_ser)
            te_val = tracking_error(strat_ret, bench_ser)
            ir_val = information_ratio(strat_ret, bench_ser)

        _progress("Done.")
        payload = {
            "ok": True,
            "strategy_name": getattr(strategy, "NAME", Path(strategy_path).stem),
            "metrics":          metrics,
            "equity":           {"dates": [str(d.date()) for d in eq.index],
                                 "values": [round(v, 2) for v in eq.tolist()]},
            "benchmark":        {"dates": bench_dates, "values": bench_values},
            "rolling_sharpe":   rolling_sharpe_data,
            "drawdown_pct":     drawdown_data,
            "monthly_returns":  monthly_data,
            "trade_log":        trade_log,
            "alpha":            alpha_val,
            "beta":             beta_val,
            "tracking_error":   te_val,
            "information_ratio": ir_val,
            "params": {
                "lookback_years": _lb,
                "top_n":          _top_n,
                "cost_bps":       _cost_bps,
                "weight_scheme":  _scheme,
            },
        }
        print(json.dumps(payload))

    except Exception as exc:
        import traceback
        _progress(traceback.format_exc())
        print(json.dumps({"ok": False, "error": str(exc)}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",       required=True)
    parser.add_argument("--lookback-years", type=int,   default=6)
    parser.add_argument("--top-n",          type=int,   default=0)
    parser.add_argument("--cost-bps",       type=float, default=-1)
    parser.add_argument("--weight-scheme",  default="")
    args = parser.parse_args()
    run(
        strategy_path  = args.strategy,
        lookback_years = args.lookback_years,
        top_n          = args.top_n,
        cost_bps       = args.cost_bps,
        weight_scheme  = args.weight_scheme,
    )
