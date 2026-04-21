"""Daily signal generation — runs after ingestion.

Pipeline:
  1. Load equity prices + features from gold layer
  2. Run cross-sectional momentum signal
  3. Compute target weights (equal-weight top-N)
  4. Run pre-trade risk checks; block if any limits violated
  5. Write target weights to data/gold/equity/target_weights.parquet (upsert by date)
  6. Append daily risk snapshot to data/gold/equity/portfolio_log.parquet

Exit codes: 0 = success, 1 = pre-trade check failed, 2 = fatal error.

Usage:
    uv run python orchestration/generate_signals.py
    uv run python orchestration/generate_signals.py --date 2024-01-15
"""

import logging
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from config.settings import DATA_DIR, LOGS_DIR
from config.universes import EQUITY_UNIVERSE
from features.compute import load_features
from risk.engine import RiskLimits, compute_exposures, generate_risk_report, historical_var
from risk.scenarios import run_all_scenarios
from signals.momentum import (
    cross_sectional_momentum,
    get_monthly_rebalance_dates,
    momentum_weights,
)
from storage.parquet_store import load_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "signals.log"),
    ],
)
log = logging.getLogger(__name__)

TARGET_WEIGHTS_PATH = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
PORTFOLIO_LOG_PATH = DATA_DIR / "gold" / "equity" / "portfolio_log.parquet"

TOP_N = 5
COST_BPS = 5.0
LOOKBACK_YEARS = 6


def _load_deployment_config_or_fallback():
    """Return (strategies list, is_config_driven).

    Each entry: {"slug": ..., "name": ..., "path": ..., "allocation_weight": ..., "params": {...}}
    Falls back to hardcoded momentum_top5 if no deployment config exists.
    """
    try:
        from portfolio.multi_strategy import read_deployment_config, discover_strategies
        cfg = read_deployment_config()
        if cfg is None:
            return None, False
        active = [s for s in cfg.strategies if s.active]
        if not active:
            return None, False
        meta_map = {m.slug: m for m in discover_strategies()}
        result = []
        for s in active:
            m = meta_map.get(s.slug)
            if m is None:
                log.warning(f"Deployed strategy {s.slug!r} not found in strategies/ — skipping.")
                continue
            result.append({
                "slug": s.slug,
                "name": s.name,
                "path": m.path,
                "allocation_weight": s.allocation_weight,
                "params": s.backtest_params,
            })
        if not result:
            return None, False
        return result, True
    except Exception as exc:
        log.warning(f"Could not load deployment config: {exc} — using default momentum signal")
        return None, False


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, df: pl.DataFrame) -> None:
    """Write df to path atomically via a .tmp sibling + os.replace()."""
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _upsert_target_weights(new_rows: pl.DataFrame) -> None:
    """Append new_rows to target_weights.parquet, replacing any rows for the same dates."""
    _ensure_parent(TARGET_WEIGHTS_PATH)
    if TARGET_WEIGHTS_PATH.exists():
        existing = pl.read_parquet(TARGET_WEIGHTS_PATH)
        dates_to_replace = new_rows["date"].unique()
        existing = existing.filter(~pl.col("date").is_in(dates_to_replace.to_list()))
        combined = pl.concat([existing, new_rows]).sort(["date", "symbol"])
    else:
        combined = new_rows.sort(["date", "symbol"])
    _atomic_write(TARGET_WEIGHTS_PATH, combined)


def _upsert_portfolio_log(new_row: dict) -> None:
    """Append one daily snapshot row to portfolio_log.parquet."""
    _ensure_parent(PORTFOLIO_LOG_PATH)
    new_df = pl.DataFrame([new_row])
    if PORTFOLIO_LOG_PATH.exists():
        existing = pl.read_parquet(PORTFOLIO_LOG_PATH)
        existing = existing.filter(pl.col("date") != new_row["date"])
        combined = pl.concat([existing, new_df]).sort("date")
    else:
        combined = new_df
    _atomic_write(PORTFOLIO_LOG_PATH, combined)


def run_generate_signals(as_of: date | None = None) -> int:
    """Generate signals for `as_of` date and persist results.

    Returns 0 on success, 1 on pre-trade check failure, 2 on fatal error.
    """
    if as_of is None:
        as_of = date.today()

    start = date(as_of.year - LOOKBACK_YEARS, as_of.month, as_of.day)
    log.info(f"=== Signal generation | as_of={as_of} | lookback: {start} to {as_of} ===")

    # 1. Load data
    try:
        symbols = [s for s in EQUITY_UNIVERSE if s]
        prices = load_bars(symbols, start, as_of, "equity")
        if prices.is_empty():
            log.error("No price data found. Run historical backfill first.")
            return 2

        features = load_features(symbols, start, as_of, "equity",
                                 feature_list=["momentum_12m_1m", "realized_vol_21d"])
        if features.is_empty():
            log.error("No features found. Run: uv run python features/compute.py")
            return 2

        log.info(f"Loaded {len(prices)} price rows, {len(features)} feature rows")
    except Exception as exc:
        log.error(f"Data load failed: {exc}")
        return 2

    # 2. Generate signal — config-driven multi-strategy or fallback to default momentum
    try:
        trading_dates = sorted(prices["date"].unique().to_list())
        rebal_dates = get_monthly_rebalance_dates(start, as_of, trading_dates)

        deployed, is_config = _load_deployment_config_or_fallback()

        if is_config and deployed:
            log.info(f"Config-driven mode: {len(deployed)} active strategies")
            import importlib.util as _ilu
            from portfolio.multi_strategy import blend_weights as _blend

            all_sym_weights: dict[str, dict[str, float]] = {}
            alloc_map: dict[str, float] = {}
            latest_rebal = None

            for strat in deployed:
                try:
                    spec = _ilu.spec_from_file_location("_strat_mod", strat["path"])
                    mod = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    p = strat["params"]
                    _top_n = int(p.get("top_n", TOP_N))
                    _wscheme = str(p.get("weight_scheme", "equal"))

                    sig = mod.get_signal(features, rebal_dates, top_n=_top_n, prices_df=prices)
                    wdf = mod.get_weights(sig, weight_scheme=_wscheme)

                    if wdf.is_empty():
                        log.warning(f"Strategy {strat['slug']!r} produced no weights — skipping.")
                        continue

                    date_col = "rebalance_date" if "rebalance_date" in wdf.columns else "date"
                    rd = wdf[date_col].max()
                    if latest_rebal is None or rd > latest_rebal:
                        latest_rebal = rd

                    wdf_latest = wdf.filter(pl.col(date_col) == rd)
                    all_sym_weights[strat["slug"]] = dict(zip(
                        wdf_latest["symbol"].to_list(),
                        wdf_latest["weight"].to_list(),
                    ))
                    alloc_map[strat["slug"]] = float(strat["allocation_weight"])
                    log.info(f"  {strat['name']}: {list(all_sym_weights[strat['slug']].keys())}")
                except Exception as exc:
                    log.error(f"Strategy {strat['slug']!r} failed: {exc}")

            if not all_sym_weights:
                log.error("All config-driven strategies failed — no weights generated.")
                return 2

            current_weights = _blend(all_sym_weights, alloc_map)
            # Trim tiny positions (< 0.1%)
            current_weights = {k: v for k, v in current_weights.items() if v >= 0.001}
        else:
            # Default: simple cross-sectional momentum
            signal = cross_sectional_momentum(features, rebal_dates, top_n=TOP_N)
            weights_df = momentum_weights(signal, weight_scheme="equal")

            if weights_df.is_empty():
                log.warning("No target weights generated (insufficient history?)")
                return 2

            latest_rebal = weights_df["rebalance_date"].max()
            current_weights_df = weights_df.filter(pl.col("rebalance_date") == latest_rebal)
            current_weights = dict(zip(
                current_weights_df["symbol"].to_list(),
                current_weights_df["weight"].to_list(),
            ))

        log.info(f"Signal generated. Latest rebalance: {latest_rebal}. "
                 f"Positions: {list(current_weights.keys())}")
    except Exception as exc:
        log.error(f"Signal generation failed: {exc}")
        return 2

    # 3. Pre-trade risk check
    try:
        from portfolio.covariance import compute_returns
        prices_filtered = prices.filter(pl.col("symbol").is_in(list(current_weights.keys())))
        # top-5 concentration cap is meaningless for a TOP_N=5 portfolio
        limits = RiskLimits(max_top5_concentration=1.0)

        returns_matrix, sym_order = compute_returns(prices_filtered)
        w_vec = np.array([current_weights.get(s, 0.0) for s in sym_order])
        var_95 = historical_var(returns_matrix, w_vec, confidence=0.95)
        var_99 = historical_var(returns_matrix, w_vec, confidence=0.99)

        from risk.engine import pre_trade_check
        check = pre_trade_check(current_weights, limits, returns_matrix=returns_matrix,
                                symbol_order=sym_order)

        if not check.passed:
            log.error("Pre-trade check FAILED — target weights will NOT be written.")
            for v in check.violations:
                log.error(f"  VIOLATION: {v}")
            # Persist snapshot with failure flag so the dashboard reflects reality,
            # then abort without publishing weights.
            snapshot = {
                "date": as_of,
                "n_positions": len(current_weights),
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "top5_concentration": 0.0,
                "var_1d_95": 0.0,
                "var_1d_99": 0.0,
                "pre_trade_passed": False,
                "worst_stress_scenario": "",
                "worst_stress_pnl": 0.0,
                "rebalance_date": latest_rebal,
            }
            _upsert_portfolio_log(snapshot)
            return 1
        else:
            log.info("Pre-trade check PASSED")
    except Exception as exc:
        log.error(f"Risk check failed: {exc}")
        var_95, var_99 = 0.0, 0.0
        check = None

    # 4. Compute exposures + stress scenarios
    try:
        exposures = compute_exposures(current_weights, as_of=as_of)
        stress = run_all_scenarios(current_weights)
        worst_scenario = min(stress, key=stress.get)
        worst_pnl = stress[worst_scenario]
        log.info(f"Exposures: gross={exposures.gross_exposure:.1%} "
                 f"net={exposures.net_exposure:.1%} top5={exposures.top_5_concentration:.1%}")
        log.info(f"Worst stress scenario: {worst_scenario} => {worst_pnl:.1%}")
    except Exception as exc:
        log.error(f"Exposure computation failed: {exc}")
        exposures = None
        worst_scenario, worst_pnl = "", 0.0

    # 5. Persist target weights
    try:
        weight_rows = pl.DataFrame({
            "date": [as_of] * len(current_weights),
            "symbol": list(current_weights.keys()),
            "weight": list(current_weights.values()),
            "rebalance_date": [latest_rebal] * len(current_weights),
        }).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("rebalance_date").cast(pl.Date),
        ])
        _upsert_target_weights(weight_rows)
        log.info(f"Target weights written: {len(current_weights)} positions")
    except Exception as exc:
        log.error(f"Failed to write target weights: {exc}")

    # 6. Persist portfolio snapshot
    try:
        snapshot = {
            "date": as_of,
            "n_positions": len(current_weights),
            "gross_exposure": exposures.gross_exposure if exposures else 0.0,
            "net_exposure": exposures.net_exposure if exposures else 0.0,
            "top5_concentration": exposures.top_5_concentration if exposures else 0.0,
            "var_1d_95": round(var_95, 6),
            "var_1d_99": round(var_99, 6),
            "pre_trade_passed": bool(check.passed) if check else False,
            "worst_stress_scenario": worst_scenario,
            "worst_stress_pnl": round(worst_pnl, 6),
            "rebalance_date": latest_rebal,
        }
        _upsert_portfolio_log(snapshot)
        log.info(f"Portfolio snapshot written for {as_of}")
    except Exception as exc:
        log.error(f"Failed to write portfolio log: {exc}")

    log.info("=== Signal generation complete ===")
    return 0 if (check is None or check.passed) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily signals and risk snapshot")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="As-of date (default: today)")
    args = parser.parse_args()
    sys.exit(run_generate_signals(args.date))


if __name__ == "__main__":
    main()
