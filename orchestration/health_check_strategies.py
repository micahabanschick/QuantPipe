"""Automated strategy health checker.

Runs nightly as Step 6 of the pipeline. For each discovered strategy with a
cached backtest result, computes five quality metrics and assigns a health status:

  HEALTHY  — passes all thresholds
  WATCH    — 1 threshold below target OR fewer than 6 months of live paper data
  FLAG     — 2+ thresholds significantly below target
  NEW      — fewer than 30 days of live paper data (insufficient to evaluate)

Outputs: data/gold/equity/strategy_health.parquet
Columns: slug, name, status, is_sharpe, oos_sharpe, is_oos_ratio,
         max_drawdown, max_correlation, live_months, flags, checked_at

Multiple-comparison note: with 13+ strategies tested, the Sharpe threshold is
raised above the naive single-test level (Harvey, Liu & Zhu 2016).  The thresholds
below reflect an adjusted bar appropriate for a 10–15 strategy universe.

Usage (standalone):
    uv run python orchestration/health_check_strategies.py

Pipeline integration:
    Called by run_pipeline.py as Step 6 (best-effort, never aborts pipeline).
"""

import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import DATA_DIR, LOGS_DIR

log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

THRESHOLDS = {
    # OOS Sharpe: last-third of backtest treated as held-out
    "oos_sharpe_healthy":  0.50,
    "oos_sharpe_watch":    0.20,
    # IS/OOS ratio: how much better IS is vs OOS (overfitting signal)
    "is_oos_ratio_watch":  2.5,
    "is_oos_ratio_flag":   4.0,
    # Max drawdown (negative number)
    "max_dd_watch":       -0.25,
    "max_dd_flag":        -0.40,
    # Max pairwise correlation with any other strategy
    "correlation_watch":   0.65,
    "correlation_flag":    0.80,
    # Minimum live paper months before full evaluation
    "live_months_new":     1,
    "live_months_watch":   6,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, periods_per_year: float = 12.0) -> float:
    """Annualised Sharpe from a return array (any frequency)."""
    if len(returns) < 3:
        return float("nan")
    mu = float(returns.mean()) * periods_per_year
    sd = float(returns.std()) * np.sqrt(periods_per_year)
    return mu / sd if sd > 1e-10 else float("nan")


def _equity_to_monthly_returns(dates: list, values: list) -> np.ndarray:
    """Convert an equity curve to approximate monthly returns."""
    if len(values) < 2:
        return np.array([])
    arr = np.array(values, dtype=float)
    return np.diff(arr) / arr[:-1]


def _live_months(slug: str) -> float:
    """Return months of live paper data for this strategy from trading_history."""
    th_path = DATA_DIR / "gold" / "equity" / "trading_history.parquet"
    if not th_path.exists():
        return 0.0
    try:
        df = pl.read_parquet(th_path)
        if df.is_empty():
            return 0.0
        earliest = df["date"].min()
        days = (date.today() - earliest).days
        return round(days / 30.44, 1)
    except Exception:
        return 0.0


# ── Health computation ─────────────────────────────────────────────────────────

def compute_health(
    slug: str,
    name: str,
    equity_dates: list,
    equity_values: list,
    metrics: dict,
    all_returns: dict[str, np.ndarray],   # {slug: monthly_returns} for correlation
    live_months: float,
) -> dict:
    """Compute health metrics for one strategy and assign a status.

    Returns a dict row ready to append to the health parquet.
    """
    rets = _equity_to_monthly_returns(equity_dates, equity_values)
    is_sharpe = float(metrics.get("sharpe", float("nan")))
    max_dd    = float(metrics.get("max_drawdown", float("nan")))

    # OOS estimate: last 1/3 of backtest as "held-out"
    oos_n = max(len(rets) // 3, 6)
    oos_rets = rets[-oos_n:] if len(rets) >= oos_n * 2 else np.array([])
    oos_sharpe = _sharpe(oos_rets) if len(oos_rets) >= 6 else float("nan")

    # IS/OOS ratio — overfitting signal
    is_oos_ratio = float("nan")
    if not np.isnan(is_sharpe) and not np.isnan(oos_sharpe) and abs(oos_sharpe) > 1e-10:
        is_oos_ratio = round(is_sharpe / oos_sharpe, 2)

    # Max pairwise correlation with any other strategy
    max_corr = 0.0
    if slug in all_returns and len(all_returns) > 1:
        my_rets = all_returns[slug]
        corrs = []
        for other_slug, other_rets in all_returns.items():
            if other_slug == slug:
                continue
            n = min(len(my_rets), len(other_rets))
            if n >= 12:
                corrs.append(abs(float(np.corrcoef(my_rets[-n:], other_rets[-n:])[0, 1])))
        max_corr = round(max(corrs), 3) if corrs else 0.0

    # ── Flag accumulation ──────────────────────────────────────────────────────
    flags: list[str] = []

    if not np.isnan(oos_sharpe):
        if oos_sharpe < THRESHOLDS["oos_sharpe_watch"]:
            flags.append("low_oos_sharpe")

    if not np.isnan(is_oos_ratio):
        if is_oos_ratio > THRESHOLDS["is_oos_ratio_flag"]:
            flags.append("severe_overfit")
        elif is_oos_ratio > THRESHOLDS["is_oos_ratio_watch"]:
            flags.append("possible_overfit")

    if not np.isnan(max_dd):
        if max_dd < THRESHOLDS["max_dd_flag"]:
            flags.append("excessive_drawdown")
        elif max_dd < THRESHOLDS["max_dd_watch"]:
            flags.append("high_drawdown")

    if max_corr > THRESHOLDS["correlation_flag"]:
        flags.append("highly_redundant")
    elif max_corr > THRESHOLDS["correlation_watch"]:
        flags.append("possibly_redundant")

    # ── Status assignment ──────────────────────────────────────────────────────
    severe_flags = {"low_oos_sharpe", "severe_overfit", "excessive_drawdown", "highly_redundant"}
    n_flags  = len(flags)
    n_severe = sum(1 for f in flags if f in severe_flags)

    if live_months < THRESHOLDS["live_months_new"]:
        status = "NEW"
    elif n_severe >= 2 or n_flags >= 3:
        status = "FLAG"
    elif n_flags >= 1 or live_months < THRESHOLDS["live_months_watch"]:
        status = "WATCH"
    else:
        status = "HEALTHY"

    return {
        "slug":          slug,
        "name":          name,
        "status":        status,
        "is_sharpe":     round(is_sharpe, 3) if not np.isnan(is_sharpe) else None,
        "oos_sharpe":    round(oos_sharpe, 3) if not np.isnan(oos_sharpe) else None,
        "is_oos_ratio":  is_oos_ratio if not np.isnan(float(is_oos_ratio or float("nan"))) else None,
        "max_drawdown":  round(max_dd, 3) if not np.isnan(max_dd) else None,
        "max_correlation": max_corr,
        "live_months":   live_months,
        "flags":         ",".join(flags) if flags else "",
        "checked_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    from portfolio.multi_strategy import discover_strategies
    from portfolio._backtest_cache import load as cache_load

    metas  = discover_strategies()
    if not metas:
        log.info("health_check: no strategies found — skipping")
        return 0

    # Load all cached backtest results (don't re-run; use what exists)
    results = {}
    for m in metas:
        cached = cache_load(m.slug)
        if cached is not None:
            results[m.slug] = cached

    if not results:
        log.info("health_check: no cached backtest results yet — run backtests first")
        return 0

    log.info("health_check: evaluating %d strategies", len(results))

    # Pre-compute monthly returns per strategy for pairwise correlation
    all_returns: dict[str, np.ndarray] = {}
    for slug, r in results.items():
        rets = _equity_to_monthly_returns(r.equity_dates, r.equity_values)
        if len(rets) >= 12:
            all_returns[slug] = rets

    # Compute health for each strategy
    rows = []
    for slug, r in results.items():
        live_m = _live_months(slug)
        row = compute_health(
            slug=slug, name=r.name,
            equity_dates=r.equity_dates, equity_values=r.equity_values,
            metrics=r.metrics, all_returns=all_returns, live_months=live_m,
        )
        rows.append(row)
        log.info(
            "health_check: %s — %s (OOS Sharpe=%.2f, flags=%s)",
            slug, row["status"],
            row["oos_sharpe"] or float("nan"),
            row["flags"] or "none",
        )

    # Write output
    out_path = DATA_DIR / "gold" / "equity" / "strategy_health.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    tmp = out_path.with_suffix(".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, out_path)
    log.info("health_check: wrote %s (%d rows)", out_path.name, len(df))

    n_flag = sum(1 for r in rows if r["status"] == "FLAG")
    if n_flag:
        log.warning("health_check: %d strategy(ies) flagged — review in Blends tab", n_flag)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "health_check.log"),
        ],
    )
    raise SystemExit(main())
