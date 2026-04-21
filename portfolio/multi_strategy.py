"""Multi-strategy portfolio management.

Public API:
  discover_strategies()             → list[StrategyMeta]
  run_backtest(slug, path, params)  → CachedResult  (via subprocess)
  get_or_run_backtest(meta, params) → CachedResult  (cache-aware)
  build_return_matrix(results)      → pd.DataFrame  (index=date, cols=slug)
  strategy_correlation(matrix)      → pd.DataFrame  (N×N)
  optimize_allocations(matrix, method, max_w) → dict[slug, float]
  blend_weights(strategy_weights, allocations) → dict[symbol, float]
  read_deployment_config()          → DeploymentConfig | None
  write_deployment_config(config)   → None
  deployment_target_weights(config, results) → dict[symbol, float]
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR
from portfolio._backtest_cache import CachedResult, is_valid, load, save

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_STRATEGIES_DIR = _ROOT / "strategies"
_BACKTEST_RUNNER = _ROOT / "tools" / "backtest_runner.py"
_DEPLOYMENT_CONFIG_PATH = DATA_DIR / "gold" / "equity" / "deployment_config.json"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class StrategyMeta:
    slug: str
    name: str
    description: str
    path: Path
    default_params: dict


@dataclass
class DeployedStrategy:
    slug: str
    name: str
    active: bool
    allocation_weight: float
    backtest_params: dict


@dataclass
class DeploymentConfig:
    version: int
    updated_at: str
    strategies: list[DeployedStrategy]


# ── Strategy discovery ─────────────────────────────────────────────────────────

def discover_strategies() -> list[StrategyMeta]:
    """Scan strategies/ for folders containing a .py file with get_signal + get_weights."""
    found: list[StrategyMeta] = []
    for folder in sorted(_STRATEGIES_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_") or folder.name == "__pycache__":
            continue
        py = folder / f"{folder.name}.py"
        if not py.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_probe", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not (hasattr(mod, "get_signal") and hasattr(mod, "get_weights")):
                continue
            found.append(StrategyMeta(
                slug=folder.name,
                name=getattr(mod, "NAME", folder.name),
                description=getattr(mod, "DESCRIPTION", ""),
                path=py,
                default_params=getattr(mod, "DEFAULT_PARAMS", {}),
            ))
        except Exception as exc:
            log.warning(f"Could not load strategy {folder.name}: {exc}")
    return found


# ── Backtest runner ────────────────────────────────────────────────────────────

def run_backtest(
    slug: str,
    strategy_path: Path,
    params: dict,
    timeout: int = 300,
) -> CachedResult:
    """Run backtest_runner.py as a subprocess and return a CachedResult."""
    cmd = [
        sys.executable, str(_BACKTEST_RUNNER),
        "--strategy", str(strategy_path),
        "--lookback-years", str(params.get("lookback_years", 6)),
        "--top-n", str(params.get("top_n", 5)),
        "--cost-bps", str(params.get("cost_bps", 5.0)),
        "--weight-scheme", str(params.get("weight_scheme", "equal")),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_ROOT))
    if proc.returncode != 0:
        err = (proc.stdout + proc.stderr).strip()
        raise RuntimeError(f"Backtest subprocess failed for {slug!r}: {err[:400]}")

    # Parse JSON from stdout (stderr has progress lines)
    payload = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if payload is None or not payload.get("ok"):
        raise RuntimeError(
            f"Backtest runner returned no valid JSON for {slug!r}. "
            f"stderr: {proc.stderr[-400:]}"
        )

    return CachedResult(
        slug=slug,
        name=payload.get("strategy_name", slug),
        equity_dates=payload["equity"]["dates"],
        equity_values=payload["equity"]["values"],
        benchmark_dates=payload["benchmark"]["dates"],
        benchmark_values=payload["benchmark"]["values"],
        metrics=payload["metrics"],
        params=payload["params"],
        cached_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def get_or_run_backtest(
    meta: StrategyMeta,
    params: dict | None = None,
    force: bool = False,
) -> CachedResult:
    """Return cached result if valid; otherwise run backtest and cache."""
    _params = {**meta.default_params, **(params or {})}
    if not force and is_valid(meta.slug, meta.path):
        cached = load(meta.slug)
        if cached is not None:
            return cached
    result = run_backtest(meta.slug, meta.path, _params)
    save(result)
    return result


# ── Return-matrix construction ─────────────────────────────────────────────────

def build_return_matrix(results: list[CachedResult]) -> pd.DataFrame:
    """Align equity curves across strategies → daily return DataFrame (cols = slugs)."""
    series: dict[str, pd.Series] = {}
    for r in results:
        if not r.equity_dates:
            continue
        idx = pd.to_datetime(r.equity_dates)
        vals = pd.Series(r.equity_values, index=idx, name=r.slug)
        series[r.slug] = vals.pct_change().dropna()

    if not series:
        return pd.DataFrame()

    df = pd.concat(series.values(), axis=1, join="inner")
    df.columns = list(series.keys())
    return df.dropna()


def strategy_correlation(return_matrix: pd.DataFrame) -> pd.DataFrame:
    return return_matrix.corr()


# ── Strategy allocation optimizer ──────────────────────────────────────────────

def optimize_allocations(
    return_matrix: pd.DataFrame,
    method: str = "min_variance",
    max_weight: float = 0.80,
) -> dict[str, float]:
    """Compute allocation weights across strategies.

    method: "equal" | "vol_scaled" | "min_variance" | "max_sharpe"
    """
    slugs = list(return_matrix.columns)
    n = len(slugs)

    if n == 0:
        return {}
    if n == 1:
        return {slugs[0]: 1.0}
    if method == "equal":
        return {s: 1.0 / n for s in slugs}

    cov_arr = (return_matrix.cov() * 252).values
    mu_arr = (return_matrix.mean() * 252).values

    if method == "vol_scaled":
        vols = np.sqrt(np.maximum(np.diag(cov_arr), 1e-10))
        inv_v = 1.0 / vols
        w = inv_v / inv_v.sum()
        w = np.clip(w, 0.0, max_weight)
        w /= w.sum()
        return {s: float(w[i]) for i, s in enumerate(slugs)}

    # min_variance or max_sharpe via scipy
    try:
        from scipy.optimize import minimize

        def _min_var(w):
            return float(w @ cov_arr @ w)

        def _neg_sharpe(w):
            ret = float(mu_arr @ w)
            vol = float(np.sqrt(w @ cov_arr @ w + 1e-12))
            return -ret / vol

        obj = _min_var if method == "min_variance" else _neg_sharpe
        x0 = np.ones(n) / n
        bounds = [(0.0, max_weight)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success and res.x is not None:
            w = np.clip(res.x, 0.0, None)
            if w.sum() > 1e-10:
                w /= w.sum()
                return {s: float(w[i]) for i, s in enumerate(slugs)}
    except Exception as exc:
        log.warning(f"optimize_allocations({method}) failed: {exc}; falling back to equal")

    return {s: 1.0 / n for s in slugs}


# ── Weight blending ────────────────────────────────────────────────────────────

def blend_weights(
    strategy_weights: dict[str, dict[str, float]],
    allocations: dict[str, float],
) -> dict[str, float]:
    """Combine symbol-level weights across strategies by allocation weight.

    strategy_weights : {slug: {symbol: weight, ...}}
    allocations      : {slug: allocation_weight}  (should sum to 1)

    Returns {symbol: blended_weight} normalised to sum ≤ 1.
    """
    total_alloc = sum(allocations.values()) or 1.0
    combined: dict[str, float] = {}
    for slug, sym_weights in strategy_weights.items():
        alloc = allocations.get(slug, 0.0) / total_alloc
        for sym, w in sym_weights.items():
            combined[sym] = combined.get(sym, 0.0) + alloc * w
    return combined


# ── Deployment config I/O ──────────────────────────────────────────────────────

def read_deployment_config() -> Optional[DeploymentConfig]:
    if not _DEPLOYMENT_CONFIG_PATH.exists():
        return None
    try:
        raw = json.loads(_DEPLOYMENT_CONFIG_PATH.read_text(encoding="utf-8"))
        strategies = [DeployedStrategy(**s) for s in raw.get("strategies", [])]
        return DeploymentConfig(
            version=raw.get("version", 1),
            updated_at=raw.get("updated_at", ""),
            strategies=strategies,
        )
    except Exception as exc:
        log.warning(f"Could not read deployment config: {exc}")
        return None


_DEPLOYMENT_HISTORY_PATH = DATA_DIR / "gold" / "equity" / "deployment_history.jsonl"


def write_deployment_config(config: DeploymentConfig) -> None:
    _DEPLOYMENT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_version = config.version + 1
    payload = {
        "version": new_version,
        "updated_at": now,
        "strategies": [asdict(s) for s in config.strategies],
    }
    tmp = _DEPLOYMENT_CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, _DEPLOYMENT_CONFIG_PATH)

    # Append to immutable deployment history for the trading dashboards
    active = [s for s in config.strategies if s.active]
    history_entry = {
        "timestamp": now,
        "version": new_version,
        "strategies": [
            {"slug": s.slug, "name": s.name, "allocation_weight": s.allocation_weight}
            for s in active
        ],
    }
    _DEPLOYMENT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _DEPLOYMENT_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")


def deployment_target_weights(
    config: DeploymentConfig,
    results_by_slug: dict[str, CachedResult],
) -> dict[str, float]:
    """Compute blended symbol weights for all active deployed strategies.

    Uses the most recent rebalance weights from each strategy's backtest result.
    In production, this should be replaced with live signal weights.
    """
    active = [s for s in config.strategies if s.active]
    if not active:
        return {}

    alloc_map = {s.slug: s.allocation_weight for s in active}
    total = sum(alloc_map.values()) or 1.0
    alloc_norm = {slug: w / total for slug, w in alloc_map.items()}

    combined: dict[str, float] = {}
    for deployed in active:
        slug = deployed.slug
        if slug not in results_by_slug:
            continue
        # Weights from the result are equity-curve only; we use allocation_weight
        # The actual live weights come from target_weights.parquet (generate_signals)
        alloc = alloc_norm.get(slug, 0.0)
        # placeholder — in a live system we'd load per-strategy weights here
        combined[slug] = alloc  # record as meta-weight

    return combined
