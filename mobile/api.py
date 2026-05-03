"""QuantPipe Mobile API — FastAPI backend for the iPhone PWA.

Read-only. Reads from existing parquet/JSON files produced by the daily pipeline.
Runs on port 8503 alongside the Streamlit desktop app (port 8501).

Endpoints:
    GET /               → serves the PWA shell (index.html)
    GET /static/*       → serves static assets
    GET /api/summary    → NAV, P&L, pipeline status, positions count
    GET /api/performance → equity curve, Sharpe, CAGR, drawdown metrics
    GET /api/portfolio  → current positions with weights and values
    GET /api/trades     → last 20 orders with slippage
    GET /api/health     → strategy health + pipeline timestamps
    GET /api/regime     → current macro regime from macro data
"""

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI
from fastapi.responses import FileNotFoundError, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from config.settings import DATA_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

app = FastAPI(title="QuantPipe Mobile", version="1.0.0", docs_url=None, redoc_url=None)

_STATIC = Path(__file__).parent / "static"
_GOLD   = DATA_DIR / "gold" / "equity"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(v) -> Any:
    """Convert NaN/inf/None to None for clean JSON."""
    if v is None:
        return None
    try:
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _read(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        log.warning("mobile/api: could not read %s: %s", path.name, exc)
        return None


def _heartbeat() -> dict:
    hb_path = PROJECT_ROOT / ".pipeline_heartbeat.json"
    if not hb_path.exists():
        return {"status": "unknown", "ts_utc": None, "date": None, "failures": []}
    try:
        return json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown", "ts_utc": None, "date": None, "failures": []}


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 10:
        return None
    import numpy as np
    arr  = np.array(values, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    mu   = float(rets.mean()) * 252
    sd   = float(rets.std()) * (252 ** 0.5)
    return _safe(round(mu / sd, 3)) if sd > 1e-10 else None


def _cagr(values: list[float], n_days: int) -> float | None:
    if len(values) < 2 or n_days < 1:
        return None
    n_years = n_days / 252
    ratio   = values[-1] / values[0]
    return _safe(round(ratio ** (1 / n_years) - 1, 4)) if ratio > 0 else None


def _max_drawdown(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    import numpy as np
    arr  = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak
    return _safe(round(float(dd.min()), 4))


# ── Static files & root ────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    index = _STATIC / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    return FileResponse(_STATIC / "manifest.json", media_type="application/json")


@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    return FileResponse(_STATIC / "sw.js", media_type="application/javascript")


# ── API: Summary ───────────────────────────────────────────────────────────────

@app.get("/api/summary")
async def summary():
    """NAV, daily P&L, pipeline status, active positions."""
    hb     = _heartbeat()
    th_df  = _read(_GOLD / "trading_history.parquet")
    tw_df  = _read(_GOLD / "target_weights.parquet")

    nav, cash, n_positions, prev_nav = None, None, 0, None

    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if not th.is_empty():
            latest = th.tail(1).to_dicts()[0]
            nav    = _safe(latest.get("nav"))
            cash   = _safe(latest.get("cash"))
            n_positions = int(latest.get("n_positions", 0))
            if len(th) >= 2:
                prev_nav = _safe(th[-2].to_dicts()[0].get("nav"))

    # Current positions count from latest target weights
    if tw_df is not None and not tw_df.is_empty():
        latest_date = tw_df["date"].max()
        lw = tw_df.filter(pl.col("date") == latest_date)
        n_positions = len(lw)

    daily_pnl      = None
    daily_pnl_pct  = None
    if nav is not None and prev_nav is not None and prev_nav > 0:
        daily_pnl     = _safe(round(nav - prev_nav, 2))
        daily_pnl_pct = _safe(round((nav - prev_nav) / prev_nav, 4))

    # Total return
    total_return = None
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if len(th) >= 2:
            first_nav = float(th.head(1)["nav"][0])
            if first_nav > 0 and nav is not None:
                total_return = _safe(round((nav - first_nav) / first_nav, 4))

    # 7-day sparkline
    sparkline = []
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date").tail(7)
        sparkline = [_safe(float(v)) for v in th["nav"].to_list()]

    return {
        "nav":           nav,
        "cash":          cash,
        "n_positions":   n_positions,
        "daily_pnl":     daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "total_return":  total_return,
        "sparkline":     sparkline,
        "pipeline": {
            "status":   hb.get("status", "unknown"),
            "ts_utc":   hb.get("ts_utc"),
            "date":     hb.get("date"),
            "failures": hb.get("failures", []),
            "elapsed_s": hb.get("elapsed_s"),
        },
    }


# ── API: Performance ───────────────────────────────────────────────────────────

@app.get("/api/performance")
async def performance(period: str = "all"):
    """Equity curve + key performance metrics."""
    tw_df  = _read(_GOLD / "target_weights.parquet")
    th_df  = _read(_GOLD / "trading_history.parquet")

    dates, values = [], []

    # Use trading_history NAV snapshots as equity curve
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")

        # Apply period filter
        cutoffs = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
        if period in cutoffs:
            cutoff = date.today() - timedelta(days=cutoffs[period])
            th = th.filter(pl.col("date") >= cutoff)

        if not th.is_empty():
            dates  = [str(d) for d in th["date"].to_list()]
            values = [_safe(float(v)) for v in th["nav"].to_list()]

    n_days = len(values)
    sharpe  = _sharpe(values)
    cagr    = _cagr(values, n_days)
    max_dd  = _max_drawdown(values)
    total_r = None
    if len(values) >= 2 and values[0] and values[-1]:
        total_r = _safe(round((values[-1] - values[0]) / values[0], 4))

    # Drawdown series
    dd_values = []
    if len(values) >= 2:
        import numpy as np
        arr  = np.array([v for v in values if v is not None], dtype=float)
        peak = np.maximum.accumulate(arr)
        dd   = ((arr - peak) / peak).tolist()
        dd_values = [_safe(round(v, 4)) for v in dd]

    # Current portfolio stats
    latest_weights = {}
    if tw_df is not None and not tw_df.is_empty():
        latest_date = tw_df["date"].max()
        lw = tw_df.filter(pl.col("date") == latest_date)
        n_pos = len(lw)
        gross = _safe(float(lw["weight"].sum()))
    else:
        n_pos, gross = 0, None

    return {
        "equity_curve": {"dates": dates, "values": values},
        "drawdown":     {"dates": dates[:len(dd_values)], "values": dd_values},
        "metrics": {
            "sharpe":       sharpe,
            "cagr":         cagr,
            "max_drawdown": max_dd,
            "total_return": total_r,
            "n_positions":  n_pos,
            "gross_exposure": gross,
        },
        "period": period,
    }


# ── API: Portfolio ─────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def portfolio():
    """Current positions with weights and estimated values."""
    tw_df = _read(_GOLD / "target_weights.parquet")
    th_df = _read(_GOLD / "trading_history.parquet")

    nav = None
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if not th.is_empty():
            nav = _safe(float(th.tail(1)["nav"][0]))

    positions = []
    gross_exposure = None

    if tw_df is not None and not tw_df.is_empty():
        latest_date = tw_df["date"].max()
        lw = tw_df.filter(pl.col("date") == latest_date).sort("weight", descending=True)
        gross_exposure = _safe(float(lw["weight"].sum()))

        for row in lw.iter_rows(named=True):
            w = float(row["weight"])
            positions.append({
                "symbol": row["symbol"],
                "weight": _safe(round(w, 4)),
                "value":  _safe(round(nav * w, 0)) if nav else None,
                "rebalance_date": str(row.get("rebalance_date", "")),
            })

    return {
        "positions":       positions,
        "n_positions":     len(positions),
        "nav":             nav,
        "gross_exposure":  gross_exposure,
        "as_of":           str(tw_df["date"].max()) if tw_df is not None and not tw_df.is_empty() else None,
    }


# ── API: Trades ────────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def trades():
    """Last 20 orders with slippage."""
    oj_df = _read(_GOLD / "order_journal.parquet")

    if oj_df is None or oj_df.is_empty():
        return {"trades": [], "total": 0}

    recent = (
        oj_df.filter(pl.col("broker") == "paper")
             .sort("ts_utc", descending=True)
             .head(20)
    )

    result = []
    for row in recent.iter_rows(named=True):
        est   = row.get("est_price")
        fill  = row.get("fill_price")
        slip  = None
        if est and fill and est > 0:
            slip = _safe(round((float(fill) - float(est)) / float(est) * 10_000, 1))

        qty = float(row.get("qty", 0))
        result.append({
            "date":          str(row.get("rebalance_date", "")),
            "symbol":        row.get("symbol", ""),
            "side":          "BUY" if qty > 0 else "SELL",
            "qty":           _safe(abs(round(qty, 0))),
            "est_price":     _safe(round(float(est), 2)) if est else None,
            "fill_price":    _safe(round(float(fill), 2)) if fill else None,
            "slippage_bps":  slip,
            "status":        row.get("status", ""),
            "order_id":      str(row.get("order_id", "")),
        })

    return {"trades": result, "total": len(oj_df)}


# ── API: Health ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Strategy health scores + pipeline status."""
    hb     = _heartbeat()
    sh_df  = _read(_GOLD / "strategy_health.parquet")

    strategies = []
    if sh_df is not None and not sh_df.is_empty():
        for row in sh_df.sort("status").iter_rows(named=True):
            strategies.append({
                "slug":            row.get("slug", ""),
                "name":            row.get("name", ""),
                "status":          row.get("status", ""),
                "oos_sharpe":      _safe(row.get("oos_sharpe")),
                "is_oos_ratio":    _safe(row.get("is_oos_ratio")),
                "max_drawdown":    _safe(row.get("max_drawdown")),
                "max_correlation": _safe(row.get("max_correlation")),
                "live_months":     _safe(row.get("live_months")),
                "flags":           row.get("flags", ""),
                "checked_at":      str(row.get("checked_at", "")),
            })

    # Pipeline schedule (server runs Mon-Fri 21:30 UTC)
    now_utc = datetime.now(timezone.utc)
    next_run = None
    for offset in range(8):
        candidate = (now_utc + timedelta(days=offset)).replace(
            hour=21, minute=30, second=0, microsecond=0
        )
        if candidate > now_utc and candidate.weekday() < 5:
            next_run = candidate.isoformat()
            break

    return {
        "pipeline": {
            "status":    hb.get("status", "unknown"),
            "ts_utc":    hb.get("ts_utc"),
            "date":      hb.get("date"),
            "failures":  hb.get("failures", []),
            "elapsed_s": hb.get("elapsed_s"),
            "next_run_utc": next_run,
        },
        "strategies": strategies,
        "n_healthy": sum(1 for s in strategies if s["status"] == "HEALTHY"),
        "n_watch":   sum(1 for s in strategies if s["status"] == "WATCH"),
        "n_flag":    sum(1 for s in strategies if s["status"] == "FLAG"),
        "n_new":     sum(1 for s in strategies if s["status"] == "NEW"),
    }


# ── API: Regime ────────────────────────────────────────────────────────────────

@app.get("/api/regime")
async def regime():
    """Current macro regime from the regime classifier."""
    try:
        from research.regime_classifier import (
            MacroRegime, REGIME_LABELS, REGIME_SECTORS, load_macro_data, classify_regime,
        )
        macro = load_macro_data()
        if not macro:
            return {"regime": None, "label": "No macro data — run pull_macro.py", "sectors": []}

        current = classify_regime(macro, date.today())
        return {
            "regime":  current.value,
            "label":   REGIME_LABELS[current],
            "sectors": REGIME_SECTORS[current],
        }
    except Exception as exc:
        log.debug("mobile/api regime: %s", exc)
        return {"regime": None, "label": "Unavailable", "sectors": []}
