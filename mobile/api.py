"""QuantPipe Mobile API — FastAPI backend for the iPhone PWA.

Read-only. Reads from existing parquet/JSON files produced by the daily pipeline.
Runs on port 8503 alongside the Streamlit desktop app (port 8501).

Endpoints:
    GET /                → PWA shell
    GET /static/*        → static assets
    GET /api/summary     → NAV, P&L, pipeline status, macro regime, ntfy topic
    GET /api/performance → equity curve + SPY benchmark, Sharpe, CAGR, drawdown
    GET /api/portfolio   → positions with weights, values, unrealized P&L
    GET /api/trades      → last 20 orders with slippage
    GET /api/health      → strategy health + pipeline timestamps
    GET /api/attribution → per-strategy P&L attribution (from backtest cache)
    GET /api/regime      → current macro regime + active sector ETFs
"""

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import DATA_DIR, PROJECT_ROOT, NTFY_TOPIC

log = logging.getLogger(__name__)

app = FastAPI(title="QuantPipe Mobile", version="2.0.0", docs_url=None, redoc_url=None)

_STATIC = Path(__file__).parent / "static"
_GOLD   = DATA_DIR / "gold" / "equity"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(v) -> Any:
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
    ratio = values[-1] / values[0]
    return _safe(round(ratio ** (252 / n_days) - 1, 4)) if ratio > 0 else None


def _max_drawdown(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    import numpy as np
    arr  = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak
    return _safe(round(float(dd.min()), 4))


def _spy_curve(start: date, end: date, target_dates: list[str]) -> list[float | None]:
    """Load SPY prices aligned to target_dates, normalised to match portfolio start."""
    try:
        from storage.parquet_store import load_bars
        bars = load_bars(["SPY"], start, end, "equity")
        if bars.is_empty():
            return []
        price_col = "adj_close" if "adj_close" in bars.columns else "close"
        spy = (
            bars.sort("date")
            .select(["date", price_col])
            .to_pandas()
            .set_index("date")[price_col]
        )
        import pandas as pd
        idx = pd.to_datetime(target_dates)
        aligned = spy.reindex(idx, method="ffill").values
        if len(aligned) < 2 or aligned[0] is None or math.isnan(float(aligned[0])):
            return []
        # Normalise to same starting value as portfolio (first clean value)
        scale = aligned[0]
        return [_safe(round(float(v) / scale, 6)) if not math.isnan(float(v)) else None
                for v in aligned]
    except Exception as exc:
        log.debug("mobile/api: spy_curve failed: %s", exc)
        return []


# ── Static & root ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    return FileResponse(_STATIC / "manifest.json", media_type="application/json")


@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    return FileResponse(_STATIC / "sw.js", media_type="application/javascript")


# ── API: Summary ───────────────────────────────────────────────────────────────

@app.get("/api/summary")
async def summary():
    hb    = _heartbeat()
    th_df = _read(_GOLD / "trading_history.parquet")
    tw_df = _read(_GOLD / "target_weights.parquet")

    nav, cash, n_positions, prev_nav = None, None, 0, None

    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if not th.is_empty():
            latest      = th.tail(1).to_dicts()[0]
            nav         = _safe(latest.get("nav"))
            cash        = _safe(latest.get("cash"))
            n_positions = int(latest.get("n_positions", 0))
            if len(th) >= 2:
                prev_nav = _safe(th.slice(-2, 1).to_dicts()[0].get("nav"))

    if tw_df is not None and not tw_df.is_empty():
        latest_date = tw_df["date"].max()
        n_positions = len(tw_df.filter(pl.col("date") == latest_date))

    daily_pnl = daily_pnl_pct = None
    if nav is not None and prev_nav is not None and prev_nav > 0:
        daily_pnl     = _safe(round(nav - prev_nav, 2))
        daily_pnl_pct = _safe(round((nav - prev_nav) / prev_nav, 4))

    total_return = None
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if len(th) >= 2 and nav is not None:
            first_nav = float(th.head(1)["nav"][0])
            if first_nav > 0:
                total_return = _safe(round((nav - first_nav) / first_nav, 4))

    sparkline = []
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date").tail(7)
        sparkline = [_safe(float(v)) for v in th["nav"].to_list()]

    # Macro regime for home summary
    regime_label, regime_sectors = None, []
    try:
        from research.regime_classifier import (
            REGIME_LABELS, REGIME_SECTORS, load_macro_data, classify_regime,
        )
        macro = load_macro_data()
        if macro:
            r = classify_regime(macro, date.today())
            regime_label   = REGIME_LABELS[r]
            regime_sectors = REGIME_SECTORS[r]
    except Exception:
        pass

    spy_today_pct = None
    try:
        from storage.parquet_store import load_bars
        spy_bars = load_bars(["SPY"], date.today() - timedelta(days=5), date.today(), "equity")
        if not spy_bars.is_empty():
            pc = "adj_close" if "adj_close" in spy_bars.columns else "close"
            spy_recent = spy_bars.sort("date").tail(2)
            if len(spy_recent) >= 2:
                p0 = float(spy_recent[pc][0])
                p1 = float(spy_recent[pc][1])
                if p0 > 0:
                    spy_today_pct = _safe(round((p1 - p0) / p0, 4))
    except Exception:
        pass

    return {
        "nav":           nav,
        "cash":          cash,
        "n_positions":   n_positions,
        "daily_pnl":     daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "total_return":  total_return,
        "sparkline":     sparkline,
        "ntfy_topic":    NTFY_TOPIC or None,
        "pipeline": {
            "status":    hb.get("status", "unknown"),
            "ts_utc":    hb.get("ts_utc"),
            "date":      hb.get("date"),
            "failures":  hb.get("failures", []),
            "elapsed_s": hb.get("elapsed_s"),
        },
        "regime":  {"label": regime_label, "sectors": regime_sectors},
        "market":  {"spy_today_pct": spy_today_pct},
    }


# ── API: Performance ───────────────────────────────────────────────────────────

@app.get("/api/performance")
async def performance(period: str = "all"):
    th_df = _read(_GOLD / "trading_history.parquet")
    tw_df = _read(_GOLD / "target_weights.parquet")

    dates, values = [], []

    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        cutoffs = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
        if period in cutoffs:
            th = th.filter(pl.col("date") >= date.today() - timedelta(days=cutoffs[period]))
        if not th.is_empty():
            pairs  = [(str(d), _safe(float(v)))
                      for d, v in zip(th["date"].to_list(), th["nav"].to_list())
                      if v is not None]
            dates  = [p[0] for p in pairs]
            values = [p[1] for p in pairs]

    clean  = [v for v in values if v is not None]
    n_days = len(clean)
    sharpe = _sharpe(clean)
    cagr   = _cagr(clean, n_days)
    max_dd = _max_drawdown(clean)
    total_r = None
    if len(clean) >= 2:
        total_r = _safe(round((clean[-1] - clean[0]) / clean[0], 4))

    dd_values = []
    if len(clean) >= 2:
        import numpy as np
        arr  = np.array(clean, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd_values = [_safe(round(v, 4)) for v in ((arr - peak) / peak).tolist()]

    # SPY benchmark — normalised to portfolio start value
    spy_values: list[float | None] = []
    if dates and values:
        start_d = date.fromisoformat(dates[0])
        end_d   = date.fromisoformat(dates[-1])
        raw_spy = _spy_curve(start_d, end_d, dates)
        if raw_spy and values[0] is not None:
            spy_values = [_safe(round(v * values[0], 2)) if v is not None else None
                          for v in raw_spy]

    n_pos, gross = 0, None
    if tw_df is not None and not tw_df.is_empty():
        lw = tw_df.filter(pl.col("date") == tw_df["date"].max())
        n_pos = len(lw)
        gross = _safe(float(lw["weight"].sum()))

    # Deployment markers from deployment_config.json
    markers: list[dict] = []
    try:
        cfg_path = _GOLD / "deployment_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            raw_dt = cfg.get("updated_at", "")
            deploy_date = raw_dt[:10] if raw_dt else None
            if deploy_date:
                ver = cfg.get("version", "")
                active = sorted(
                    [(s.get("slug", ""), s.get("name", s.get("slug", "")), s.get("allocation_weight", 0))
                     for s in cfg.get("strategies", [])
                     if s.get("slug") and s.get("active") and s.get("allocation_weight", 0) > 1e-6],
                    key=lambda x: -x[2],
                )
                if active:
                    top_name, top_w = active[0][1], active[0][2]
                    n_others = len(active) - 1
                    label = f"v{ver}: {top_name} ({round(top_w * 100)}%)"
                    if n_others:
                        label += f" +{n_others}"
                    markers.append({"date": deploy_date, "label": label})
    except Exception as exc:
        log.debug("mobile/api performance markers: %s", exc)

    return {
        "equity_curve": {"dates": dates, "values": values},
        "benchmark":    {"dates": dates, "values": spy_values, "label": "SPY"},
        "drawdown":     {"dates": dates[:len(dd_values)], "values": dd_values},
        "markers":      markers,
        "metrics": {
            "sharpe": sharpe, "cagr": cagr,
            "max_drawdown": max_dd, "total_return": total_r,
            "n_positions": n_pos, "gross_exposure": gross,
        },
        "period": period,
    }


# ── API: Portfolio ─────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def portfolio():
    tw_df = _read(_GOLD / "target_weights.parquet")
    th_df = _read(_GOLD / "trading_history.parquet")

    nav = None
    if th_df is not None and not th_df.is_empty():
        th = th_df.filter(pl.col("broker") == "paper").sort("date")
        if not th.is_empty():
            nav = _safe(float(th.tail(1)["nav"][0]))

    positions, gross_exposure = [], None

    if tw_df is not None and not tw_df.is_empty():
        lw = tw_df.filter(pl.col("date") == tw_df["date"].max()).sort("weight", descending=True)
        gross_exposure = _safe(float(lw["weight"].sum()))

        # Try to get latest prices for unrealized P&L
        symbols = lw["symbol"].to_list()
        latest_prices: dict[str, float] = {}
        try:
            from storage.parquet_store import load_bars
            price_df = load_bars(symbols, date.today() - timedelta(days=5), date.today(), "equity")
            if not price_df.is_empty():
                pc = "adj_close" if "adj_close" in price_df.columns else "close"
                for sym, grp in price_df.group_by("symbol"):
                    latest_prices[sym[0]] = float(grp.sort("date").tail(1)[pc][0])
        except Exception:
            pass

        for row in lw.iter_rows(named=True):
            sym = row["symbol"]
            w   = float(row["weight"])
            val = _safe(round(nav * w, 0)) if nav is not None else None
            price = latest_prices.get(sym)
            positions.append({
                "symbol":  sym,
                "weight":  _safe(round(w, 4)),
                "value":   val,
                "price":   _safe(round(price, 2)) if price else None,
                "rebalance_date": str(row.get("rebalance_date", "")),
            })

    return {
        "positions":      positions,
        "n_positions":    len(positions),
        "nav":            nav,
        "gross_exposure": gross_exposure,
        "as_of":          str(tw_df["date"].max()) if tw_df is not None and not tw_df.is_empty() else None,
    }


# ── API: Trades ────────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def trades():
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
        est  = row.get("est_price")
        fill = row.get("fill_price")
        slip = None
        if est is not None and fill is not None and float(est) > 0:
            slip = _safe(round((float(fill) - float(est)) / float(est) * 10_000, 1))
        qty = float(row.get("qty", 0))
        result.append({
            "date":         str(row.get("rebalance_date", "")),
            "symbol":       row.get("symbol", ""),
            "side":         "BUY" if qty > 0 else "SELL",
            "qty":          _safe(abs(round(qty, 0))),
            "est_price":    _safe(round(float(est), 2)) if est is not None else None,
            "fill_price":   _safe(round(float(fill), 2)) if fill is not None else None,
            "slippage_bps": slip,
            "status":       row.get("status", ""),
        })
    return {"trades": result, "total": len(oj_df)}


# ── API: Attribution ───────────────────────────────────────────────────────────

@app.get("/api/attribution")
async def attribution():
    """Per-strategy contribution to the blended portfolio P&L."""
    try:
        from portfolio.multi_strategy import discover_strategies
        from portfolio._backtest_cache import load as cache_load

        metas   = discover_strategies()
        results: dict[str, Any] = {}
        for m in metas:
            r = cache_load(m.slug)
            if r:
                results[m.slug] = r

        # Load current allocation
        alloc: dict[str, float] = {}
        cfg_path = _GOLD / "deployment_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            for s in cfg.get("strategies", []):
                if s.get("active"):
                    alloc[s["slug"]] = s.get("allocation_weight", 0.0)

        strategies = []
        for slug, r in results.items():
            m = r.metrics
            w = alloc.get(slug, 0.0)
            strategies.append({
                "slug":         slug,
                "name":         r.name,
                "weight":       _safe(round(w, 4)),
                "cagr":         _safe(m.get("cagr")),
                "sharpe":       _safe(m.get("sharpe")),
                "max_drawdown": _safe(m.get("max_drawdown")),
                "total_return": _safe(m.get("total_return")),
                "contribution": _safe(round(m.get("cagr", 0) * w, 4)) if m.get("cagr") and w else None,
            })

        strategies.sort(key=lambda x: -(x.get("contribution") or 0))
        return {"strategies": strategies, "n_active": sum(1 for s in strategies if (s["weight"] or 0) > 1e-6)}

    except Exception as exc:
        log.warning("mobile/api attribution: %s", exc)
        return {"strategies": [], "n_active": 0}


# ── API: Health ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    hb    = _heartbeat()
    sh_df = _read(_GOLD / "strategy_health.parquet")

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

    now_utc  = datetime.now(timezone.utc)
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
            "status":       hb.get("status", "unknown"),
            "ts_utc":       hb.get("ts_utc"),
            "date":         hb.get("date"),
            "failures":     hb.get("failures", []),
            "elapsed_s":    hb.get("elapsed_s"),
            "next_run_utc": next_run,
        },
        "strategies": strategies,
        "n_healthy": sum(s["status"] == "HEALTHY" for s in strategies),
        "n_watch":   sum(s["status"] == "WATCH"   for s in strategies),
        "n_flag":    sum(s["status"] == "FLAG"     for s in strategies),
        "n_new":     sum(s["status"] == "NEW"      for s in strategies),
    }


# ── API: Regime ────────────────────────────────────────────────────────────────

@app.get("/api/regime")
async def regime():
    try:
        from research.regime_classifier import (
            MacroRegime, REGIME_LABELS, REGIME_SECTORS, load_macro_data, classify_regime,
        )
        macro = load_macro_data()
        if not macro:
            return {"regime": None, "label": "No macro data — run pull_macro.py", "sectors": []}
        current = classify_regime(macro, date.today())
        return {"regime": current.value, "label": REGIME_LABELS[current], "sectors": REGIME_SECTORS[current]}
    except Exception as exc:
        log.debug("mobile/api regime: %s", exc)
        return {"regime": None, "label": "Unavailable", "sectors": []}
