"""Position reconciler — compares internal ledger to broker-reported positions.

Detects drift between what QuantPipe thinks it holds and what the broker
actually holds. Material drift must be investigated and resolved before the
next rebalance — automated position correction is intentionally NOT done here.

Public API:
    reconcile(internal, broker, prices, nav) -> ReconcileReport
    has_material_drift(report, threshold_pct) -> bool
    format_reconcile_report(report) -> str
    write_reconcile_log(report, path) -> None
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from .base import Position


@dataclass
class PositionDrift:
    symbol: str
    internal_qty: float
    broker_qty: float
    qty_diff: float
    notional_diff: float    # absolute value, in dollars
    drift_pct: float        # |qty_diff / internal_qty| * 100 — NaN if internal is zero


@dataclass
class ReconcileReport:
    as_of: datetime
    drifts: list[PositionDrift] = field(default_factory=list)
    symbols_only_internal: list[str] = field(default_factory=list)
    symbols_only_broker: list[str] = field(default_factory=list)
    total_notional_drift: float = 0.0
    nav: float = 0.0

    @property
    def is_clean(self) -> bool:
        return (
            not self.drifts
            and not self.symbols_only_internal
            and not self.symbols_only_broker
        )


def reconcile(
    internal: list[Position],
    broker: list[Position],
    prices: dict[str, float],
    nav: float,
) -> ReconcileReport:
    """Compare internal position ledger to live broker positions.

    Parameters
    ----------
    internal : Positions as recorded by QuantPipe's internal ledger
    broker   : Live positions returned by broker.get_positions()
    prices   : {symbol: price} for notional calculations
    nav      : Total portfolio NAV for context

    Returns
    -------
    ReconcileReport with any drifts identified.
    """
    internal_map = {p.symbol: p.qty for p in internal}
    broker_map = {p.symbol: p.qty for p in broker}

    all_symbols = set(internal_map) | set(broker_map)
    drifts: list[PositionDrift] = []
    total_drift = 0.0

    for sym in sorted(all_symbols):
        i_qty = internal_map.get(sym, 0.0)
        b_qty = broker_map.get(sym, 0.0)
        diff = b_qty - i_qty

        if abs(diff) < 1e-9:
            continue

        price = prices.get(sym, 0.0)
        notional = abs(diff * price)
        drift_pct = (abs(diff / i_qty) * 100) if abs(i_qty) > 1e-9 else float("nan")

        drifts.append(PositionDrift(
            symbol=sym,
            internal_qty=i_qty,
            broker_qty=b_qty,
            qty_diff=diff,
            notional_diff=round(notional, 2),
            drift_pct=round(drift_pct, 2),
        ))
        total_drift += notional

    only_internal = [s for s in internal_map if s not in broker_map and abs(internal_map[s]) > 1e-9]
    only_broker = [s for s in broker_map if s not in internal_map and abs(broker_map[s]) > 1e-9]

    return ReconcileReport(
        as_of=datetime.now(timezone.utc),
        drifts=drifts,
        symbols_only_internal=only_internal,
        symbols_only_broker=only_broker,
        total_notional_drift=round(total_drift, 2),
        nav=nav,
    )


def has_material_drift(report: ReconcileReport, threshold_pct: float = 5.0) -> bool:
    """Return True if any position has drifted more than threshold_pct of its quantity,
    or if there are symbols present in one ledger but not the other."""
    if report.symbols_only_internal or report.symbols_only_broker:
        return True
    for d in report.drifts:
        if abs(d.qty_diff) > 1e-9 and (
            float("nan") != d.drift_pct and d.drift_pct > threshold_pct
        ):
            return True
    return False


def format_reconcile_report(report: ReconcileReport) -> str:
    lines = [
        f"Reconcile Report — {report.as_of.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"NAV: ${report.nav:,.0f}  |  Total notional drift: ${report.total_notional_drift:,.2f}",
    ]

    if report.is_clean:
        lines.append("STATUS: CLEAN — all positions match.")
        return "\n".join(lines)

    lines.append(f"STATUS: DRIFT DETECTED")

    if report.symbols_only_internal:
        lines.append(f"\nInternal-only (missing from broker): {report.symbols_only_internal}")
    if report.symbols_only_broker:
        lines.append(f"\nBroker-only (missing from internal): {report.symbols_only_broker}")

    if report.drifts:
        lines.append(f"\n{'Symbol':<10} {'Internal':>12} {'Broker':>12} {'Diff':>10} {'Notional':>12} {'Drift%':>8}")
        lines.append("-" * 68)
        for d in report.drifts:
            drift_str = f"{d.drift_pct:.1f}%" if d.drift_pct == d.drift_pct else "N/A"
            lines.append(
                f"{d.symbol:<10} {d.internal_qty:>12.2f} {d.broker_qty:>12.2f} "
                f"{d.qty_diff:>+10.2f} ${d.notional_diff:>10,.2f} {drift_str:>8}"
            )
    return "\n".join(lines)


def write_reconcile_log(report: ReconcileReport, path: Path) -> None:
    """Append a reconcile snapshot to a Parquet log file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if report.is_clean:
        rows.append({
            "as_of": report.as_of.date(),
            "symbol": "__CLEAN__",
            "internal_qty": 0.0,
            "broker_qty": 0.0,
            "qty_diff": 0.0,
            "notional_diff": 0.0,
            "drift_pct": 0.0,
            "nav": report.nav,
        })
    else:
        for d in report.drifts:
            rows.append({
                "as_of": report.as_of.date(),
                "symbol": d.symbol,
                "internal_qty": d.internal_qty,
                "broker_qty": d.broker_qty,
                "qty_diff": d.qty_diff,
                "notional_diff": d.notional_diff,
                "drift_pct": d.drift_pct if d.drift_pct == d.drift_pct else 0.0,
                "nav": report.nav,
            })
        for sym in report.symbols_only_internal:
            rows.append({
                "as_of": report.as_of.date(), "symbol": sym,
                "internal_qty": 1.0, "broker_qty": 0.0,
                "qty_diff": -1.0, "notional_diff": 0.0, "drift_pct": 100.0,
                "nav": report.nav,
            })
        for sym in report.symbols_only_broker:
            rows.append({
                "as_of": report.as_of.date(), "symbol": sym,
                "internal_qty": 0.0, "broker_qty": 1.0,
                "qty_diff": 1.0, "notional_diff": 0.0, "drift_pct": float("inf"),
                "nav": report.nav,
            })

    new_df = pl.DataFrame(rows).with_columns(pl.col("as_of").cast(pl.Date))

    if path.exists():
        existing = pl.read_parquet(path)
        today = report.as_of.date()
        existing = existing.filter(pl.col("as_of") != today)
        combined = pl.concat([existing, new_df]).sort("as_of")
    else:
        combined = new_df

    combined.write_parquet(path)
