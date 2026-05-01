"""Daily rebalance script — loads target weights, computes orders, places them.

Pipeline:
  1. Load latest target weights from data/gold/equity/target_weights.parquet
  2. Connect to broker (paper by default; live requires .env credentials)
  3. Fetch current prices, positions, and NAV from broker
  4. Run pre-trade risk check — abort if any hard limit is violated
  5. Compute minimal order set via trader.compute_orders()
  6. Place orders (skipped in --dry-run mode)
  7. Wait briefly then reconcile broker positions against internal ledger
  8. Write reconcile log to data/gold/equity/reconcile_log.parquet
  9. Send Pushover/ntfy alert on material drift or pre-trade failure

Exit codes: 0 = success, 1 = pre-trade blocked / drift detected, 2 = fatal error.

Usage:
    uv run python orchestration/rebalance.py --broker paper
    uv run python orchestration/rebalance.py --broker ibkr --dry-run
    uv run python orchestration/rebalance.py --broker paper --date 2024-06-01
"""

import logging
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from config.settings import DATA_DIR, LOGS_DIR
from orchestration._halt import check_halt
from execution.order_journal import append_order, update_order_fill
from execution.reconciler import (
    format_reconcile_report,
    has_material_drift,
    reconcile,
    write_reconcile_log,
)
from execution.trader import compute_orders, nav_from_positions, orders_summary
from risk.engine import RiskLimits, pre_trade_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "rebalance.log"),
    ],
)
log = logging.getLogger(__name__)

TARGET_WEIGHTS_PATH = DATA_DIR / "gold" / "equity" / "target_weights.parquet"
RECONCILE_LOG_PATH  = DATA_DIR / "gold" / "equity" / "reconcile_log.parquet"
ORDER_JOURNAL_PATH  = DATA_DIR / "gold" / "equity" / "order_journal.parquet"

# Seconds to wait after placing orders before reconciling
RECONCILE_DELAY = 5


def _send_alert(message: str) -> None:
    from config.settings import NTFY_TOPIC, PUSHOVER_TOKEN, PUSHOVER_USER
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        try:
            import requests
            requests.post("https://api.pushover.net/1/messages.json", data={
                "token": PUSHOVER_TOKEN, "user": PUSHOVER_USER,
                "message": message, "title": "QuantPipe Rebalance",
            }, timeout=10)
        except Exception as exc:
            log.warning(f"Alert failed: {exc}")
    elif NTFY_TOPIC:
        try:
            import requests
            requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode(), timeout=10)
        except Exception as exc:
            log.warning(f"ntfy alert failed: {exc}")
    log.info(f"ALERT: {message}")


def _load_target_weights(as_of: date) -> dict[str, float]:
    """Return most recent target weights on or before as_of."""
    if not TARGET_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No target weights found at {TARGET_WEIGHTS_PATH}. "
            "Run: uv run python orchestration/generate_signals.py"
        )
    df = pl.read_parquet(TARGET_WEIGHTS_PATH)
    available = df.filter(pl.col("date") <= as_of)
    if available.is_empty():
        raise ValueError(f"No target weights available on or before {as_of}")

    latest_date = available["date"].max()
    weights_df = available.filter(pl.col("date") == latest_date)
    weights = dict(zip(weights_df["symbol"].to_list(), weights_df["weight"].to_list()))
    log.info(f"Loaded target weights from {latest_date}: {list(weights.keys())}")
    return weights


def _build_broker(broker_name: str, ibkr_host: str | None = None,
                  ibkr_port: int | None = None, ibkr_client_id: int | None = None,
                  ibkr_is_paper: bool = True):
    """Construct the appropriate broker adapter."""
    if broker_name == "paper":
        from execution.paper_broker import PaperBroker
        return PaperBroker(initial_cash=100_000.0)

    elif broker_name == "ibkr":
        from config.settings import IBKR_CLIENT_ID, IBKR_HOST, IBKR_PAPER, IBKR_PORT
        from execution.ibkr_adapter import IBKRAdapter
        host = ibkr_host or IBKR_HOST or "127.0.0.1"
        port = ibkr_port or int(IBKR_PORT or 4002)
        cid  = ibkr_client_id if ibkr_client_id is not None else int(IBKR_CLIENT_ID or 1)
        is_paper = ibkr_is_paper
        mode = "paper" if is_paper else "LIVE"
        log.info(f"Building IBKR adapter: {host}:{port} client_id={cid} mode={mode}")
        return IBKRAdapter(host=host, port=port, client_id=cid, is_paper=is_paper)

    elif broker_name == "ccxt":
        from config.settings import KRAKEN_API_KEY, KRAKEN_API_SECRET
        from execution.ccxt_broker import CCXTBroker
        return CCXTBroker(
            exchange_id="kraken",
            api_key=KRAKEN_API_KEY or "",
            api_secret=KRAKEN_API_SECRET or "",
            is_paper=not bool(KRAKEN_API_KEY),
        )
    else:
        raise ValueError(f"Unknown broker: {broker_name!r}. Use 'paper', 'ibkr', or 'ccxt'.")


def _get_prices_from_storage(symbols: list[str], as_of: date) -> dict[str, float]:
    """Fetch latest close prices from bronze layer for the given symbols."""
    from storage.parquet_store import load_bars
    end = as_of
    start = end - timedelta(days=5)
    df = load_bars(symbols, start, end, "equity")
    if df.is_empty():
        return {}
    latest = df.group_by("symbol").agg(pl.last("adj_close").alias("price"))
    return dict(zip(latest["symbol"].to_list(), latest["price"].to_list()))


def run_rebalance(
    broker_name: str = "paper",
    dry_run: bool = False,
    as_of: date | None = None,
    ibkr_host: str | None = None,
    ibkr_port: int | None = None,
    ibkr_client_id: int | None = None,
    ibkr_is_paper: bool = True,
) -> int:
    check_halt()   # abort immediately if QP_HALT file exists

    today = as_of or date.today()
    mode = "paper" if ibkr_is_paper else "LIVE"

    # Semantic label written to parquet files — used by dashboard to filter
    # "paper" = simulated (paper broker or IBKR in paper mode)
    # "live"  = real money (IBKR live account)
    if broker_name == "ibkr":
        snap_broker = "paper" if ibkr_is_paper else "live"
    else:
        snap_broker = "paper"   # built-in paper broker and ccxt are always paper

    log.info(f"======== Rebalance | {today} | broker={broker_name} "
             f"| mode={mode} | dry_run={dry_run} ========")

    # 1. Load target weights
    try:
        target_weights = _load_target_weights(today)
    except Exception as exc:
        log.error(f"Failed to load target weights: {exc}")
        return 2

    # 2. Connect to broker
    broker = _build_broker(broker_name,
                          ibkr_host=ibkr_host, ibkr_port=ibkr_port,
                          ibkr_client_id=ibkr_client_id, ibkr_is_paper=ibkr_is_paper)
    try:
        if hasattr(broker, "connect"):
            broker.connect()
    except Exception as exc:
        log.error(f"Broker connection failed: {exc}")
        return 2

    try:
        # 3. Fetch positions + prices + NAV
        positions = broker.get_positions()
        cash = broker.get_cash()
        prices = _get_prices_from_storage(
            list(set(target_weights) | {p.symbol for p in positions}), today
        )

        if broker_name == "paper" and hasattr(broker, "set_prices"):
            broker.set_prices(prices)
        elif broker_name == "ccxt" and hasattr(broker, "set_paper_prices"):
            broker.set_paper_prices(prices)

        nav = nav_from_positions(positions, cash, prices)
        log.info(f"NAV: ${nav:,.0f} | Cash: ${cash:,.0f} | Positions: {len(positions)}")

        if nav <= 0:
            log.error("NAV is zero or negative — cannot rebalance")
            return 2

        # 4. Pre-trade risk check (hard gate)
        # top-5 concentration is 100% by definition for any portfolio with ≤5 positions;
        # only enforce it when there are enough positions for it to be meaningful.
        n_pos = len(target_weights)
        top5_cap = 1.0 if n_pos <= 5 else 0.80
        check = pre_trade_check(target_weights, RiskLimits(max_top5_concentration=top5_cap))
        if not check.passed:
            msg = f"Pre-trade check FAILED: {check.violations}"
            log.error(msg)
            _send_alert(f"[{today}] BLOCKED — {msg}")
            return 1

        # 5. Compute orders
        orders = compute_orders(target_weights, positions, prices, nav)
        log.info(orders_summary(orders, prices))

        if not orders:
            log.info("No orders needed — portfolio already at target")
            return 0

        # 6. Place orders — journal every attempt before and after the broker call
        rebalance_start = datetime.now(timezone.utc)
        if dry_run:
            log.info("DRY RUN — skipping order placement")
            for order in orders:
                append_order(
                    ORDER_JOURNAL_PATH, today, snap_broker,
                    order.symbol, order.qty,
                    est_price=prices.get(order.symbol, 0.0),
                    order_id="dry-run",
                    status="skipped",
                )
        else:
            placed = 0
            for order in orders:
                est_price = prices.get(order.symbol, 0.0)
                try:
                    order_id = broker.place_order(order)
                    append_order(
                        ORDER_JOURNAL_PATH, today, snap_broker,
                        order.symbol, order.qty, est_price, order_id, "placed",
                    )
                    log.info(f"Placed order {order_id}: {order.symbol} qty={order.qty:+.0f}")
                    placed += 1
                except Exception as exc:
                    append_order(
                        ORDER_JOURNAL_PATH, today, snap_broker,
                        order.symbol, order.qty, est_price, "failed", "failed",
                    )
                    log.error(f"Failed to place order for {order.symbol}: {exc}")
            log.info(f"Placed {placed}/{len(orders)} orders")

        # 7. Reconcile (after brief delay for fills to arrive)
        if not dry_run:
            log.info(f"Waiting {RECONCILE_DELAY}s for fills...")
            time.sleep(RECONCILE_DELAY)

            # Write actual fill prices back to the order journal
            try:
                fills = broker.get_fills(since=rebalance_start)
                for fill in fills:
                    update_order_fill(ORDER_JOURNAL_PATH, fill.order_id, fill.avg_price)
                if fills:
                    log.info(f"Recorded fill prices for {len(fills)} order(s)")
            except Exception as exc:
                log.warning(f"Could not write fill prices to journal: {exc}")

        broker_positions_after = broker.get_positions()

        # For paper broker, internal state IS broker state — build internal from orders
        if broker_name == "paper" or dry_run:
            internal_positions = broker_positions_after
        else:
            # Internal = what we expected after applying our orders
            from execution.base import Position as Pos
            qty_map = {p.symbol: p.qty for p in positions}
            for o in orders:
                qty_map[o.symbol] = qty_map.get(o.symbol, 0.0) + o.qty
            internal_positions = [
                Pos(symbol=s, qty=q, avg_cost=prices.get(s, 0.0))
                for s, q in qty_map.items()
                if abs(q) > 1e-9
            ]

        recon = reconcile(internal_positions, broker_positions_after, prices, nav)
        log.info(format_reconcile_report(recon))
        write_reconcile_log(recon, RECONCILE_LOG_PATH)

        # Persist NAV snapshot for the trading dashboards
        if not dry_run:
            try:
                from execution.trading_log import append_nav_snapshot
                TRADING_HISTORY_PATH = DATA_DIR / "gold" / "equity" / "trading_history.parquet"
                append_nav_snapshot(
                    TRADING_HISTORY_PATH, today, snap_broker,
                    nav=recon.nav,
                    cash=broker.get_cash() if hasattr(broker, "get_cash") else 0.0,
                    n_positions=len([p for p in broker_positions_after if abs(p.qty) > 1e-9]),
                )
            except Exception as exc:
                log.warning(f"Could not write NAV snapshot: {exc}")

        if has_material_drift(recon):
            _send_alert(f"[{today}] Material position drift detected! Check reconcile log.")
            return 1

    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()

    log.info("======== Rebalance complete ========")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily rebalance")
    parser.add_argument("--broker", choices=["paper", "ibkr", "ccxt"], default="paper")
    parser.add_argument("--dry-run", action="store_true", help="Compute orders but do not place")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    # IBKR-specific overrides (used by portfolio dashboard subprocess calls)
    parser.add_argument("--ibkr-host",      default=None, help="Override IBKR host")
    parser.add_argument("--ibkr-port",      type=int, default=None, help="Override IBKR port")
    parser.add_argument("--ibkr-client-id", type=int, default=None, help="Override IBKR client ID")
    parser.add_argument("--ibkr-live",      action="store_true",
                        help="Use live IBKR account (default: paper)")
    args = parser.parse_args()
    sys.exit(run_rebalance(
        broker_name=args.broker,
        dry_run=args.dry_run,
        as_of=args.date,
        ibkr_host=args.ibkr_host,
        ibkr_port=args.ibkr_port,
        ibkr_client_id=args.ibkr_client_id,
        ibkr_is_paper=not args.ibkr_live,
    ))


if __name__ == "__main__":
    main()
