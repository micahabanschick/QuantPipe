"""Order computation — pure function: target weights + positions + prices → orders.

No I/O, no broker calls. Idempotent: the same inputs always produce the same orders.
This makes testing trivial and means a rebalance script can safely re-run.

Public API:
    compute_orders(target_weights, positions, prices, nav, min_trade_pct) -> list[Order]
    nav_from_positions(positions, cash, prices) -> float
"""

import math
from dataclasses import dataclass

from .base import Order, Position


def nav_from_positions(
    positions: list[Position],
    cash: float,
    prices: dict[str, float],
) -> float:
    """Compute NAV = cash + sum(market_value of all positions).

    Uses prices dict for mark-to-market if position.market_value is stale.
    """
    equity = sum(
        pos.qty * prices.get(pos.symbol, pos.avg_cost)
        for pos in positions
    )
    return cash + equity


def compute_orders(
    target_weights: dict[str, float],
    positions: list[Position],
    prices: dict[str, float],
    nav: float,
    min_trade_pct: float = 0.005,
    asset_class: str = "equity",
) -> list[Order]:
    """Compute the minimal set of orders to move from current to target weights.

    Parameters
    ----------
    target_weights : {symbol: weight} — desired portfolio weights (sum ≈ 1.0)
    positions      : Current broker positions
    prices         : {symbol: price} — required for all symbols in target + positions
    nav            : Net asset value used to convert weights to share quantities
    min_trade_pct  : Skip orders whose notional is < min_trade_pct × NAV (avoids dust)
    asset_class    : Passed through to Order.asset_class

    Returns
    -------
    List of Order objects (positive qty = buy, negative = sell).
    Returns an empty list if already at target (idempotent).

    Notes
    -----
    - Quantities are rounded to integer shares (fractional shares not supported).
    - Symbols in current positions but absent from target_weights are closed (weight=0).
    - The caller must run a pre-trade check before submitting the returned orders.
    """
    if nav <= 0:
        return []

    # Build current qty map
    current_qty: dict[str, float] = {p.symbol: p.qty for p in positions}

    # Merge all symbols: targets + any open positions not in target (must close)
    all_symbols: set[str] = set(target_weights) | set(current_qty)

    min_notional = min_trade_pct * nav
    orders: list[Order] = []

    for sym in sorted(all_symbols):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue

        target_weight = target_weights.get(sym, 0.0)
        target_notional = nav * target_weight
        target_qty = math.floor(target_notional / price)  # integer shares, no fractions

        current = current_qty.get(sym, 0.0)
        delta = target_qty - current

        # Skip dust trades
        if abs(delta * price) < min_notional:
            continue

        orders.append(Order(
            symbol=sym,
            qty=delta,
            order_type="MKT",
            asset_class=asset_class,
        ))

    return orders


def orders_summary(orders: list[Order], prices: dict[str, float]) -> str:
    """Return a compact human-readable summary of a list of orders."""
    if not orders:
        return "No orders (already at target)"
    buys = [o for o in orders if o.qty > 0]
    sells = [o for o in orders if o.qty < 0]
    buy_notional = sum(o.qty * prices.get(o.symbol, 0) for o in buys)
    sell_notional = sum(abs(o.qty) * prices.get(o.symbol, 0) for o in sells)
    lines = [
        f"Orders: {len(orders)} total ({len(buys)} buys, {len(sells)} sells)",
        f"  Buy notional : ${buy_notional:,.0f}",
        f"  Sell notional: ${sell_notional:,.0f}",
    ]
    for o in orders:
        price = prices.get(o.symbol, 0)
        notional = o.qty * price
        side = "BUY " if o.qty > 0 else "SELL"
        lines.append(f"  {side}  {abs(o.qty):6.0f}  {o.symbol:<8}  @ ${price:.2f}  = ${abs(notional):>10,.0f}")
    return "\n".join(lines)
