"""In-process paper broker for testing the execution loop without a live connection.

Fills orders at the last known close price. Does not simulate slippage or
partial fills — use this for smoke-testing order flow, not realistic simulation.
"""

import uuid
from datetime import datetime

import polars as pl

from .base import BrokerAdapter, Fill, Order, Position


class PaperBroker:
    """Simple in-memory paper broker. Fills instantly at provided prices."""

    name = "paper"
    is_paper = True

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._prices: dict[str, float] = {}

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update mark-to-market prices (call before each rebalance cycle)."""
        self._prices = prices

    def get_positions(self) -> list[Position]:
        for pos in self._positions.values():
            if pos.symbol in self._prices:
                pos.market_value = pos.qty * self._prices[pos.symbol]
                pos.unrealized_pnl = pos.market_value - pos.qty * pos.avg_cost
        return list(self._positions.values())

    def get_cash(self) -> float:
        return self._cash

    def place_order(self, order: Order) -> str:
        price = self._prices.get(order.symbol)
        if price is None:
            raise ValueError(f"PaperBroker: no price for {order.symbol}")

        order_id = str(uuid.uuid4())[:8]
        cost = order.qty * price
        commission = 0.0   # IBKR Lite on ETFs = $0

        self._cash -= cost + commission

        existing = self._positions.get(order.symbol)
        if existing:
            new_qty = existing.qty + order.qty
            if abs(new_qty) < 1e-9:
                del self._positions[order.symbol]
            else:
                total_cost = existing.qty * existing.avg_cost + order.qty * price
                existing.qty = new_qty
                existing.avg_cost = total_cost / new_qty
        else:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=order.qty,
                avg_cost=price,
                asset_class=order.asset_class,
            )

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            filled_qty=order.qty,
            avg_price=price,
            commission=commission,
            filled_at=datetime.utcnow(),
            asset_class=order.asset_class,
        )
        self._fills.append(fill)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        return False   # paper broker fills instantly

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f.filled_at >= since]
