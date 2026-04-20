"""Broker adapter protocol and data classes — Phase 6.

All broker adapters (IBKR, CCXT, paper) implement BrokerAdapter.
The Trader module depends only on this protocol so live vs paper is a flag,
not a code branch.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class Order:
    symbol: str
    qty: float            # positive = buy, negative = sell
    order_type: str = "MKT"
    limit_price: float | None = None
    client_order_id: str = ""
    asset_class: str = "equity"


@dataclass
class Fill:
    order_id: str
    symbol: str
    filled_qty: float
    avg_price: float
    commission: float
    filled_at: datetime
    asset_class: str = "equity"


@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    asset_class: str = "equity"


@runtime_checkable
class BrokerAdapter(Protocol):
    """Protocol every broker adapter must satisfy."""

    name: str
    is_paper: bool

    def get_positions(self) -> list[Position]: ...
    def get_cash(self) -> float: ...
    def place_order(self, order: Order) -> str: ...       # returns order_id
    def cancel_order(self, order_id: str) -> bool: ...
    def get_fills(self, since: datetime | None = None) -> list[Fill]: ...
