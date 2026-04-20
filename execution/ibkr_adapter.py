"""IBKR broker adapter — wraps ib_insync for equities/ETFs/futures.

Implements the BrokerAdapter protocol. Connects to TWS or IB Gateway
running locally (or forwarded via SSH).

Prerequisites:
    uv sync --extra execution
    TWS / IB Gateway running with API enabled on the configured host:port.

Usage:
    from execution.ibkr_adapter import IBKRAdapter
    broker = IBKRAdapter(host="127.0.0.1", port=7497, client_id=1, is_paper=True)
    broker.connect()
    positions = broker.get_positions()
    broker.disconnect()

    Or use as a context manager:
    with IBKRAdapter(...) as broker:
        orders = compute_orders(target_weights, broker.get_positions(), ...)
        for order in orders:
            broker.place_order(order)

TWS paper trading port:  7497
TWS live trading port:   7496
IB Gateway paper port:   4002
IB Gateway live port:    4001
"""

import logging
import time
import uuid
from datetime import datetime

from .base import BrokerAdapter, Fill, Order, Position

log = logging.getLogger(__name__)

# Seconds to wait for order status before timing out
_ORDER_TIMEOUT = 30


class IBKRAdapter:
    """Synchronous wrapper around ib_insync for the QuantPipe execution loop."""

    name = "ibkr"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        is_paper: bool = True,
        readonly: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.is_paper = is_paper
        self.readonly = readonly
        self._ib = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError(
                "ib_insync is not installed. Run: uv sync --extra execution"
            )
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, readonly=self.readonly)
        log.info(f"Connected to IBKR at {self.host}:{self.port} (paper={self.is_paper})")

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IBKR")

    def __enter__(self) -> "IBKRAdapter":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError(
                "Not connected to IBKR. Call connect() or use as context manager."
            )

    # ── BrokerAdapter protocol ────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        self._require_connected()
        raw = self._ib.positions()
        positions = []
        for p in raw:
            contract = p.contract
            symbol = contract.localSymbol or contract.symbol
            # Request market data for mark-to-market
            try:
                ticker = self._ib.reqMktData(contract, "", True, False)
                self._ib.sleep(0.5)
                price = ticker.last or ticker.close or p.avgCost
            except Exception:
                price = p.avgCost
            mkt_val = p.position * price
            positions.append(Position(
                symbol=symbol,
                qty=float(p.position),
                avg_cost=float(p.avgCost),
                market_value=float(mkt_val),
                unrealized_pnl=float(mkt_val - p.position * p.avgCost),
                asset_class="equity",
            ))
        return positions

    def get_cash(self) -> float:
        self._require_connected()
        account_values = self._ib.accountValues()
        for av in account_values:
            if av.tag == "CashBalance" and av.currency == "USD":
                return float(av.value)
        # Fallback: TotalCashValue
        for av in account_values:
            if av.tag == "TotalCashValue" and av.currency == "USD":
                return float(av.value)
        return 0.0

    def place_order(self, order: Order) -> str:
        self._require_connected()
        if self.readonly:
            raise PermissionError("IBKRAdapter is in read-only mode")

        from ib_insync import MarketOrder, Stock, LimitOrder

        contract = Stock(order.symbol, "SMART", "USD")
        action = "BUY" if order.qty > 0 else "SELL"
        qty = abs(order.qty)

        if order.order_type == "LMT" and order.limit_price is not None:
            ib_order = LimitOrder(action, qty, order.limit_price)
        else:
            ib_order = MarketOrder(action, qty)

        # Attach client-side ref for idempotency checks
        ref = order.client_order_id or str(uuid.uuid4())[:8]
        ib_order.orderRef = ref

        trade = self._ib.placeOrder(contract, ib_order)
        log.info(f"Placed {action} {qty} {order.symbol} — orderRef={ref} orderId={trade.order.orderId}")

        # Wait for fill (paper fills almost instantly)
        deadline = time.monotonic() + _ORDER_TIMEOUT
        while time.monotonic() < deadline:
            self._ib.sleep(0.5)
            if trade.isDone():
                break

        if not trade.isDone():
            log.warning(f"Order {ref} did not fill within {_ORDER_TIMEOUT}s — leaving open")

        return str(trade.order.orderId)

    def cancel_order(self, order_id: str) -> bool:
        self._require_connected()
        try:
            oid = int(order_id)
        except ValueError:
            return False
        open_trades = self._ib.openTrades()
        for trade in open_trades:
            if trade.order.orderId == oid:
                self._ib.cancelOrder(trade.order)
                log.info(f"Cancelled order {order_id}")
                return True
        return False

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        self._require_connected()
        fills = []
        for trade in self._ib.trades():
            for exec_ in trade.fills:
                filled_at = exec_.time
                if isinstance(filled_at, str):
                    filled_at = datetime.fromisoformat(filled_at.replace(" ", "T"))
                if since and filled_at < since:
                    continue
                fills.append(Fill(
                    order_id=str(trade.order.orderId),
                    symbol=trade.contract.symbol,
                    filled_qty=float(exec_.execution.shares),
                    avg_price=float(exec_.execution.price),
                    commission=float(exec_.commissionReport.commission
                                     if exec_.commissionReport else 0.0),
                    filled_at=filled_at,
                    asset_class="equity",
                ))
        return fills
