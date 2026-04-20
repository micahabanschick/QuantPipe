"""CCXT broker adapter — routes live crypto orders via any CCXT-supported exchange.

Implements the BrokerAdapter protocol. Distinct from data_adapters/ccxt_adapter.py
(which only fetches OHLCV bars). This adapter handles actual order placement,
position queries, and fill history.

Prerequisites:
    API keys set in .env: KRAKEN_API_KEY, KRAKEN_API_SECRET (or exchange-specific vars)

Usage:
    from execution.ccxt_broker import CCXTBroker
    broker = CCXTBroker(exchange_id="kraken", is_paper=True)
    positions = broker.get_positions()
    order_id = broker.place_order(Order(symbol="BTC/USDT", qty=0.001))

Paper mode:
    When is_paper=True, order placement is simulated locally (no API calls made).
    Set is_paper=False and provide valid API keys for live trading.

Symbol convention:
    Use CCXT format: "BTC/USDT", "ETH/USDT", etc.
"""

import logging
import uuid
from datetime import datetime, timezone

from .base import Fill, Order, Position

log = logging.getLogger(__name__)


class CCXTBroker:
    """CCXT-backed broker adapter for crypto execution."""

    name: str
    is_paper: bool

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: str = "",
        api_secret: str = "",
        is_paper: bool = True,
    ) -> None:
        self.exchange_id = exchange_id
        self.is_paper = is_paper
        self.name = f"ccxt_{exchange_id}"
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange = None
        # Paper mode state
        self._paper_cash: float = 100_000.0
        self._paper_positions: dict[str, Position] = {}
        self._paper_prices: dict[str, float] = {}
        self._paper_fills: list[Fill] = []

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.is_paper:
            log.info(f"CCXTBroker [{self.exchange_id}] running in paper mode")
            return
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt is not installed. Run: uv sync --extra execution")
        cls = getattr(ccxt, self.exchange_id)
        self._exchange = cls({
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
        })
        log.info(f"CCXTBroker connected to {self.exchange_id} (live)")

    def set_paper_prices(self, prices: dict[str, float]) -> None:
        """Update mark-to-market prices for paper mode."""
        self._paper_prices = prices

    # ── BrokerAdapter protocol ────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        if self.is_paper:
            return list(self._paper_positions.values())
        self._require_exchange()
        try:
            balance = self._exchange.fetch_balance()
        except Exception as exc:
            log.error(f"[{self.exchange_id}] fetch_balance failed: {exc}")
            return []

        positions = []
        for currency, info in balance["total"].items():
            if info > 0 and currency != "USD" and currency != "USDT":
                symbol = f"{currency}/USDT"
                price = self._fetch_price(symbol)
                mkt_val = info * price if price else 0.0
                positions.append(Position(
                    symbol=symbol,
                    qty=float(info),
                    avg_cost=price or 0.0,
                    market_value=mkt_val,
                    asset_class="crypto",
                ))
        return positions

    def get_cash(self) -> float:
        if self.is_paper:
            return self._paper_cash
        self._require_exchange()
        try:
            balance = self._exchange.fetch_balance()
            return float(balance["total"].get("USDT", 0.0))
        except Exception as exc:
            log.error(f"[{self.exchange_id}] fetch_balance for cash failed: {exc}")
            return 0.0

    def place_order(self, order: Order) -> str:
        if self.is_paper:
            return self._paper_place_order(order)
        self._require_exchange()

        action = "buy" if order.qty > 0 else "sell"
        qty = abs(order.qty)

        try:
            if order.order_type == "LMT" and order.limit_price is not None:
                result = self._exchange.create_limit_order(
                    order.symbol, action, qty, order.limit_price
                )
            else:
                result = self._exchange.create_market_order(order.symbol, action, qty)

            order_id = str(result.get("id", uuid.uuid4().hex[:8]))
            log.info(f"[{self.exchange_id}] {action.upper()} {qty} {order.symbol} => id={order_id}")
            return order_id
        except Exception as exc:
            log.error(f"[{self.exchange_id}] place_order failed: {exc}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        if self.is_paper:
            return False   # paper orders fill instantly
        self._require_exchange()
        try:
            self._exchange.cancel_order(order_id)
            return True
        except Exception as exc:
            log.warning(f"[{self.exchange_id}] cancel_order {order_id} failed: {exc}")
            return False

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        if self.is_paper:
            if since is None:
                return list(self._paper_fills)
            return [f for f in self._paper_fills if f.filled_at >= since]
        self._require_exchange()
        try:
            since_ms = int(since.timestamp() * 1000) if since else None
            raw_trades = self._exchange.fetch_my_trades(since=since_ms, limit=500)
        except Exception as exc:
            log.error(f"[{self.exchange_id}] fetch_my_trades failed: {exc}")
            return []

        fills = []
        for trade in raw_trades:
            fills.append(Fill(
                order_id=str(trade.get("order", "")),
                symbol=trade.get("symbol", ""),
                filled_qty=float(trade.get("amount", 0)),
                avg_price=float(trade.get("price", 0)),
                commission=float(trade.get("fee", {}).get("cost", 0)),
                filled_at=datetime.utcfromtimestamp(trade["timestamp"] / 1000),
                asset_class="crypto",
            ))
        return fills

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_exchange(self) -> None:
        if self._exchange is None:
            raise ConnectionError(
                f"CCXTBroker [{self.exchange_id}] not connected. Call connect() first."
            )

    def _fetch_price(self, symbol: str) -> float | None:
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close") or 0)
        except Exception:
            return None

    def _paper_place_order(self, order: Order) -> str:
        price = self._paper_prices.get(order.symbol)
        if price is None or price <= 0:
            raise ValueError(f"CCXTBroker paper mode: no price for {order.symbol}")

        order_id = uuid.uuid4().hex[:8]
        cost = order.qty * price
        self._paper_cash -= cost

        existing = self._paper_positions.get(order.symbol)
        if existing:
            new_qty = existing.qty + order.qty
            if abs(new_qty) < 1e-9:
                del self._paper_positions[order.symbol]
            else:
                total_cost = existing.qty * existing.avg_cost + order.qty * price
                existing.qty = new_qty
                existing.avg_cost = total_cost / new_qty
        else:
            self._paper_positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=order.qty,
                avg_cost=price,
                market_value=order.qty * price,
                asset_class="crypto",
            )

        self._paper_fills.append(Fill(
            order_id=order_id,
            symbol=order.symbol,
            filled_qty=order.qty,
            avg_price=price,
            commission=0.0,
            filled_at=datetime.now(timezone.utc),
            asset_class="crypto",
        ))
        log.info(f"[paper-{self.exchange_id}] {'BUY' if order.qty > 0 else 'SELL'} "
                 f"{abs(order.qty)} {order.symbol} @ {price:.4f}")
        return order_id
