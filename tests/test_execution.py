"""Tests for execution layer — trader, paper broker, reconciler."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from execution.base import Fill, Order, Position
from execution.paper_broker import PaperBroker
from execution.reconciler import (
    ReconcileReport,
    format_reconcile_report,
    has_material_drift,
    reconcile,
    write_reconcile_log,
)
from execution.trader import compute_orders, nav_from_positions, orders_summary


# ── nav_from_positions ────────────────────────────────────────────────────────

class TestNavFromPositions:
    def test_cash_only(self):
        nav = nav_from_positions([], 100_000.0, {})
        assert nav == 100_000.0

    def test_positions_use_prices(self):
        pos = [Position(symbol="SPY", qty=100, avg_cost=400.0)]
        nav = nav_from_positions(pos, 50_000.0, {"SPY": 450.0})
        assert abs(nav - 95_000.0) < 1e-6

    def test_falls_back_to_avg_cost_when_no_price(self):
        pos = [Position(symbol="QQQ", qty=50, avg_cost=350.0)]
        nav = nav_from_positions(pos, 0.0, {})
        assert abs(nav - 17_500.0) < 1e-6


# ── compute_orders ────────────────────────────────────────────────────────────

class TestComputeOrders:
    def _prices(self):
        return {"SPY": 450.0, "QQQ": 380.0, "IWM": 200.0, "XLK": 175.0}

    def test_no_orders_when_already_at_target(self):
        prices = self._prices()
        # NAV = $100k, target 50% SPY => 111 shares @ $450 = $49,950
        nav = 100_000.0
        positions = [Position("SPY", qty=111, avg_cost=450.0)]
        target = {"SPY": 0.50, "QQQ": 0.50}
        # Need to also have QQQ for true match
        positions = [
            Position("SPY", qty=111, avg_cost=450.0),
            Position("QQQ", qty=131, avg_cost=380.0),
        ]
        orders = compute_orders(target, positions, prices, nav)
        # May have tiny residual orders but notional should be below min_trade_pct
        for o in orders:
            assert abs(o.qty * prices.get(o.symbol, 0)) < 0.005 * nav + 1

    def test_empty_positions_buys_target(self):
        prices = {"SPY": 450.0, "QQQ": 380.0}
        nav = 100_000.0
        target = {"SPY": 0.50, "QQQ": 0.50}
        orders = compute_orders(target, [], prices, nav)
        assert len(orders) == 2
        spy_order = next(o for o in orders if o.symbol == "SPY")
        assert spy_order.qty > 0
        assert spy_order.qty == math.floor(50_000.0 / 450.0)

    def test_closes_positions_not_in_target(self):
        prices = {"SPY": 450.0, "OLD_ETF": 100.0}
        nav = 100_000.0
        target = {"SPY": 1.0}
        positions = [Position("OLD_ETF", qty=50, avg_cost=100.0)]
        orders = compute_orders(target, positions, prices, nav)
        close_order = next((o for o in orders if o.symbol == "OLD_ETF"), None)
        assert close_order is not None
        assert close_order.qty < 0    # sell

    def test_dust_orders_skipped(self):
        prices = {"SPY": 450.0}
        nav = 100_000.0
        target = {"SPY": 0.50}
        # Current position is already almost exactly right
        positions = [Position("SPY", qty=111, avg_cost=450.0)]
        orders = compute_orders(target, positions, prices, nav, min_trade_pct=0.005)
        for o in orders:
            assert abs(o.qty * prices[o.symbol]) >= 0.005 * nav - 1

    def test_zero_nav_returns_no_orders(self):
        orders = compute_orders({"SPY": 1.0}, [], {"SPY": 450.0}, 0.0)
        assert orders == []

    def test_missing_price_symbol_skipped(self):
        orders = compute_orders({"NOPRICE": 1.0}, [], {}, 100_000.0)
        assert all(o.symbol != "NOPRICE" for o in orders)

    def test_orders_are_integer_shares(self):
        prices = {"SPY": 450.0, "QQQ": 380.0, "IWM": 200.0}
        target = {"SPY": 0.40, "QQQ": 0.35, "IWM": 0.25}
        orders = compute_orders(target, [], prices, 100_000.0)
        for o in orders:
            assert o.qty == int(o.qty)

    def test_idempotent(self):
        prices = {"SPY": 450.0, "QQQ": 380.0}
        target = {"SPY": 0.50, "QQQ": 0.50}
        o1 = compute_orders(target, [], prices, 100_000.0)
        o2 = compute_orders(target, [], prices, 100_000.0)
        assert [(o.symbol, o.qty) for o in o1] == [(o.symbol, o.qty) for o in o2]


# ── PaperBroker ───────────────────────────────────────────────────────────────

class TestPaperBroker:
    def _broker(self):
        b = PaperBroker(initial_cash=100_000.0)
        b.set_prices({"SPY": 450.0, "QQQ": 380.0})
        return b

    def test_initial_cash(self):
        b = PaperBroker(100_000.0)
        assert b.get_cash() == 100_000.0

    def test_buy_reduces_cash(self):
        b = self._broker()
        b.place_order(Order(symbol="SPY", qty=10))
        assert b.get_cash() == 100_000.0 - 10 * 450.0

    def test_buy_creates_position(self):
        b = self._broker()
        b.place_order(Order(symbol="SPY", qty=10))
        positions = b.get_positions()
        spy = next(p for p in positions if p.symbol == "SPY")
        assert spy.qty == 10

    def test_sell_closes_position(self):
        b = self._broker()
        b.place_order(Order(symbol="SPY", qty=10))
        b.place_order(Order(symbol="SPY", qty=-10))
        positions = b.get_positions()
        assert all(p.symbol != "SPY" for p in positions)

    def test_no_price_raises(self):
        b = PaperBroker()
        with pytest.raises(ValueError):
            b.place_order(Order(symbol="NOPRICE", qty=1))

    def test_fill_recorded(self):
        b = self._broker()
        b.place_order(Order(symbol="SPY", qty=5))
        fills = b.get_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "SPY"
        assert fills[0].filled_qty == 5

    def test_fills_filtered_by_since(self):
        b = self._broker()
        b.place_order(Order(symbol="SPY", qty=5))
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        fills = b.get_fills(since=future)
        assert len(fills) == 0

    def test_cancel_always_false(self):
        b = self._broker()
        assert b.cancel_order("any_id") is False

    def test_market_value_updated_with_new_price(self):
        b = PaperBroker(100_000.0)
        b.set_prices({"SPY": 400.0})
        b.place_order(Order(symbol="SPY", qty=10))
        b.set_prices({"SPY": 450.0})
        positions = b.get_positions()
        spy = next(p for p in positions if p.symbol == "SPY")
        assert abs(spy.market_value - 4500.0) < 1e-6


# ── Reconciler ────────────────────────────────────────────────────────────────

class TestReconcile:
    def _pos(self, symbol: str, qty: float) -> Position:
        return Position(symbol=symbol, qty=qty, avg_cost=100.0)

    def test_clean_when_identical(self):
        internal = [self._pos("SPY", 100), self._pos("QQQ", 50)]
        broker = [self._pos("SPY", 100), self._pos("QQQ", 50)]
        report = reconcile(internal, broker, {"SPY": 450.0, "QQQ": 380.0}, 100_000.0)
        assert report.is_clean

    def test_detects_qty_drift(self):
        internal = [self._pos("SPY", 100)]
        broker = [self._pos("SPY", 95)]
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert not report.is_clean
        assert len(report.drifts) == 1
        assert report.drifts[0].qty_diff == -5

    def test_detects_missing_from_broker(self):
        internal = [self._pos("SPY", 100)]
        broker = []
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert "SPY" in report.symbols_only_internal

    def test_detects_extra_in_broker(self):
        internal = []
        broker = [self._pos("SURPRISE", 50)]
        report = reconcile(internal, broker, {"SURPRISE": 200.0}, 100_000.0)
        assert "SURPRISE" in report.symbols_only_broker

    def test_total_notional_drift(self):
        internal = [self._pos("SPY", 100)]
        broker = [self._pos("SPY", 95)]
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert abs(report.total_notional_drift - 5 * 450.0) < 0.01


class TestHasMaterialDrift:
    def _pos(self, symbol, qty):
        return Position(symbol=symbol, qty=qty, avg_cost=100.0)

    def test_clean_is_not_material(self):
        internal = [self._pos("SPY", 100)]
        broker = [self._pos("SPY", 100)]
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert not has_material_drift(report)

    def test_large_drift_is_material(self):
        internal = [self._pos("SPY", 100)]
        broker = [self._pos("SPY", 90)]  # 10% drift
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert has_material_drift(report, threshold_pct=5.0)

    def test_small_drift_not_material(self):
        internal = [self._pos("SPY", 100)]
        broker = [self._pos("SPY", 99)]  # 1% drift
        report = reconcile(internal, broker, {"SPY": 450.0}, 100_000.0)
        assert not has_material_drift(report, threshold_pct=5.0)

    def test_extra_broker_symbol_is_material(self):
        internal = []
        broker = [self._pos("ROGUE", 10)]
        report = reconcile(internal, broker, {"ROGUE": 100.0}, 100_000.0)
        assert has_material_drift(report)


class TestWriteReconcileLog:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "reconcile_log.parquet"
        pos = [Position("SPY", 100, 450.0)]
        report = reconcile(pos, pos, {"SPY": 450.0}, 100_000.0)
        write_reconcile_log(report, path)
        assert path.exists()

    def test_appends_new_date(self, tmp_path):
        import polars as pl
        path = tmp_path / "log.parquet"
        pos = [Position("SPY", 100, 450.0)]

        r1 = reconcile(pos, pos, {"SPY": 450.0}, 100_000.0)
        r1.as_of = datetime(2024, 1, 15, 10, 0, 0)
        write_reconcile_log(r1, path)

        r2 = reconcile(pos, pos, {"SPY": 450.0}, 100_000.0)
        r2.as_of = datetime(2024, 1, 16, 10, 0, 0)
        write_reconcile_log(r2, path)

        loaded = pl.read_parquet(path)
        assert loaded["as_of"].n_unique() == 2


# ── orders_summary ────────────────────────────────────────────────────────────

class TestOrdersSummary:
    def test_empty_orders(self):
        result = orders_summary([], {})
        assert "No orders" in result

    def test_shows_buys_and_sells(self):
        orders = [
            Order(symbol="SPY", qty=10),
            Order(symbol="QQQ", qty=-5),
        ]
        prices = {"SPY": 450.0, "QQQ": 380.0}
        result = orders_summary(orders, prices)
        assert "BUY" in result
        assert "SELL" in result
        assert "SPY" in result
        assert "QQQ" in result
