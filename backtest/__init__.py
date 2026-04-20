"""Backtesting module — VectorBT integration and strategy wrappers.

Phase 3 work. Strategy interface:
    run_backtest(features, params, cost_model) -> BacktestResult

BacktestResult contains: equity curve, positions over time, tearsheet metrics.
"""
