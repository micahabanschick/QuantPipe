"""Risk management module — Phase 4.

Core functions:
    compute_exposures(positions, prices) -> ExposureReport
    historical_var(returns, weights, confidence=0.95, lookback=252) -> float
    pre_trade_check(proposed_orders, current_positions, limits) -> CheckResult

Pre-trade checks are hard blocks — they reject orders, not advisory warnings.
"""
