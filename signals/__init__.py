"""Signals module — combines features into cross-sectional ranking signals.

Phase 3+ work. Each signal is a pure function:
    (features_df: pl.DataFrame, params: dict) -> pl.DataFrame

The output DataFrame must have columns: [date, symbol, signal_value].
"""
