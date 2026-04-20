"""The 5 canonical features for Phase 2.

Every function is a pure transformation of point-in-time data.
No lookahead: all windows are backward-looking, no shift(-1) shortcuts.

Snapshot tests in tests/test_features.py pin the output to a known CSV
so accidental lookahead contamination is immediately caught.
"""

from datetime import date

import polars as pl


def log_return(prices: pl.Series, periods: int = 1) -> pl.Series:
    """Daily log return: ln(P_t / P_{t-n}).

    NaN for the first `periods` rows (insufficient history).
    """
    shifted = prices.shift(periods)
    return (prices / shifted).log()


def realized_vol(prices: pl.Series, window: int = 21) -> pl.Series:
    """Rolling annualized realized volatility using daily log returns.

    Uses a `window`-day rolling standard deviation, annualized by sqrt(252).
    """
    rets = log_return(prices)
    return rets.rolling_std(window_size=window) * (252 ** 0.5)


def momentum_12m_1m(prices: pl.Series) -> pl.Series:
    """Cross-sectional 12-1 momentum: return from 252 days ago to 21 days ago.

    Excludes the most recent month to avoid short-term reversal contamination.
    Standard academic definition: Jegadeesh & Titman (1993).
    """
    p_252 = prices.shift(252)
    p_21 = prices.shift(21)
    return (p_21 / p_252) - 1


def dollar_volume(close: pl.Series, volume: pl.Series, window: int = 63) -> pl.Series:
    """Rolling average dollar volume over `window` days.

    Used as a liquidity filter — exclude bottom decile before ranking signals.
    """
    dv = close * volume
    return dv.rolling_mean(window_size=window)


def reversal_5d(prices: pl.Series) -> pl.Series:
    """Short-term reversal: negative 5-day return (sign flip = reversal signal).

    A positive reversal signal = recent losers (expected to mean-revert up).
    """
    return -((prices / prices.shift(5)) - 1)


def compute_features(
    bars: pl.DataFrame,
    feature_list: list[str] | None = None,
) -> pl.DataFrame:
    """Compute all canonical features for a DataFrame of OHLCV bars.

    Input must have columns: [date, symbol, adj_close, volume].
    Returns a wide DataFrame: [date, symbol, feature1, feature2, ...].

    If feature_list is None, computes all 5 canonical features.
    """
    available = {
        "log_return_1d",
        "realized_vol_21d",
        "momentum_12m_1m",
        "dollar_volume_63d",
        "reversal_5d",
    }
    if feature_list is None:
        feature_list = sorted(available)
    unknown = set(feature_list) - available
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Available: {available}")

    result_frames = []

    for symbol, group in bars.group_by("symbol"):
        group = group.sort("date")
        prices = group["adj_close"]
        volume = group["volume"]

        feature_cols: dict[str, pl.Series] = {}

        if "log_return_1d" in feature_list:
            feature_cols["log_return_1d"] = log_return(prices, 1)
        if "realized_vol_21d" in feature_list:
            feature_cols["realized_vol_21d"] = realized_vol(prices, 21)
        if "momentum_12m_1m" in feature_list:
            feature_cols["momentum_12m_1m"] = momentum_12m_1m(prices)
        if "dollar_volume_63d" in feature_list:
            feature_cols["dollar_volume_63d"] = dollar_volume(prices, volume, 63)
        if "reversal_5d" in feature_list:
            feature_cols["reversal_5d"] = reversal_5d(prices)

        out = group.select(["date", "symbol"]).with_columns([
            pl.Series(name=k, values=v) for k, v in feature_cols.items()
        ])
        result_frames.append(out)

    if not result_frames:
        return pl.DataFrame()

    return pl.concat(result_frames).sort(["date", "symbol"])
