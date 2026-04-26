"""Long-memory analytics -- Hurst exponent estimation.

Three estimators (all pure numpy, no Streamlit, no Plotly):
  hurst_rs()      -- classic R/S (Hurst & Mandelbrot)
  hurst_dfa()     -- Detrended Fluctuation Analysis (Peng et al. 1994)
  hurst_label()   -- human-readable regime label
  rolling_hurst() -- rolling window Hurst over a time series

H > 0.55 : persistent / trending
H ~ 0.50 : random walk (Brownian motion)
H < 0.45 : mean-reverting / anti-persistent
"""

from __future__ import annotations

import numpy as np


def hurst_rs(
    x: np.ndarray,
    min_window: int = 8,
    max_window: int | None = None,
    n_points: int = 16,
) -> float:
    """Hurst exponent via R/S (rescaled range) analysis.

    Parameters
    ----------
    x          : 1-D array of observations (log-prices, factor values, returns)
    min_window : smallest window size tested
    max_window : largest window size; defaults to len(x) // 2
    n_points   : number of log-spaced window sizes

    Returns
    -------
    H in [0, 1], or nan if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * min_window:
        return float("nan")

    max_w = min(max_window or n // 2, n // 2)
    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max(max_w, min_window)), n_points).astype(int)
    )

    log_n, log_rs = [], []
    for w in windows:
        n_blocks = n // w
        if n_blocks < 1:
            continue
        rs_vals = []
        for b in range(n_blocks):
            chunk = x[b * w: (b + 1) * w]
            mu    = chunk.mean()
            dev   = np.cumsum(chunk - mu)
            r     = dev.max() - dev.min()
            s     = chunk.std(ddof=1)
            if s > 1e-12:
                rs_vals.append(r / s)
        if rs_vals:
            log_n.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_vals)))

    if len(log_n) < 2:
        return float("nan")

    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def hurst_dfa(
    x: np.ndarray,
    min_window: int = 8,
    max_window: int | None = None,
    n_points: int = 16,
    order: int = 1,
) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis.

    Parameters
    ----------
    x          : 1-D array (returns or factor values)
    min_window : smallest segment size
    max_window : largest segment size; defaults to len(x) // 4
    n_points   : number of log-spaced segment sizes
    order      : polynomial detrending order (1=linear, 2=quadratic)

    Returns
    -------
    H in [0, 1], or nan if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * min_window:
        return float("nan")

    y = np.cumsum(x - x.mean())

    max_w = min(max_window or n // 4, n // 4)
    if max_w < min_window:
        max_w = min_window
    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_w), n_points).astype(int)
    )

    log_n, log_f = [], []
    for w in windows:
        n_segs = n // w
        if n_segs < 1:
            continue
        flucts = []
        for seg in range(n_segs):
            seg_y  = y[seg * w: (seg + 1) * w]
            t      = np.arange(len(seg_y))
            coeffs = np.polyfit(t, seg_y, order)
            trend  = np.polyval(coeffs, t)
            flucts.append(np.mean((seg_y - trend) ** 2))
        f = np.sqrt(np.mean(flucts))
        if f > 1e-12:
            log_n.append(np.log(w))
            log_f.append(np.log(f))

    if len(log_n) < 2:
        return float("nan")

    slope, _ = np.polyfit(log_n, log_f, 1)
    return float(np.clip(slope, 0.0, 1.0))


def hurst_label(h: float) -> str:
    """Human-readable regime label for a Hurst exponent."""
    if not np.isfinite(h):
        return "Unknown"
    if h > 0.65:
        return "Strongly Persistent"
    if h > 0.55:
        return "Persistent"
    if h >= 0.45:
        return "Random Walk"
    if h >= 0.35:
        return "Mean-Reverting"
    return "Strongly Mean-Reverting"


def rolling_hurst(
    x: np.ndarray,
    window: int = 126,
    estimator: str = "rs",
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Hurst exponent over a time series.

    Parameters
    ----------
    x         : 1-D array of observations
    window    : rolling window length (default 126 ~ 6 months of daily data)
    estimator : "rs" (faster) or "dfa" (more robust for short windows)
    step      : compute every `step` observations (1=every point, higher=faster)

    Returns
    -------
    (indices, hurst_values) : parallel arrays of int index and float H
    """
    x  = np.asarray(x, dtype=float)
    fn = hurst_rs if estimator == "rs" else hurst_dfa
    indices, values = [], []
    for i in range(window, len(x) + 1, step):
        h = fn(x[i - window: i])
        indices.append(i - 1)
        values.append(h)
    return np.array(indices, dtype=int), np.array(values, dtype=float)
