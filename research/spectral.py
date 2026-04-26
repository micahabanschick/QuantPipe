"""Spectral and signal-processing analytics for quantitative finance.

Pure analytics -- no Streamlit, no Plotly.

Public API
----------
compute_psd(x, fs, nperseg)            -> (freqs, power)       Welch smoothed PSD
dominant_cycles(freqs, power, top_n)   -> list[(period, pct)]  top spectral peaks
fft_filter(x, cutoff_frac, mode)       -> np.ndarray           low/high/band-pass
haar_wavelet_1d(x, levels)             -> list[np.ndarray]     [approx, d1, d2, ...]
gbm_paths(S0, mu, sigma, T, n, seed)   -> np.ndarray           (n, T+1) price paths
estimate_gbm_params(prices)            -> (mu_ann, sigma_ann)
acf(x, max_lag)                        -> np.ndarray           autocorrelation [1..max_lag]
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, find_peaks


# ── Power Spectral Density ────────────────────────────────────────────────────

def compute_psd(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch smoothed Power Spectral Density.

    Parameters
    ----------
    x       : 1-D array (e.g. demeaned log-prices or returns)
    fs      : sampling frequency (1.0 = 1 sample/day for daily data)
    nperseg : Welch segment length; defaults to min(len(x)//4, 256)

    Returns
    -------
    freqs : frequencies in cycles per sample (or per day if fs=1)
    power : one-sided PSD (same units as x squared per Hz)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.array([]), np.array([])
    seg = nperseg or min(len(x) // 4, 256)
    seg = max(seg, 8)
    freqs, power = welch(x, fs=fs, nperseg=seg, window="hann", detrend="constant")
    return freqs, power


def dominant_cycles(
    freqs: np.ndarray,
    power: np.ndarray,
    top_n: int = 5,
    min_period: float = 2.0,
) -> list[dict]:
    """Return the top-N spectral peaks as (period_days, relative_power_pct).

    Parameters
    ----------
    freqs      : frequency array from compute_psd
    power      : power array from compute_psd
    top_n      : number of peaks to return
    min_period : minimum cycle period to consider (default 2 days)

    Returns
    -------
    list of dicts: {freq, period_days, power, rel_power_pct, label}
    """
    if len(freqs) < 4 or len(power) < 4:
        return []

    # Exclude DC (freq=0) and very high frequencies (period < min_period)
    mask = (freqs > 0) & (1.0 / np.where(freqs > 0, freqs, np.inf) >= min_period)
    f_sub = freqs[mask]
    p_sub = power[mask]

    if len(f_sub) == 0:
        return []

    peaks, _ = find_peaks(p_sub, height=np.percentile(p_sub, 50))
    if len(peaks) == 0:
        # Fall back to top-N by power
        peaks = np.argsort(p_sub)[::-1][:top_n]

    peaks_sorted = peaks[np.argsort(p_sub[peaks])[::-1]][:top_n]
    total_power = float(p_sub.sum()) or 1.0

    result = []
    for idx in peaks_sorted:
        freq   = float(f_sub[idx])
        period = 1.0 / freq if freq > 0 else np.inf
        label  = _period_label(period)
        result.append({
            "freq":          round(freq, 6),
            "period_days":   round(period, 1),
            "power":         float(p_sub[idx]),
            "rel_power_pct": round(float(p_sub[idx]) / total_power * 100, 2),
            "label":         label,
        })
    return result


def _period_label(days: float) -> str:
    if days < 4:
        return "~Daily"
    if days < 8:
        return "~Weekly"
    if days < 16:
        return "Bi-weekly"
    if days < 35:
        return "~Monthly"
    if days < 70:
        return "Bi-monthly"
    if days < 100:
        return "~Quarterly"
    if days < 200:
        return "Semi-annual"
    return "~Annual"


# ── FFT Frequency Filter ──────────────────────────────────────────────────────

def fft_filter(
    x: np.ndarray,
    cutoff_frac: float = 0.05,
    mode: str = "low",
) -> np.ndarray:
    """Apply a brick-wall FFT filter to extract trend or cycles.

    Parameters
    ----------
    x            : 1-D array
    cutoff_frac  : cutoff as a fraction of Nyquist (0..0.5); 0.05 ~ 20-day cycle
    mode         : "low" (trend), "high" (noise), or "band" (cycles between
                   cutoff_frac and 2*cutoff_frac)

    Returns
    -------
    Filtered signal (same length as x).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)

    if mode == "low":
        X[freqs > cutoff_frac] = 0.0
    elif mode == "high":
        X[freqs < cutoff_frac] = 0.0
    elif mode == "band":
        X[(freqs < cutoff_frac) | (freqs > 2 * cutoff_frac)] = 0.0

    return np.fft.irfft(X, n=n)


# ── Haar Wavelet ──────────────────────────────────────────────────────────────

def haar_wavelet_1d(
    x: np.ndarray,
    levels: int = 3,
) -> list[np.ndarray]:
    """Multi-level Haar wavelet decomposition.

    Parameters
    ----------
    x      : 1-D array (should have length divisible by 2^levels)
    levels : decomposition depth

    Returns
    -------
    [approx, detail_L1, detail_L2, ..., detail_Ln]
    Lengths halve with each level. detail_L1 is finest (highest-frequency).
    """
    x = np.asarray(x, dtype=float)
    result = []
    current = x.copy()
    for _ in range(levels):
        n = len(current)
        if n < 4:
            break
        if n % 2:
            current = np.append(current, current[-1])
        approx  = (current[0::2] + current[1::2]) / np.sqrt(2)
        detail  = (current[0::2] - current[1::2]) / np.sqrt(2)
        result.append(detail)
        current = approx
    result.append(current)          # final approximation
    result.reverse()                # [approx, coarse->fine details]
    return result


# ── Geometric Brownian Motion ─────────────────────────────────────────────────

def estimate_gbm_params(prices: np.ndarray) -> tuple[float, float]:
    """Estimate annualised GBM drift (mu) and volatility (sigma) from prices.

    Returns
    -------
    (mu_ann, sigma_ann) : annualised drift and volatility
    """
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 2:
        return 0.0, 0.0
    log_rets = np.diff(np.log(prices))
    sigma_d  = float(log_rets.std(ddof=1))
    mu_d     = float(log_rets.mean()) + 0.5 * sigma_d ** 2  # correct for Ito
    return mu_d * 252, sigma_d * np.sqrt(252)


def gbm_paths(
    S0: float,
    mu_ann: float,
    sigma_ann: float,
    T_days: int,
    n_paths: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Simulate GBM price paths.

    Parameters
    ----------
    S0        : starting price
    mu_ann    : annualised drift
    sigma_ann : annualised volatility
    T_days    : number of daily steps to simulate
    n_paths   : number of Monte Carlo paths
    seed      : RNG seed

    Returns
    -------
    paths : np.ndarray of shape (n_paths, T_days + 1)
            paths[:, 0] == S0 for all paths
    """
    dt    = 1.0 / 252
    mu_d  = mu_ann * dt
    sig_d = sigma_ann * np.sqrt(dt)
    rng   = np.random.default_rng(seed)
    z     = rng.standard_normal((n_paths, T_days))
    log_r = (mu_d - 0.5 * sig_d ** 2) + sig_d * z
    paths = S0 * np.exp(np.cumsum(log_r, axis=1))
    return np.column_stack([np.full(n_paths, S0), paths])


# ── Autocorrelation ───────────────────────────────────────────────────────────

def acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelation function at lags 1..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom < 1e-12:
        return np.zeros(max_lag)
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            result.append(0.0)
        else:
            result.append(float(np.dot(xm[:-lag], xm[lag:]) / denom))
    return np.array(result)
