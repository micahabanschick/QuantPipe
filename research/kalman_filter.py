"""Kalman filter - time-varying parameter (TVP) regression for dynamic factor betas.

Pure analytics. No Streamlit, no Plotly, no I/O.

Model (Fahrmeir & Tutz parameterisation):

    Observation:  r_t  = H_t @ beta_t + eps_t,   eps_t ~ N(0, R)
    Transition:   beta_t  = beta_{t-1} + eta_t,       eta_t ~ N(0, Q)

where
    beta_t in ?^K   (K = n_factors + 1,  first element is the intercept)
    H_t = [1, f1_t, f2_t, ...]   row vector of factor values at t
    Q   = (delta / (1-delta)) * I_K       (delta controls how fast betas adapt)
    R   = estimated from full-sample OLS residuals

Public API:
    kalman_smooth_betas(portfolio_returns, factor_returns, delta, P_init_scale) -> KalmanResult
    kalman_hedge_ratio(asset_returns, hedge_returns, delta) -> (betas, variances)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


# ?? Dataclass ??????????????????????????????????????????????????????????????????

@dataclass
class KalmanResult:
    """Output of a Kalman TVP regression.

    Attributes
    ----------
    dates               : T observation dates (aligned to the input series)
    filtered_betas      : (T, K) filtered state means.  Column 0 = alpha, 1..K-1 = betas.
    filtered_vars       : (T, K, K) filtered posterior covariances
    innovations         : (T,) one-step-ahead forecast errors (v_t = r_t - H_t @ beta_pred)
    innovation_variances: (T,) innovation variances (S_t = H_t @ P_pred @ H_t + R)
    log_likelihood      : scalar, sum of Gaussian log-likelihoods
    factor_names        : K-1 factor names (excluding intercept)
    """
    dates:                list[date]      = field(default_factory=list)
    filtered_betas:       np.ndarray     = field(default_factory=lambda: np.empty(0))
    filtered_vars:        np.ndarray     = field(default_factory=lambda: np.empty(0))
    innovations:          np.ndarray     = field(default_factory=lambda: np.empty(0))
    innovation_variances: np.ndarray     = field(default_factory=lambda: np.empty(0))
    log_likelihood:       float          = float("-inf")
    factor_names:         list[str]      = field(default_factory=list)


# ?? Core implementation ????????????????????????????????????????????????????????

def kalman_smooth_betas(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    delta: float = 1e-4,
    P_init_scale: float = 1.0,
) -> KalmanResult:
    """Time-varying parameter regression via Kalman filter.

    Parameters
    ----------
    portfolio_returns : pd.Series  (T,)  daily returns of the portfolio / asset
    factor_returns   : pd.DataFrame (T x K) factor daily returns, one column per factor
    delta            : float  process noise scaling.  Higher = betas adapt faster.
                       Typical range: 1e-5 (very slow) to 1e-2 (fast adaptation).
                       delta/(1-delta) is the ratio of process-to-observation noise variance.
    P_init_scale     : float  scales the initial posterior covariance P_0.

    Returns
    -------
    KalmanResult with filtered betas (T x K), variances (T x K x K),
    innovations (T,), innovation variances (T,), log-likelihood.
    """
    # ?? Align data ?????????????????????????????????????????????????????????????
    pret = portfolio_returns.copy()
    fret = factor_returns.copy()
    pret.index = pd.to_datetime(pret.index)
    fret.index = pd.to_datetime(fret.index)

    common = pret.index.intersection(fret.index)
    pret   = pret.loc[common]
    fret   = fret.loc[common].dropna(how="any")
    pret   = pret.loc[fret.index].dropna()
    fret   = fret.loc[pret.index]

    T = len(pret)
    K = fret.shape[1] + 1   # +1 for intercept
    factor_names = list(fret.columns)

    if T < K + 5:
        return KalmanResult(factor_names=factor_names)

    y = pret.values.astype(float)               # (T,)
    F = fret.values.astype(float)               # (T, K-1)

    # Design matrix: prepend intercept column
    H_all = np.column_stack([np.ones(T), F])    # (T, K)

    # ?? Initialise via full-sample OLS ?????????????????????????????????????????
    X_aug = H_all
    try:
        coeffs_ols, resid_ols, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        return KalmanResult(factor_names=factor_names)

    y_hat_ols = X_aug @ coeffs_ols
    resid     = y - y_hat_ols
    R         = float(np.var(resid))             # observation noise variance
    if R < 1e-12:
        R = 1e-8

    # Process noise covariance: Q = (delta/(1-delta)) * I_K
    q_scale = delta / (1.0 - delta)
    Q       = q_scale * np.eye(K)

    # Initial state and covariance
    beta_t  = coeffs_ols.copy()                  # (K,)
    XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
    P_t     = P_init_scale * R * XtX_inv         # (K, K)

    # ?? Storage ????????????????????????????????????????????????????????????????
    betas_out  = np.empty((T, K))
    vars_out   = np.empty((T, K, K))
    innov      = np.empty(T)
    innov_var  = np.empty(T)
    log_lik    = 0.0
    _LOG2PI    = float(np.log(2.0 * np.pi))

    # ?? Kalman predict-update loop ?????????????????????????????????????????????
    for t in range(T):
        H_t = H_all[t]                           # (K,)
        y_t = y[t]

        # Predict
        beta_pred = beta_t                        # (K,)  (random walk: F=I)
        P_pred    = P_t + Q                       # (K, K)

        # Innovation
        v_t  = y_t - H_t @ beta_pred             # scalar
        S_t  = float(H_t @ P_pred @ H_t) + R     # scalar, innovation variance

        if S_t < 1e-14:
            S_t = 1e-14

        # Log-likelihood contribution
        log_lik += -0.5 * (_LOG2PI + np.log(S_t) + v_t ** 2 / S_t)

        # Kalman gain: K = P_pred @ H' / S  (K,)
        K_gain  = (P_pred @ H_t) / S_t           # (K,)

        # Update
        beta_t = beta_pred + K_gain * v_t        # (K,)
        P_t    = P_pred - np.outer(K_gain, H_t) @ P_pred  # (K, K)

        # Symmetrise to prevent numerical drift
        P_t = 0.5 * (P_t + P_t.T)

        betas_out[t]  = beta_t
        vars_out[t]   = P_t
        innov[t]      = v_t
        innov_var[t]  = S_t

    dates_out = [d.date() if hasattr(d, "date") else d for d in pret.index]

    return KalmanResult(
        dates=dates_out,
        filtered_betas=betas_out,
        filtered_vars=vars_out,
        innovations=innov,
        innovation_variances=innov_var,
        log_likelihood=float(log_lik),
        factor_names=factor_names,
    )


def kalman_hedge_ratio(
    asset_returns: pd.Series,
    hedge_returns: pd.Series,
    delta: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-factor dynamic hedge ratio via Kalman filter.

    Convenience wrapper around ``kalman_smooth_betas`` with one factor.

    Parameters
    ----------
    asset_returns : pd.Series  returns of the asset to hedge
    hedge_returns : pd.Series  returns of the hedging instrument
    delta         : float  process noise (controls adaptation speed)

    Returns
    -------
    (betas, variances)
        betas     : (T,) dynamic hedge ratio (beta to the hedge instrument)
        variances : (T,) posterior variance of the hedge ratio (uncertainty)
    """
    factor_df = hedge_returns.to_frame(name="hedge")
    result    = kalman_smooth_betas(asset_returns, factor_df, delta=delta)

    if result.filtered_betas.size == 0:
        return np.empty(0), np.empty(0)

    # Column 0 = intercept, column 1 = hedge beta
    betas     = result.filtered_betas[:, 1]
    variances = result.filtered_vars[:, 1, 1]
    return betas, variances
