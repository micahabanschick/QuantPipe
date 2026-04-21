"""Monte Carlo block-bootstrap analysis for strategy returns.

Adapted from mcda.py (QuantConnect bootstrap notebook).
Pure analytics. No Streamlit, no Plotly.

Public API:
    MCConfig              : simulation configuration
    MCResult              : pre-computed results (percentile bands + distributions)
    load_returns_csv()    : load trade returns or equity CSV
    run(returns, config)  -> MCResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MCConfig:
    n_simulations:   int   = 10_000
    block_size:      int   = 10       # circular block bootstrap block length
    initial_capital: float = 100_000.0
    periods_per_yr:  float = 252.0    # 252 = daily, 26 = biweekly, 12 = monthly
    seed:            int   = 42
    target_sharpe:   float = 1.0
    target_calmar:   float = 0.5
    target_max_dd:   float = -0.20    # expressed as negative fraction (e.g. -0.20 = 20% DD)


# ── Results ────────────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Pre-computed Monte Carlo results — ready for dashboard rendering."""

    n_simulations: int
    n_periods:     int
    block_size:    int

    # Equity fan chart (period-index → percentile value)
    fan_x:   list
    fan_p5:  list
    fan_p10: list
    fan_p25: list
    fan_p50: list
    fan_p75: list
    fan_p90: list
    fan_p95: list
    orig_equity: list

    # Terminal wealth
    terminal_values:       list    # all N_SIM final equity values
    terminal_percentiles:  dict    # {5: v, 25: v, 50: v, 75: v, 95: v}
    p_loss:                float   # P(final < initial)
    p_loss_25pct:          float   # P(final < 75% of initial)
    p_double:              float   # P(final > 2× initial)

    # Metric distributions (one float per simulation)
    sharpe_dist:   list
    calmar_dist:   list
    max_dd_dist:   list
    sortino_dist:  list
    omega_dist:    list
    ann_vol_dist:  list

    # Original path metrics
    orig_sharpe:       float
    orig_calmar:       float
    orig_max_dd:       float
    orig_sortino:      float
    orig_ann_return:   float
    orig_total_return: float
    orig_ann_vol:      float

    # Return series statistics
    ret_mean:  float
    ret_std:   float
    ret_skew:  float
    ret_kurt:  float
    ret_min:   float
    ret_max:   float

    # Autocorrelation (lags 1–10 of returns and squared returns)
    acf_returns: list   # [rho_lag1, ..., rho_lag10]
    acf_squared: list   # [rho_lag1, ..., rho_lag10]
    acf_ci:      float  # 95% CI bound = 1.96 / sqrt(n)

    # Target achievement probabilities
    p_meets_sharpe: float
    p_meets_calmar: float
    p_meets_max_dd: float
    p_meets_all:    float
    p_meets_any_two: float

    # Convergence: running percentiles as function of sim count
    conv_n:        list   # simulation counts (log-spaced)
    conv_p5:       list   # running P5 of final equity
    conv_p50:      list   # running P50 of final equity
    conv_p95:      list   # running P95 of final equity
    conv_sharpe_p50: list  # running P50 of Sharpe

    # Rolling Sharpe (original path + bootstrap percentile bands)
    rolling_window:    int
    rolling_x:         list
    rolling_orig:      list
    rolling_p10:       list
    rolling_p50:       list
    rolling_p90:       list

    # Summary table (list of dicts for pd.DataFrame)
    summary_rows: list


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_returns_csv(
    path: str,
    from_equity: bool = False,
    initial_capital: float = 100_000.0,
) -> np.ndarray:
    """Load period returns from CSV.

    Handles three formats:
      1. QC trade log with columns 'Entry Price', 'Exit Price', 'P&L', 'Exit Time':
         aggregates concurrent trades by exit date into portfolio-level period returns.
      2. CSV with a 'return' / 'returns' / 'ret' column.
      3. Equity curve: any column named 'equity', 'value', 'nav', etc. → pct_change.

    Parameters
    ----------
    path            : path to CSV file
    from_equity     : True = treat as equity curve, False = trade/return format
    initial_capital : used when computing P&L-based returns

    Returns
    -------
    np.ndarray of float (period returns)
    """
    df = pd.read_csv(path)

    if from_equity:
        for c in df.columns:
            if c.strip().lower() in ("equity", "value", "portfolio", "nav", "balance", "close"):
                eq = df[c].dropna().values.astype(float)
                return np.diff(eq) / eq[:-1]
        numeric = df.select_dtypes(include="number").columns
        eq = df[numeric[-1]].dropna().values.astype(float)
        return np.diff(eq) / eq[:-1]

    # QC trade-log format
    if all(c in df.columns for c in ["Entry Price", "Exit Price", "P&L", "Exit Time"]):
        df["Exit Time"] = pd.to_datetime(df["Exit Time"])
        fees_col = "Fees" if "Fees" in df.columns else None
        agg = {"P&L": "sum"}
        if fees_col:
            agg[fees_col] = "sum"
        by_exit = df.groupby("Exit Time").agg(agg).sort_index()
        capital = float(initial_capital)
        period_rets = []
        for _, row in by_exit.iterrows():
            net = float(row["P&L"]) - (float(row[fees_col]) if fees_col else 0.0)
            period_rets.append(net / capital)
            capital += net
        return np.array(period_rets)

    # Explicit return column
    for c in df.columns:
        if c.strip().lower() in ("return", "returns", "trade_return", "ret", "period_return"):
            return df[c].dropna().values.astype(float)

    raise ValueError(
        f"Cannot parse returns from CSV. Columns: {list(df.columns)}\n"
        "Expected: QC trade log (Entry Price / P&L / Exit Time), "
        "a 'return' column, or an equity curve with from_equity=True."
    )


# ── Core helpers ──────────────────────────────────────────────────────────────

def _block_bootstrap(returns: np.ndarray, n_out: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: preserves short-range serial dependence."""
    n = len(returns)
    path = np.empty(n_out)
    i = 0
    while i < n_out:
        start  = int(rng.integers(0, n))
        length = min(block_size, n_out - i)
        idx    = np.arange(start, start + length) % n
        path[i:i + length] = returns[idx]
        i += length
    return path


def _build_equity(rets: np.ndarray, capital: float) -> np.ndarray:
    eq = np.empty(len(rets) + 1)
    eq[0] = capital
    eq[1:] = capital * np.cumprod(1.0 + rets)
    return eq


def _metrics(equity: np.ndarray, rets: np.ndarray, ppyr: float, capital: float) -> dict:
    """Compute performance metrics for a single path."""
    total_ret = equity[-1] / capital - 1.0
    years     = len(rets) / ppyr
    ann_ret   = (1.0 + total_ret) ** (1.0 / max(years, 1e-3)) - 1.0 if total_ret > -1.0 else -1.0
    ann_vol   = float(np.std(rets, ddof=1) * np.sqrt(ppyr))

    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

    running_max = np.maximum.accumulate(equity)
    dd          = (equity - running_max) / running_max
    max_dd      = float(dd.min())
    calmar      = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    neg    = rets[rets < 0]
    ds_dev = float(np.std(neg, ddof=1) * np.sqrt(ppyr)) if len(neg) > 1 else 1e-9
    sortino = ann_ret / ds_dev if ds_dev > 1e-9 else 0.0

    gains   = float(rets[rets > 0].sum())
    losses  = float(-rets[rets < 0].sum())
    omega   = gains / losses if losses > 1e-12 else 99.0

    return dict(
        total_ret=total_ret, ann_ret=ann_ret, ann_vol=ann_vol,
        sharpe=sharpe, max_dd=max_dd, calmar=calmar,
        sortino=sortino, omega=omega,
        final_equity=float(equity[-1]),
    )


def _rolling_sharpe_series(rets: np.ndarray, window: int, ppyr: float) -> np.ndarray:
    n = len(rets)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = rets[i - window + 1: i + 1]
        mu    = chunk.mean()
        sigma = chunk.std(ddof=1)
        out[i] = mu / sigma * np.sqrt(ppyr) if sigma > 1e-9 else 0.0
    return out


def _acf(series: np.ndarray, max_lag: int) -> list[float]:
    n = len(series)
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            result.append(0.0)
        else:
            r = float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
            result.append(r if not np.isnan(r) else 0.0)
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def run(returns: np.ndarray, config: MCConfig) -> MCResult:
    """Run Monte Carlo block-bootstrap analysis.

    Parameters
    ----------
    returns : 1-D array of period returns (e.g. daily, biweekly, monthly)
    config  : MCConfig

    Returns
    -------
    MCResult with all pre-computed data for dashboard rendering.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n       = len(returns)
    if n < 10:
        raise ValueError(f"Need at least 10 return observations, got {n}.")

    C      = config
    capital = C.initial_capital
    rng    = np.random.default_rng(C.seed)

    # ── Descriptive stats + ACF ───────────────────────────────────────────────
    ret_mean = float(returns.mean())
    ret_std  = float(returns.std(ddof=1))
    ret_skew = float(_scipy_stats.skew(returns))
    ret_kurt = float(_scipy_stats.kurtosis(returns))
    acf_r    = _acf(returns, 10)
    acf_sq   = _acf(returns ** 2, 10)
    acf_ci   = 1.96 / np.sqrt(n)

    # ── Original path metrics ─────────────────────────────────────────────────
    orig_equity = _build_equity(returns, capital)
    orig_m      = _metrics(orig_equity, returns, C.periods_per_yr, capital)

    # ── Bootstrap simulation ──────────────────────────────────────────────────
    # Store only final equity + metric scalars (not full paths, to save memory)
    # We do store equity paths temporarily to build percentile bands, then discard.
    n_paths = C.n_simulations
    equity_paths = np.empty((n_paths, n + 1))
    sim_rets     = np.empty((n_paths, n))

    for i in range(n_paths):
        r = _block_bootstrap(returns, n, C.block_size, rng)
        sim_rets[i]     = r
        equity_paths[i] = _build_equity(r, capital)

    # ── Fan chart percentile bands ────────────────────────────────────────────
    pcts  = np.percentile(equity_paths, [5, 10, 25, 50, 75, 90, 95], axis=0)
    fan_x = list(range(n + 1))

    # ── Terminal wealth ───────────────────────────────────────────────────────
    terminal = equity_paths[:, -1]
    term_pct = {int(p): float(np.percentile(terminal, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
    p_loss      = float((terminal < capital).mean())
    p_loss_25   = float((terminal < 0.75 * capital).mean())
    p_double    = float((terminal > 2.0 * capital).mean())

    # ── Per-path metrics ──────────────────────────────────────────────────────
    sharpe_d  = []
    calmar_d  = []
    max_dd_d  = []
    sortino_d = []
    omega_d   = []
    ann_vol_d = []

    for i in range(n_paths):
        m = _metrics(equity_paths[i], sim_rets[i], C.periods_per_yr, capital)
        sharpe_d.append(m["sharpe"])
        calmar_d.append(m["calmar"])
        max_dd_d.append(m["max_dd"])
        sortino_d.append(m["sortino"])
        omega_d.append(min(m["omega"], 20.0))  # cap outliers for display
        ann_vol_d.append(m["ann_vol"])

    sharpe_arr  = np.array(sharpe_d)
    calmar_arr  = np.array(calmar_d)
    max_dd_arr  = np.array(max_dd_d)

    # ── Target achievement ────────────────────────────────────────────────────
    m_sharpe = sharpe_arr >= C.target_sharpe
    m_calmar = calmar_arr >= C.target_calmar
    m_dd     = max_dd_arr >= C.target_max_dd   # DD is negative; target = -0.20

    p_meets_sharpe  = float(m_sharpe.mean())
    p_meets_calmar  = float(m_calmar.mean())
    p_meets_dd      = float(m_dd.mean())
    p_meets_all     = float((m_sharpe & m_calmar & m_dd).mean())
    p_meets_any_two = float(
        ((m_sharpe.astype(int) + m_calmar.astype(int) + m_dd.astype(int)) >= 2).mean()
    )

    # ── Convergence diagnostics ───────────────────────────────────────────────
    checkpoints = np.unique(
        np.logspace(np.log10(100), np.log10(n_paths), 150).astype(int)
    ).tolist()
    conv_n        = checkpoints
    conv_p5       = [float(np.percentile(terminal[:k], 5))  for k in checkpoints]
    conv_p50      = [float(np.percentile(terminal[:k], 50)) for k in checkpoints]
    conv_p95      = [float(np.percentile(terminal[:k], 95)) for k in checkpoints]
    conv_sharpe50 = [float(np.percentile(sharpe_arr[:k], 50)) for k in checkpoints]

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    roll_win  = min(max(int(C.periods_per_yr // 2), 20), n // 3)
    orig_roll = _rolling_sharpe_series(returns, roll_win, C.periods_per_yr)

    # Percentile bands from a 500-path sample
    sample_idx = rng.integers(0, n_paths, size=min(500, n_paths))
    roll_paths = np.array([
        _rolling_sharpe_series(sim_rets[i], roll_win, C.periods_per_yr)
        for i in sample_idx
    ])
    # Align to common non-NaN suffix
    start_valid = roll_win - 1
    roll_x      = list(range(start_valid, n))
    roll_data   = roll_paths[:, start_valid:]
    r_p10 = np.percentile(roll_data, 10, axis=0).tolist()
    r_p50 = np.percentile(roll_data, 50, axis=0).tolist()
    r_p90 = np.percentile(roll_data, 90, axis=0).tolist()
    roll_orig_valid = orig_roll[start_valid:].tolist()

    # ── Summary table ─────────────────────────────────────────────────────────
    pct_labels = [5, 10, 25, 50, 75, 90, 95]
    metrics_keys = ["sharpe", "calmar", "max_dd", "sortino", "ann_vol"]
    metrics_arrs = {
        "sharpe":  sharpe_arr,
        "calmar":  calmar_arr,
        "max_dd":  max_dd_arr,
        "sortino": np.array(sortino_d),
        "ann_vol": np.array(ann_vol_d),
    }
    summary_rows = []
    for p in pct_labels:
        row: dict = {"Percentile": f"P{p}"}
        for k, arr in metrics_arrs.items():
            row[k] = float(np.percentile(arr, p))
        row["final_equity"] = float(np.percentile(terminal, p))
        summary_rows.append(row)
    orig_row: dict = {"Percentile": "Original"}
    for k in metrics_keys:
        orig_row[k] = orig_m[k if k in orig_m else k]
    orig_row["sharpe"]       = orig_m["sharpe"]
    orig_row["calmar"]       = orig_m["calmar"]
    orig_row["max_dd"]       = orig_m["max_dd"]
    orig_row["sortino"]      = orig_m["sortino"]
    orig_row["ann_vol"]      = orig_m["ann_vol"]
    orig_row["final_equity"] = float(orig_equity[-1])
    summary_rows.append(orig_row)

    return MCResult(
        n_simulations=n_paths,
        n_periods=n,
        block_size=C.block_size,

        fan_x=fan_x,
        fan_p5=pcts[0].tolist(),
        fan_p10=pcts[1].tolist(),
        fan_p25=pcts[2].tolist(),
        fan_p50=pcts[3].tolist(),
        fan_p75=pcts[4].tolist(),
        fan_p90=pcts[5].tolist(),
        fan_p95=pcts[6].tolist(),
        orig_equity=orig_equity.tolist(),

        terminal_values=terminal.tolist(),
        terminal_percentiles=term_pct,
        p_loss=p_loss,
        p_loss_25pct=p_loss_25,
        p_double=p_double,

        sharpe_dist=sharpe_d,
        calmar_dist=calmar_d,
        max_dd_dist=max_dd_d,
        sortino_dist=sortino_d,
        omega_dist=omega_d,
        ann_vol_dist=ann_vol_d,

        orig_sharpe=orig_m["sharpe"],
        orig_calmar=orig_m["calmar"],
        orig_max_dd=orig_m["max_dd"],
        orig_sortino=orig_m["sortino"],
        orig_ann_return=orig_m["ann_ret"],
        orig_total_return=orig_m["total_ret"],
        orig_ann_vol=orig_m["ann_vol"],

        ret_mean=ret_mean,
        ret_std=ret_std,
        ret_skew=ret_skew,
        ret_kurt=ret_kurt,
        ret_min=float(returns.min()),
        ret_max=float(returns.max()),

        acf_returns=acf_r,
        acf_squared=acf_sq,
        acf_ci=float(acf_ci),

        p_meets_sharpe=p_meets_sharpe,
        p_meets_calmar=p_meets_calmar,
        p_meets_max_dd=p_meets_dd,
        p_meets_all=p_meets_all,
        p_meets_any_two=p_meets_any_two,

        conv_n=conv_n,
        conv_p5=conv_p5,
        conv_p50=conv_p50,
        conv_p95=conv_p95,
        conv_sharpe_p50=conv_sharpe50,

        rolling_window=roll_win,
        rolling_x=roll_x,
        rolling_orig=roll_orig_valid,
        rolling_p10=r_p10,
        rolling_p50=r_p50,
        rolling_p90=r_p90,

        summary_rows=summary_rows,
    )
