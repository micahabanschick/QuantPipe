"""Portfolio construction — pure function: signals + covariance + constraints → weights.

No I/O, no state. Every call is deterministic given the same inputs.
This makes testing trivial and keeps the research ↔ live code path identical.

Methods available:
  "equal"         — equal weight across selected symbols (strong baseline)
  "vol_scaled"    — inverse-volatility weighting (1/σ_i, normalised)
  "mean_variance" — mean-variance optimisation via cvxpy with Ledoit-Wolf cov
  "min_variance"  — minimum variance portfolio (ignores expected returns)
  "max_sharpe"    — maximum Sharpe ratio via PyPortfolioOpt

Start with "vol_scaled" — it often beats mean-variance once costs are real.
Graduate to "mean_variance" / "max_sharpe" once the signal has proven edge.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class PortfolioConstraints:
    max_position: float = 0.40       # max weight in any single name
    min_position: float = 0.00       # min non-zero weight (0 = no floor)
    max_gross: float = 1.00          # max sum of absolute weights (1 = long-only)
    max_turnover: float | None = None  # max one-way turnover per rebalance (None = uncapped)
    long_only: bool = True


def construct_portfolio(
    signals: pl.DataFrame,
    cov_matrix: np.ndarray | None = None,
    symbols: list[str] | None = None,
    constraints: PortfolioConstraints | None = None,
    method: str = "vol_scaled",
    expected_returns: np.ndarray | None = None,
    current_weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Compute target portfolio weights from a signal DataFrame.

    Parameters
    ----------
    signals          : [rebalance_date, symbol, weight] output of momentum_weights()
                       OR [rebalance_date, symbol, signal_value] raw signal scores
    cov_matrix       : (N×N) annualised covariance matrix (required for mean_variance/min_variance)
    symbols          : Symbol list aligned with cov_matrix columns
    constraints      : PortfolioConstraints (defaults if None)
    method           : "equal" | "vol_scaled" | "mean_variance" | "min_variance" | "max_sharpe"
    expected_returns : (N,) array of expected returns (required for mean_variance/max_sharpe)
    current_weights  : Dict of current weights for turnover constraint

    Returns
    -------
    DataFrame [rebalance_date, symbol, weight]  — weights sum to 1.0 per date
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    if method == "equal":
        return _equal_weights(signals, constraints)
    elif method == "vol_scaled":
        if cov_matrix is None or symbols is None:
            raise ValueError("vol_scaled requires cov_matrix and symbols")
        return _vol_scaled_weights(signals, cov_matrix, symbols, constraints)
    elif method == "mean_variance":
        if cov_matrix is None or symbols is None or expected_returns is None:
            raise ValueError("mean_variance requires cov_matrix, symbols, and expected_returns")
        return _mean_variance_weights(signals, cov_matrix, symbols, expected_returns, constraints)
    elif method == "min_variance":
        if cov_matrix is None or symbols is None:
            raise ValueError("min_variance requires cov_matrix and symbols")
        return _min_variance_weights(signals, cov_matrix, symbols, constraints)
    elif method == "max_sharpe":
        if cov_matrix is None or symbols is None or expected_returns is None:
            raise ValueError("max_sharpe requires cov_matrix, symbols, and expected_returns")
        return _max_sharpe_weights(signals, cov_matrix, symbols, expected_returns, constraints)
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ── Private implementations ───────────────────────────────────────────────────

def _equal_weights(
    signals: pl.DataFrame,
    constraints: PortfolioConstraints,
) -> pl.DataFrame:
    """Equal weight — 1/N across all symbols with non-zero signal."""
    selected = _get_selected(signals)
    counts = selected.group_by("rebalance_date").agg(pl.len().alias("n"))
    weights = (
        selected.join(counts, on="rebalance_date")
        .with_columns((1.0 / pl.col("n")).alias("weight"))
        .select(["rebalance_date", "symbol", "weight"])
    )
    return _apply_position_cap(weights, constraints)


def _vol_scaled_weights(
    signals: pl.DataFrame,
    cov_matrix: np.ndarray,
    symbols: list[str],
    constraints: PortfolioConstraints,
) -> pl.DataFrame:
    """Inverse-volatility weighting: w_i = (1/σ_i) / Σ(1/σ_j)."""
    vols = np.sqrt(np.diag(cov_matrix))  # annualised vol per symbol
    vol_map = dict(zip(symbols, vols))

    selected = _get_selected(signals)
    rows = []
    for rebal_date, group in selected.group_by("rebalance_date"):
        syms = group["symbol"].to_list()
        inv_vols = np.array([1.0 / vol_map.get(s, 1.0) for s in syms])
        raw_weights = inv_vols / inv_vols.sum()
        for sym, w in zip(syms, raw_weights):
            rows.append({"rebalance_date": rebal_date[0], "symbol": sym, "weight": float(w)})

    if not rows:
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})

    weights = pl.DataFrame(rows).with_columns(pl.col("rebalance_date").cast(pl.Date))
    return _apply_position_cap(weights, constraints)


def _mean_variance_weights(
    signals: pl.DataFrame,
    cov_matrix: np.ndarray,
    symbols: list[str],
    expected_returns: np.ndarray,
    constraints: PortfolioConstraints,
) -> pl.DataFrame:
    """Mean-variance optimisation via cvxpy.

    Maximises: μᵀw - (λ/2) wᵀΣw  subject to constraints.
    λ = 1 (unit risk aversion) — adjust if needed.
    """
    import cvxpy as cp

    selected = _get_selected(signals)
    rows = []
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    for rebal_date, group in selected.group_by("rebalance_date"):
        syms = group["symbol"].to_list()
        n = len(syms)
        idx = [sym_to_idx[s] for s in syms if s in sym_to_idx]
        if len(idx) != n:
            continue

        sub_cov = cov_matrix[np.ix_(idx, idx)]
        sub_mu = expected_returns[idx]

        w = cp.Variable(n)
        risk_aversion = 1.0
        objective = cp.Maximize(sub_mu @ w - (risk_aversion / 2) * cp.quad_form(w, sub_cov))

        constraints_list = [cp.sum(w) == 1.0]
        if constraints.long_only:
            constraints_list.append(w >= constraints.min_position)
        constraints_list.append(w <= constraints.max_position)

        prob = cp.Problem(objective, constraints_list)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            # Fall back to equal weight on solver failure
            raw = np.ones(n) / n
        else:
            raw = np.clip(w.value, 0.0, None)
            raw /= raw.sum() if raw.sum() > 0 else 1.0

        for sym, wt in zip(syms, raw):
            rows.append({"rebalance_date": rebal_date[0], "symbol": sym, "weight": float(wt)})

    if not rows:
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("rebalance_date").cast(pl.Date)).sort(["rebalance_date", "symbol"])


def _min_variance_weights(
    signals: pl.DataFrame,
    cov_matrix: np.ndarray,
    symbols: list[str],
    constraints: PortfolioConstraints,
) -> pl.DataFrame:
    """Minimum variance portfolio via PyPortfolioOpt."""
    from pypfopt import EfficientFrontier

    selected = _get_selected(signals)
    rows = []
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    for rebal_date, group in selected.group_by("rebalance_date"):
        syms = group["symbol"].to_list()
        idx = [sym_to_idx[s] for s in syms if s in sym_to_idx]
        if len(idx) != len(syms):
            continue

        sub_cov = cov_matrix[np.ix_(idx, idx)]
        ef = EfficientFrontier(None, sub_cov, weight_bounds=(0, constraints.max_position))
        try:
            ef.min_volatility()
            raw_vals = list(ef.clean_weights().values())
        except Exception:
            raw_vals = [1.0 / len(syms)] * len(syms)

        for sym, wt in zip(syms, raw_vals):
            rows.append({"rebalance_date": rebal_date[0], "symbol": sym, "weight": float(wt)})

    if not rows:
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("rebalance_date").cast(pl.Date)).sort(["rebalance_date", "symbol"])


def _max_sharpe_weights(
    signals: pl.DataFrame,
    cov_matrix: np.ndarray,
    symbols: list[str],
    expected_returns: np.ndarray,
    constraints: PortfolioConstraints,
) -> pl.DataFrame:
    """Maximum Sharpe ratio portfolio via PyPortfolioOpt."""
    from pypfopt import EfficientFrontier

    selected = _get_selected(signals)
    rows = []
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    for rebal_date, group in selected.group_by("rebalance_date"):
        syms = group["symbol"].to_list()
        idx = [sym_to_idx[s] for s in syms if s in sym_to_idx]
        if len(idx) != len(syms):
            continue

        sub_cov = cov_matrix[np.ix_(idx, idx)]
        sub_mu = expected_returns[idx]
        ef = EfficientFrontier(sub_mu, sub_cov, weight_bounds=(0, constraints.max_position))
        try:
            ef.max_sharpe(risk_free_rate=0.0)
            raw_vals = list(ef.clean_weights().values())
        except Exception:
            raw_vals = [1.0 / len(syms)] * len(syms)

        for sym, wt in zip(syms, raw_vals):
            rows.append({"rebalance_date": rebal_date[0], "symbol": sym, "weight": float(wt)})

    if not rows:
        return pl.DataFrame(schema={"rebalance_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("rebalance_date").cast(pl.Date)).sort(["rebalance_date", "symbol"])


def _get_selected(signals: pl.DataFrame) -> pl.DataFrame:
    """Return only selected rows, or all rows if no 'selected' column."""
    if "selected" in signals.columns:
        return signals.filter(pl.col("selected"))
    return signals


def _apply_position_cap(weights: pl.DataFrame, constraints: PortfolioConstraints) -> pl.DataFrame:
    """Clip weights to max_position and renormalise to sum=1."""
    weights = weights.with_columns(
        pl.col("weight").clip(0.0, constraints.max_position)
    )
    totals = weights.group_by("rebalance_date").agg(pl.col("weight").sum().alias("total"))
    weights = (
        weights.join(totals, on="rebalance_date")
        .with_columns((pl.col("weight") / pl.col("total")).alias("weight"))
        .drop("total")
    )
    return weights.sort(["rebalance_date", "symbol"])
