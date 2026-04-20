"""Risk management engine — pre-trade checks and exposure monitoring.

All check functions are hard blocks, not advisory warnings.
If a check fails, the calling code must not proceed with the trade.

Core public API:
    compute_exposures(weights, prices, sector_map) -> ExposureReport
    historical_var(returns, weights, confidence, lookback)  -> float
    pre_trade_check(proposed_weights, current_weights, prices, limits) -> CheckResult
    generate_risk_report(...)  -> RiskReport
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RiskLimits:
    max_position: float = 0.40        # max weight in any single name
    max_sector: float = 0.60          # max aggregate weight in any one sector
    max_gross: float = 1.00           # max sum of absolute weights
    max_net: float = 1.00             # max net (long - short) exposure
    max_top5_concentration: float = 0.80  # max weight in top-5 names combined
    var_limit_pct: float | None = None    # max 1-day 95% VaR as % of NAV (None = uncapped)


@dataclass
class ExposureReport:
    as_of: date
    gross_exposure: float
    net_exposure: float
    sector_exposures: dict[str, float]
    top_5_concentration: float
    top_10_concentration: float
    largest_position: tuple[str, float]   # (symbol, weight)
    n_positions: int


@dataclass
class CheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "PRE-TRADE CHECK PASSED"
        return "PRE-TRADE CHECK FAILED:\n" + "\n".join(f"  - {v}" for v in self.violations)


@dataclass
class RiskReport:
    as_of: date
    var_1d_95: float
    var_1d_99: float
    exposures: ExposureReport
    check: CheckResult
    stress_results: dict[str, float] = field(default_factory=dict)  # scenario → P&L %


# ── GICS sector map for the equity ETF universe ───────────────────────────────

EQUITY_SECTOR_MAP: dict[str, str] = {
    # 9-box (style/size — treat as "Equity Blend" sector)
    "IWB": "US Equity", "IWF": "US Equity", "IWD": "US Equity",
    "IWR": "US Equity", "IWP": "US Equity", "IWS": "US Equity",
    "IWM": "US Equity", "IWO": "US Equity", "IWN": "US Equity",
    # GICS sectors
    "XLK": "Technology",      "XLV": "Health Care",   "XLF": "Financials",
    "XLY": "Consumer Disc",   "XLP": "Consumer Stap", "XLE": "Energy",
    "XLI": "Industrials",     "XLB": "Materials",     "XLU": "Utilities",
    "XLRE": "Real Estate",    "XLC": "Communication",
    # Benchmarks
    "SPY": "Broad Market",    "QQQ": "Broad Market",  "DIA": "Broad Market",
    "AGG": "Fixed Income",    "TLT": "Fixed Income",  "GLD": "Commodities",
}


# ── Core functions ────────────────────────────────────────────────────────────

def compute_exposures(
    weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
    as_of: date | None = None,
) -> ExposureReport:
    """Compute portfolio exposures from a {symbol: weight} dict.

    Parameters
    ----------
    weights    : {symbol: weight} — may include negative (short) weights
    sector_map : {symbol: sector_name} — uses EQUITY_SECTOR_MAP if None
    as_of      : Report date
    """
    if sector_map is None:
        sector_map = EQUITY_SECTOR_MAP
    if as_of is None:
        as_of = date.today()

    if not weights:
        return ExposureReport(
            as_of=as_of, gross_exposure=0.0, net_exposure=0.0,
            sector_exposures={}, top_5_concentration=0.0,
            top_10_concentration=0.0, largest_position=("", 0.0), n_positions=0,
        )

    w_arr = np.array(list(weights.values()))
    syms = list(weights.keys())

    gross = float(np.abs(w_arr).sum())
    net = float(w_arr.sum())

    # Sector aggregation
    sector_exposures: dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "Other")
        sector_exposures[sector] = sector_exposures.get(sector, 0.0) + abs(w)

    # Concentration
    sorted_w = sorted(np.abs(w_arr), reverse=True)
    top5 = float(sum(sorted_w[:5]))
    top10 = float(sum(sorted_w[:10]))

    largest_idx = int(np.argmax(np.abs(w_arr)))
    largest = (syms[largest_idx], float(w_arr[largest_idx]))
    n_pos = int(np.sum(np.abs(w_arr) > 1e-6))

    return ExposureReport(
        as_of=as_of,
        gross_exposure=round(gross, 4),
        net_exposure=round(net, 4),
        sector_exposures={k: round(v, 4) for k, v in sorted(sector_exposures.items(), key=lambda x: -x[1])},
        top_5_concentration=round(top5, 4),
        top_10_concentration=round(top10, 4),
        largest_position=(largest[0], round(largest[1], 4)),
        n_positions=n_pos,
    )


def historical_var(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
    lookback: int = 252,
) -> float:
    """1-day historical simulation VaR as a fraction of NAV.

    Parameters
    ----------
    returns_matrix : (T × N) daily return matrix, columns aligned with weights
    weights        : (N,) portfolio weight vector
    confidence     : VaR confidence level (0.95 = 95%)
    lookback       : Number of most-recent trading days to use

    Returns
    -------
    VaR as a positive fraction (e.g. 0.021 = 2.1% 1-day loss at given confidence)
    """
    if len(returns_matrix) == 0 or len(weights) == 0:
        return 0.0

    tail = returns_matrix[-lookback:] if len(returns_matrix) > lookback else returns_matrix
    portfolio_returns = tail @ weights
    var = float(-np.percentile(portfolio_returns, (1 - confidence) * 100))
    return round(max(var, 0.0), 6)


def pre_trade_check(
    proposed_weights: dict[str, float],
    limits: RiskLimits | None = None,
    sector_map: dict[str, str] | None = None,
    returns_matrix: np.ndarray | None = None,
    symbol_order: list[str] | None = None,
) -> CheckResult:
    """Hard pre-trade risk check. Returns CheckResult(passed=False) to BLOCK a trade.

    This is a hard gate — any violation must prevent order submission.
    Never downgrade a failing check to a warning in production.
    """
    if limits is None:
        limits = RiskLimits()

    violations: list[str] = []
    exposures = compute_exposures(proposed_weights, sector_map)

    # 1. Single-name position cap
    for sym, w in proposed_weights.items():
        if abs(w) > limits.max_position:
            violations.append(
                f"Position cap: {sym} weight {w:.1%} > limit {limits.max_position:.1%}"
            )

    # 2. Sector cap
    for sector, exp in exposures.sector_exposures.items():
        if exp > limits.max_sector:
            violations.append(
                f"Sector cap: {sector} exposure {exp:.1%} > limit {limits.max_sector:.1%}"
            )

    # 3. Gross exposure cap
    if exposures.gross_exposure > limits.max_gross:
        violations.append(
            f"Gross cap: {exposures.gross_exposure:.1%} > limit {limits.max_gross:.1%}"
        )

    # 4. Net exposure cap
    if abs(exposures.net_exposure) > limits.max_net:
        violations.append(
            f"Net cap: {exposures.net_exposure:.1%} > limit {limits.max_net:.1%}"
        )

    # 5. Top-5 concentration cap
    if exposures.top_5_concentration > limits.max_top5_concentration:
        violations.append(
            f"Concentration cap: top-5 {exposures.top_5_concentration:.1%} "
            f"> limit {limits.max_top5_concentration:.1%}"
        )

    # 6. VaR cap (only if returns data provided)
    if limits.var_limit_pct is not None and returns_matrix is not None and symbol_order is not None:
        w_vec = np.array([proposed_weights.get(s, 0.0) for s in symbol_order])
        var_95 = historical_var(returns_matrix, w_vec, confidence=0.95)
        if var_95 > limits.var_limit_pct:
            violations.append(
                f"VaR cap: 1-day 95% VaR {var_95:.1%} > limit {limits.var_limit_pct:.1%}"
            )

    return CheckResult(passed=len(violations) == 0, violations=violations)


def generate_risk_report(
    weights: dict[str, float],
    prices: pl.DataFrame,
    as_of: date | None = None,
    limits: RiskLimits | None = None,
    sector_map: dict[str, str] | None = None,
    stress_results: dict[str, float] | None = None,
) -> RiskReport:
    """Generate a full risk report for a set of portfolio weights.

    Parameters
    ----------
    weights      : {symbol: weight}
    prices       : [date, symbol, adj_close] — used for VaR returns history
    as_of        : Report date (today if None)
    limits       : Risk limits to check against
    sector_map   : Sector assignments (EQUITY_SECTOR_MAP if None)
    stress_results: Pre-computed stress scenario P&L {scenario: pct}
    """
    from portfolio.covariance import compute_returns

    if as_of is None:
        as_of = date.today()
    if limits is None:
        limits = RiskLimits()

    exposures = compute_exposures(weights, sector_map, as_of)

    # Build returns matrix aligned to weight symbols
    symbols_in_portfolio = [s for s in weights if abs(weights[s]) > 1e-6]
    prices_filtered = prices.filter(pl.col("symbol").is_in(symbols_in_portfolio))

    var_95, var_99 = 0.0, 0.0
    if not prices_filtered.is_empty():
        returns_matrix, sym_order = compute_returns(prices_filtered)
        w_vec = np.array([weights.get(s, 0.0) for s in sym_order])
        var_95 = historical_var(returns_matrix, w_vec, confidence=0.95)
        var_99 = historical_var(returns_matrix, w_vec, confidence=0.99)
        check = pre_trade_check(weights, limits, sector_map, returns_matrix, sym_order)
    else:
        check = pre_trade_check(weights, limits, sector_map)

    return RiskReport(
        as_of=as_of,
        var_1d_95=var_95,
        var_1d_99=var_99,
        exposures=exposures,
        check=check,
        stress_results=stress_results or {},
    )


def print_risk_report(report: RiskReport) -> None:
    """Print a formatted risk report to stdout."""
    e = report.exposures
    print(f"\n{'='*60}")
    print(f"  Risk Report — {report.as_of}")
    print(f"{'='*60}")
    print(f"  Positions       : {e.n_positions}")
    print(f"  Gross exposure  : {e.gross_exposure:.1%}")
    print(f"  Net exposure    : {e.net_exposure:.1%}")
    print(f"  Top-5 conc.     : {e.top_5_concentration:.1%}")
    print(f"  Largest name    : {e.largest_position[0]} ({e.largest_position[1]:.1%})")
    print(f"{'─'*60}")
    print(f"  1-day VaR 95%   : {report.var_1d_95:.2%}")
    print(f"  1-day VaR 99%   : {report.var_1d_99:.2%}")
    print(f"{'─'*60}")
    print("  Sector exposures:")
    for sector, exp in e.sector_exposures.items():
        bar = "#" * int(exp * 20)
        print(f"    {sector:<20} {exp:.1%}  {bar}")
    if report.stress_results:
        print(f"{'─'*60}")
        print("  Stress scenarios:")
        for scenario, pnl in report.stress_results.items():
            print(f"    {scenario:<25} {pnl:>+.1%}")
    print(f"{'─'*60}")
    print(f"  {report.check}")
    print(f"{'='*60}\n")
