"""Data quality validators run after every ingestion batch.

Each check returns a ValidationResult. The orchestrator collects all results,
logs them, and alerts on any failures. Never silent-fix — we want to see
vendor quality degrade over time.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl


@dataclass
class ValidationResult:
    symbol: str
    check: str
    passed: bool
    message: str = ""
    severity: str = "warning"   # "warning" | "error"


def check_row_counts(
    df: pl.DataFrame,
    expected_trading_days: int | None = None,
) -> ValidationResult:
    """Flag if row count deviates significantly from expected trading-day count."""
    n = len(df)
    symbol = df["symbol"][0] if "symbol" in df.columns and n > 0 else "unknown"

    if n == 0:
        return ValidationResult(symbol, "row_counts", False, "Empty DataFrame", "error")

    if expected_trading_days is not None:
        ratio = n / expected_trading_days
        if ratio < 0.8:
            return ValidationResult(
                symbol, "row_counts", False,
                f"Only {n}/{expected_trading_days} expected rows ({ratio:.0%})", "warning",
            )

    return ValidationResult(symbol, "row_counts", True, f"{n} rows")


def check_null_rate(df: pl.DataFrame, threshold: float = 0.01) -> list[ValidationResult]:
    """Flag columns where null rate exceeds threshold."""
    symbol = df["symbol"][0] if "symbol" in df.columns and len(df) > 0 else "unknown"
    results = []
    for col in ["open", "high", "low", "close", "volume", "adj_close"]:
        if col not in df.columns:
            continue
        null_rate = df[col].is_null().mean()
        if null_rate > threshold:
            results.append(ValidationResult(
                symbol, f"null_rate:{col}", False,
                f"{col} null rate {null_rate:.1%} > {threshold:.1%}", "warning",
            ))
    if not results:
        results.append(ValidationResult(symbol, "null_rate", True, "All columns within threshold"))
    return results


def check_price_jumps(df: pl.DataFrame, sigma_threshold: float = 8.0) -> ValidationResult:
    """Flag days with price moves exceeding sigma_threshold standard deviations."""
    symbol = df["symbol"][0] if "symbol" in df.columns and len(df) > 0 else "unknown"

    if "adj_close" not in df.columns or len(df) < 10:
        return ValidationResult(symbol, "price_jumps", True, "Not enough data to check")

    prices = df["adj_close"].drop_nulls().to_numpy()
    log_rets = np.diff(np.log(prices[prices > 0]))
    if len(log_rets) == 0:
        return ValidationResult(symbol, "price_jumps", True, "No returns to check")

    std = log_rets.std()
    if std == 0:
        return ValidationResult(symbol, "price_jumps", True, "Zero variance — static price")

    jumps = np.abs(log_rets) > sigma_threshold * std
    if jumps.any():
        n_jumps = jumps.sum()
        return ValidationResult(
            symbol, "price_jumps", False,
            f"{n_jumps} day(s) with >{sigma_threshold}σ move — review manually", "warning",
        )
    return ValidationResult(symbol, "price_jumps", True, f"No jumps >{sigma_threshold}σ")


def check_staleness(df: pl.DataFrame, max_age_days: int = 3) -> ValidationResult:
    """Flag if most recent bar is older than max_age_days trading days."""
    symbol = df["symbol"][0] if "symbol" in df.columns and len(df) > 0 else "unknown"

    if "date" not in df.columns or len(df) == 0:
        return ValidationResult(symbol, "staleness", False, "No date column or empty", "error")

    latest = df["date"].max()
    age = (date.today() - latest).days

    # Allow up to max_age_days + weekend buffer
    threshold = max_age_days + 3
    if age > threshold:
        return ValidationResult(
            symbol, "staleness", False,
            f"Last bar is {age} days old (latest={latest})", "warning",
        )
    return ValidationResult(symbol, "staleness", True, f"Last bar {age} days old")


def validate_ingestion(df: pl.DataFrame) -> list[ValidationResult]:
    """Run all standard checks on a freshly ingested DataFrame.

    Returns a flat list of ValidationResult objects.
    """
    results: list[ValidationResult] = []
    results.append(check_row_counts(df))
    results.extend(check_null_rate(df))
    results.append(check_price_jumps(df))
    results.append(check_staleness(df))
    return results


def has_failures(results: list[ValidationResult], severity: str = "error") -> bool:
    return any(not r.passed and r.severity == severity for r in results)


def format_results(results: list[ValidationResult]) -> str:
    lines = []
    for r in results:
        icon = "✓" if r.passed else "✗"
        lines.append(f"  {icon} [{r.severity.upper()}] {r.symbol}/{r.check}: {r.message}")
    return "\n".join(lines)
