"""One-shot script to generate snapshot fixture files for feature tests.

Run this once (or whenever canonical feature logic intentionally changes):
    uv run python tests/generate_fixtures.py

This writes tests/fixtures/feature_snapshot.parquet — a pinned, known-good
output that test_features.py compares against on every future test run.
Any deviation means either a bug or an intentional change (in which case
re-run this script and commit the updated fixture).
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# Make sure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.canonical import compute_features

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SNAPSHOT_PATH = FIXTURES_DIR / "feature_snapshot.parquet"

# Deterministic seed input — fixed prices, no randomness
SEED_START = date(2020, 1, 2)
N_ROWS = 320
SYMBOLS = ["SEED_A", "SEED_B"]
DAILY_DRIFTS = {"SEED_A": 0.0008, "SEED_B": -0.0003}


def _make_seed_bars() -> pl.DataFrame:
    frames = []
    for sym, drift in DAILY_DRIFTS.items():
        dates = [SEED_START + timedelta(days=i) for i in range(N_ROWS)]
        prices = [100.0 * ((1 + drift) ** i) for i in range(N_ROWS)]
        # Inject a known split on day 150 to verify adjustment handling is isolated
        volume = [1_000_000.0 + i * 100 for i in range(N_ROWS)]
        frames.append(pl.DataFrame({
            "date": dates,
            "symbol": [sym] * N_ROWS,
            "adj_close": prices,
            "volume": volume,
        }).with_columns(pl.col("date").cast(pl.Date)))
    return pl.concat(frames)


def main() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    bars = _make_seed_bars()
    snapshot = compute_features(bars)
    snapshot.write_parquet(SNAPSHOT_PATH, compression="snappy")
    print(f"Fixture written: {SNAPSHOT_PATH}")
    print(f"  Rows: {len(snapshot)}, Columns: {snapshot.columns}")


if __name__ == "__main__":
    main()
