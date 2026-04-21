"""Signal scanner — current factor rankings and universe snapshot.

Pure functions: (features_df) -> structured results.
No I/O, no Streamlit, no Plotly. Usable from notebooks.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

FEATURE_LABELS: dict[str, str] = {
    "momentum_12m_1m":   "12-1 Momentum",
    "realized_vol_21d":  "Realized Vol (21d)",
    "log_return_1d":     "1d Log Return",
    "dollar_volume_63d": "Dollar Volume (63d)",
    "reversal_5d":       "5d Reversal",
}
ALL_FEATURES: list[str] = list(FEATURE_LABELS.keys())

# Features where higher z-score is "better" for display colouring
HIGHER_IS_BETTER: set[str] = {"momentum_12m_1m", "reversal_5d", "dollar_volume_63d"}

# Features whose tick format should be percentage
PCT_FEATURES: set[str] = {"momentum_12m_1m", "reversal_5d", "realized_vol_21d", "log_return_1d"}


@dataclass
class SnapshotResult:
    """Output of get_snapshot()."""
    latest_date: str
    snap_pd: pd.DataFrame         # symbol-indexed, raw factor values
    z_scores: pd.DataFrame        # same index/columns, z-score normalised
    present_features: list[str]   # features that exist in data
    n_universe: int
    n_valid_momentum: int
    top5_momentum: list[str]      # symbols ranked 1-5 by momentum


def get_snapshot(features_df: pl.DataFrame) -> SnapshotResult:
    """Return the latest cross-section of all factor values for the universe.

    Parameters
    ----------
    features_df : wide Polars DataFrame [date, symbol, feature1, ...]

    Returns
    -------
    SnapshotResult with raw values and cross-sectionally z-scored values.
    """
    latest_date = features_df["date"].max()
    snap = features_df.filter(pl.col("date") == latest_date)

    present = [f for f in ALL_FEATURES if f in snap.columns]
    snap_pd = snap.select(["symbol"] + present).to_pandas().set_index("symbol")

    z_df = zscore_normalize(snap_pd[present])

    n_valid_mom = int(snap_pd["momentum_12m_1m"].notna().sum()) if "momentum_12m_1m" in snap_pd.columns else 0
    top5 = (
        snap_pd["momentum_12m_1m"].dropna().nlargest(5).index.tolist()
        if "momentum_12m_1m" in snap_pd.columns else []
    )

    return SnapshotResult(
        latest_date=str(latest_date),
        snap_pd=snap_pd,
        z_scores=z_df,
        present_features=present,
        n_universe=len(snap_pd),
        n_valid_momentum=n_valid_mom,
        top5_momentum=top5,
    )


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally z-score each column. Leaves NaN intact."""
    result = df.copy().astype(float)
    for col in result.columns:
        mu    = result[col].mean()
        sigma = result[col].std()
        if sigma > 1e-10:
            result[col] = (result[col] - mu) / sigma
        else:
            result[col] = np.nan
    return result


def momentum_ranked(snap_pd: pd.DataFrame) -> pd.Series:
    """Return the momentum column sorted ascending (bottom → top)."""
    if "momentum_12m_1m" not in snap_pd.columns:
        return pd.Series(dtype=float)
    return snap_pd["momentum_12m_1m"].dropna().sort_values()
