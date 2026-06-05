"""Baseline price predictors — the floor the forecast must beat.

For day-ahead prices there is no public operational baseline (unlike load, which
has the TSO forecast), so the comparison is the **seasonal-naive**: the realised
price exactly one week earlier at the same quarter-hour. Prices from D-7 cleared
long before the D-1 12:00 issue time, so this is leakage-safe.

Each predictor matches the forecast interface `(df, issue_time) -> DataFrame`
with columns p10/p50/p90, so it is a drop-in for the replay engine. The naive
baseline returns degenerate quantiles (p10 == p50 == p90).
"""

from __future__ import annotations

import pandas as pd

from ..data.loader import target_index_for

DE_PRICE = "price__germany_luxembourg"


def _delivery_target_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    return target_index_for(delivery_local.date())


def seasonal_naive_price_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    price_col: str = DE_PRICE,
    lag_days: int = 7,
) -> pd.DataFrame:
    """Predict each quarter-hour with the realised price `lag_days` earlier."""
    target_idx = _delivery_target_index(issue_time)
    lagged_idx = target_idx - pd.Timedelta(days=lag_days)
    vals = df[price_col].reindex(lagged_idx).to_numpy()
    out = pd.DataFrame({"p10": vals, "p50": vals, "p90": vals}, index=target_idx)
    out.index.name = "target_ts"
    return out


__all__ = ["seasonal_naive_price_predict"]
