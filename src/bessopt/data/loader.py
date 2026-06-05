"""Data loading + Berlin-day window helpers.

The canonical dataset is the merged DE-LU 15-min parquet produced by
`bessopt.data.refresh`. Issue time is fixed at D-1 12:00 Europe/Berlin (the
EPEX day-ahead gate); the target index is the 96 quarter-hours of the Berlin
calendar delivery day.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

BERLIN = "Europe/Berlin"
GATE_HOUR_LOCAL = 12  # German day-ahead auction gate closure: D-1 12:00 CET/CEST
DEFAULT_PARQUET = Path("data/de_lu_15min.parquet")


def load_de_lu_15min(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Load the merged DE-LU 15-min parquet with a tz-aware UTC datetime index."""
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    return df


# Backwards-compatible alias — vendored forecast/backtest code referenced the
# old name; both resolve to the same loader.
load_smard_15min = load_de_lu_15min


def issue_time_for(delivery_date: date) -> pd.Timestamp:
    """Forecast issue time = D-1 12:00 Europe/Berlin, returned as UTC."""
    d_minus_1 = pd.Timestamp(delivery_date, tz=BERLIN) - pd.Timedelta(days=1)
    issue_local = d_minus_1.replace(hour=GATE_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    return issue_local.tz_convert("UTC")


def target_index_for(delivery_date: date) -> pd.DatetimeIndex:
    """The 96 quarter-hour UTC timestamps covering Berlin calendar day `delivery_date`."""
    start_local = pd.Timestamp(delivery_date, tz=BERLIN)
    end_local = start_local + pd.Timedelta(days=1)
    idx_local = pd.date_range(start_local, end_local, freq="15min", inclusive="left")
    return idx_local.tz_convert("UTC")


def slice_history(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Return rows strictly before issue_time. Defensive copy."""
    return df.loc[df.index < issue_time].copy()
