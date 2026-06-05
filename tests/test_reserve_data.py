"""Reserve-market data wiring: availability classification + representative prices."""

from __future__ import annotations

import pandas as pd

from bessopt.data.sources.regelleistung import (
    RESERVE_COLUMNS,
    fetch,
    reserve_block_prices,
)
from bessopt.features.availability import RULES, classify_column


def test_reserve_columns_classified_and_known_at_issue():
    for col in RESERVE_COLUMNS:
        assert classify_column(col) == "reserve"
    # Reserve clears D-1 morning ⇒ known at the D-1 12:00 issue (48h forecast-like window).
    assert RULES["reserve"].max_age_offset == pd.Timedelta(hours=48)
    # Imbalance/activated energy is settlement-lagged ⇒ never known ahead.
    assert classify_column("imbalance_rebap__price") == "imbalance"
    assert RULES["imbalance"].max_age_offset == pd.Timedelta(0)


def test_representative_block_prices_shape_and_sign():
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-02-08", tz="UTC")
    blocks = reserve_block_prices(start, end)
    assert list(blocks.columns) == list(RESERVE_COLUMNS)
    assert (blocks.to_numpy() > 0).all()
    assert len(blocks) >= 7 * 6  # ~6 four-hour blocks/day over 7 days (tz boundary slack)
    # POS aFRR is dearer than NEG on average (typical German asymmetry).
    assert (blocks["reserve_afrr_cap__pos__capacity_price"].mean()
            > blocks["reserve_afrr_cap__neg__capacity_price"].mean())


def test_fetch_expands_blocks_to_15min():
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-02-02", tz="UTC")

    class _Col:
        name = "reserve_fcr__capacity_price"

    s = fetch(_Col(), start, end)
    assert len(s) == 96  # one UTC day at 15-min
    assert (s > 0).all()
    # Prices are piecewise-constant over Berlin 4h blocks (≤6/day, +1 for the
    # UTC/Berlin offset straddling a boundary) — not 96 distinct values.
    assert s.nunique() <= 7
