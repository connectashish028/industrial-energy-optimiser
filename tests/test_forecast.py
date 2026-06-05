"""Smoke tests for the vendored XGBoost price forecaster."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bessopt.data.loader import issue_time_for, load_de_lu_15min, target_index_for
from bessopt.forecast.predict import DEFAULT_XGB_PRICE_DIR, xgboost_price_predict_full

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists() or not DEFAULT_XGB_PRICE_DIR.exists(),
    reason="parquet or price model checkpoint not present.",
)


@pytest.fixture(scope="module")
def df():
    return load_de_lu_15min(PARQUET)


def test_price_forecast_shape_and_monotone(df):
    fc = xgboost_price_predict_full(df, issue_time_for(date(2026, 2, 15)))
    target = target_index_for(date(2026, 2, 15))
    assert list(fc.columns) == ["p10", "p50", "p90"]
    assert len(fc) == len(target)
    assert fc.index.equals(target)
    # Quantiles are sorted per row.
    assert (fc["p10"] <= fc["p50"] + 1e-9).all()
    assert (fc["p50"] <= fc["p90"] + 1e-9).all()
    # Sane price range — no all-NaN / runaway values on a normal winter day.
    assert fc["p50"].notna().all()
    assert fc["p50"].between(-500, 1000).all()


def test_forecast_only_sees_pre_issue_data(df):
    """Corrupting each column *beyond its own availability cutoff* must not change
    the forecast — the predictor routes through the same point-in-time gate as the
    features. (TSO day-ahead forecasts for the delivery day are legitimately known
    at issue time, so they are NOT corrupted — only data past each column's rule.)"""
    import numpy as np

    from bessopt.features.availability import RULES, classify_column

    delivery = date(2026, 1, 20)
    issue = issue_time_for(delivery)
    clean = xgboost_price_predict_full(df, issue)

    corrupt = df.copy()
    rng = np.random.default_rng(0)
    for col in corrupt.columns:
        cutoff = issue + RULES[classify_column(col)].max_age_offset
        mask = corrupt.index >= cutoff
        if mask.any():
            corrupt.loc[mask, col] = rng.normal(1e6, 1e6, size=int(mask.sum()))
    dirty = xgboost_price_predict_full(corrupt, issue)

    assert np.allclose(clean.to_numpy(), dirty.to_numpy(), equal_nan=True)
