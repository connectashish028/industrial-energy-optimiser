"""Tests for the procurement strategy analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bessopt.data.loader import load_de_lu_15min
from bessopt.procurement import procurement_analysis

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="parquet not present.")


@pytest.fixture(scope="module")
def df():
    return load_de_lu_15min(PARQUET)


def test_spot_cheaper_than_fixed_and_shapes(df):
    r = procurement_analysis(df, 12.0, date(2026, 1, 1), date(2026, 3, 31))
    assert r.spot_eur_per_year < r.fixed_eur_per_year         # fixed pays the premium
    assert r.cost_per_year[0] == pytest.approx(r.spot_eur_per_year, rel=1e-6)   # ratio 0 = spot
    assert r.cost_per_year[-1] == pytest.approx(r.fixed_eur_per_year, rel=1e-6)  # ratio 1 = fixed
    assert len(r.ratios) == len(r.cost_per_year) == len(r.risk_eur)


def test_risk_falls_to_zero_when_fully_fixed(df):
    r = procurement_analysis(df, 12.0, date(2026, 1, 1), date(2026, 3, 31))
    # Risk (weekly unit-price volatility) decreases monotonically and ~0 at fully fixed.
    assert np.all(np.diff(r.risk_eur) <= 1e-6)
    assert r.risk_eur[-1] == pytest.approx(0.0, abs=1e-6)
    assert r.risk_eur[0] > 0


def test_recommendation_in_range_and_reduces_risk(df):
    r = procurement_analysis(df, 12.0, date(2026, 1, 1), date(2026, 3, 31))
    assert 0.0 <= r.recommended_ratio <= 1.0
    ri = int(np.argmin(np.abs(r.ratios - r.recommended_ratio)))
    assert r.risk_eur[ri] <= 0.25 * r.risk_eur[0] + 1e-9      # cuts volatility to ≤25%
