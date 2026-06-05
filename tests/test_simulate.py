"""Tests for the interactive BESS what-if simulation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bessopt.data.loader import load_de_lu_15min, target_index_for
from bessopt.optimiser.spec import BatterySpec
from bessopt.simulate import default_window, simulate_asset

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="parquet not present.")

SPEC = BatterySpec(power_mw=10.0, energy_mwh=20.0)


@pytest.fixture(scope="module")
def df():
    return load_de_lu_15min(PARQUET)


def test_default_window_ends_at_complete_day(df):
    start, end = default_window(df, 10)
    assert (end - start).days == 10
    assert not df["price__germany_luxembourg"].reindex(target_index_for(end)).isna().any()


def test_simulation_returns_sane_result(df):
    start, end = default_window(df, 8)
    sim = simulate_asset(SPEC, df, start, end)
    assert sim.oracle_eur_per_mw_yr > 0
    assert 0 < sim.avg_cycles_per_day < 10
    assert sim.value_stack_eur_per_mw_yr is not None
    assert set(sim.value_stack_eur_per_mw_yr) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
    assert len(sim.best_day_dispatch.soc_mwh) > 1


def test_forecast_leg_returns_realistic_vcr(df):
    start, end = default_window(df, 8)
    sim = simulate_asset(SPEC, df, start, end, with_value_stack=False, with_forecast=True)
    assert sim.forecast_eur_per_mw_yr is not None
    assert 0.0 <= sim.vcr <= 1.0
    assert sim.forecast_eur_per_mw_yr > 0


def test_higher_degradation_cost_reduces_cycling_and_revenue(df):
    start, end = default_window(df, 8)
    cheap = simulate_asset(replace(SPEC, deg_cost_eur_per_mwh=2.0), df, start, end,
                           with_value_stack=False)
    dear = simulate_asset(replace(SPEC, deg_cost_eur_per_mwh=40.0), df, start, end,
                          with_value_stack=False)
    assert dear.avg_cycles_per_day < cheap.avg_cycles_per_day
    assert dear.oracle_eur_per_mw_yr < cheap.oracle_eur_per_mw_yr
