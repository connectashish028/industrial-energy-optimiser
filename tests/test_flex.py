"""Tests for the industrial flexibility MILP cost optimiser."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bessopt.flex import ConsumerSpec, optimise_day, run_flex
from bessopt.flex.pv import pv_available

SPEC = ConsumerSpec()
# Cheap nights / expensive days, 96 slots.
SPOT = np.tile([30.0, 30, 30, 30, 30, 30, 30, 30, 150, 150, 150, 150, 150, 150, 150, 150], 6)
PV = np.zeros(96)


def test_pv_available_bounds():
    idx = pd.date_range("2026-01-01", periods=48, freq="15min", tz="UTC")
    df = pd.DataFrame({"weather__shortwave_radiation": np.linspace(0, 1200, 48)}, index=idx)
    pv = pv_available(df, idx, capacity_mwp=10.0)
    assert (pv >= 0).all()
    assert (pv <= 10.0 + 1e-9).all()           # capped at capacity
    assert pv[0] == 0.0                          # no sun, no power


def test_process_meets_runtime_and_respects_grid_limit():
    res = optimise_day(SPOT, PV, SPEC, soc_init=SPEC.battery.soc_min_mwh)
    assert res.status == "optimal"
    assert res.proc_on.sum() >= SPEC.proc_slots - 1e-6        # runs its required hours
    assert (res.grid_mw >= -1e-6).all()
    assert (res.grid_mw <= SPEC.grid_limit_mw + 1e-6).all()


def test_energy_balance_holds():
    res = optimise_day(SPOT, PV, SPEC, soc_init=SPEC.battery.soc_min_mwh)
    lhs = (res.baseline_mw + SPEC.proc_power_mw * res.proc_on
           + res.charge_mw - res.discharge_mw - res.pv_used_mw)
    assert np.allclose(res.grid_mw, lhs, atol=1e-6)
    # SoC stays in band.
    b = SPEC.battery
    assert (res.soc_mwh >= b.soc_min_mwh - 1e-6).all()
    assert (res.soc_mwh <= b.soc_max_mwh + 1e-6).all()


def test_process_shifts_to_cheap_hours():
    """With clearly cheap nights, most run-hours land in the cheap half."""
    res = optimise_day(SPOT, PV, SPEC, soc_init=SPEC.battery.soc_min_mwh)
    cheap = SPOT <= 100
    on_in_cheap = res.proc_on[cheap].sum()
    assert on_in_cheap >= 0.8 * res.proc_on.sum()


@pytest.mark.skipif(not Path("data/de_lu_15min.parquet").exists(), reason="parquet not present.")
def test_optimised_beats_naive_on_real_data():
    from bessopt.data.loader import load_de_lu_15min

    df = load_de_lu_15min()
    run = run_flex(df, SPEC, date(2026, 2, 1), date(2026, 2, 10))
    assert run.savings_eur >= -1.0                 # optimisation never costs more
    assert run.optimised_cost_eur <= run.naive_cost_eur + 1.0
    assert run.savings_eur_per_year > 0            # a flexible site saves money
