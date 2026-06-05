"""Hand-checkable invariants for the dispatch LP (`solve_dispatch`).

These are the cardinal correctness tests for the heart of the optimiser. If any
fails, every downstream revenue number is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from bessopt.optimiser.lp import settle, solve_dispatch
from bessopt.optimiser.spec import BatterySpec

# Lossless, hourly, full-band 1h asset — makes arithmetic exact.
LOSSLESS = BatterySpec(
    power_mw=10.0, energy_mwh=10.0, eta_rt=1.0,
    soc_min_frac=0.0, soc_max_frac=1.0, slot_hours=1.0,
)
# Realistic default 1h / 2h assets (15-min, 90% RTE, 10-90% band).
ASSET_1H = BatterySpec(power_mw=10.0, energy_mwh=10.0)
ASSET_2H = BatterySpec(power_mw=10.0, energy_mwh=20.0)


def test_closed_form_two_cycles():
    """[10,100,10,100] lossless ⇒ charge low / discharge high, revenue = usable·spread·cycles."""
    prices = np.array([10.0, 100.0, 10.0, 100.0])
    res = solve_dispatch(prices, LOSSLESS, soc_initial_mwh=0.0)
    assert res.status == "optimal"
    # Two full 10 MWh cycles at a 90 €/MWh spread.
    assert res.revenue_eur == pytest.approx(1800.0, abs=1e-6)
    assert res.charge_mw == pytest.approx([10, 0, 10, 0], abs=1e-6)
    assert res.discharge_mw == pytest.approx([0, 10, 0, 10], abs=1e-6)
    assert res.soc_mwh == pytest.approx([0, 10, 0, 10, 0], abs=1e-6)


@pytest.mark.parametrize("spec", [ASSET_1H, ASSET_2H])
def test_soc_stays_in_band(spec):
    rng = np.random.default_rng(1)
    for _ in range(5):
        prices = rng.normal(80, 50, size=96)
        res = solve_dispatch(prices, spec)
        assert (res.soc_mwh >= spec.soc_min_mwh - 1e-6).all()
        assert (res.soc_mwh <= spec.soc_max_mwh + 1e-6).all()


@pytest.mark.parametrize("spec", [ASSET_1H, ASSET_2H])
def test_energy_balance_holds(spec):
    rng = np.random.default_rng(2)
    prices = rng.normal(70, 60, size=96)
    res = solve_dispatch(prices, spec)
    dt = spec.slot_hours
    for t in range(len(prices)):
        flow = spec.eta_c * res.charge_mw[t] * dt - res.discharge_mw[t] * dt / spec.eta_d
        assert res.soc_mwh[t + 1] - res.soc_mwh[t] == pytest.approx(flow, abs=1e-7)


def test_no_simultaneous_charge_discharge_when_lossy():
    rng = np.random.default_rng(3)
    prices = np.abs(rng.normal(80, 40, size=96)) + 1.0  # strictly positive
    res = solve_dispatch(prices, ASSET_1H)
    overlap = (res.charge_mw > 1e-6) & (res.discharge_mw > 1e-6)
    assert not overlap.any()


def test_flat_prices_zero_revenue():
    prices = np.full(96, 50.0)
    res = solve_dispatch(prices, ASSET_1H)
    assert res.revenue_eur == pytest.approx(0.0, abs=1e-6)
    assert res.throughput_mwh == pytest.approx(0.0, abs=1e-6)


def test_2h_beats_1h_same_prices():
    rng = np.random.default_rng(4)
    prices = rng.normal(80, 50, size=96)
    r1 = solve_dispatch(prices, ASSET_1H)
    r2 = solve_dispatch(prices, ASSET_2H)
    assert r2.revenue_eur >= r1.revenue_eur - 1e-6


def test_settle_matches_solve_on_same_prices():
    """Settling a schedule against the prices it was optimised on == the LP revenue."""
    rng = np.random.default_rng(5)
    prices = rng.normal(80, 50, size=96)
    res = solve_dispatch(prices, ASSET_1H)
    settled = settle(res.charge_mw, res.discharge_mw, prices, ASSET_1H)
    assert settled == pytest.approx(res.revenue_eur, abs=1e-6)


def test_terminal_soc_constraint_respected():
    rng = np.random.default_rng(6)
    prices = rng.normal(80, 50, size=96)
    target = ASSET_2H.soc_min_mwh + 5.0
    res = solve_dispatch(prices, ASSET_2H, soc_terminal_mwh=target)
    assert res.soc_mwh[-1] == pytest.approx(target, abs=1e-6)
