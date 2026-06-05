"""PICASSO buffer mechanics — the constraint that drives the 1h-vs-2h story.

The headline analytical result of the whole project, made into tests: a 1h
battery that commits aFRR consumes its usable energy as the 60-min buffer and
can no longer arbitrage, while a 2h battery reserves the same MW *and* still
cycles. This emerges from the bounds, not from any assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from bessopt.market.products import ReserveCommitment
from bessopt.optimiser.lp import solve_dispatch
from bessopt.optimiser.spec import BatterySpec

ASSET_1H = BatterySpec(power_mw=10.0, energy_mwh=10.0)
ASSET_2H = BatterySpec(power_mw=10.0, energy_mwh=20.0)
# A volatile day with clear arbitrage opportunity.
PRICES = np.tile([20.0, 20, 20, 20, 120, 120, 120, 120], 12)  # 96 slots, big spread


def _afrr(n_slots: int, mw: float) -> ReserveCommitment:
    """Constant symmetric aFRR commitment of `mw` across all slots."""
    blocks = np.full(6, mw)
    return ReserveCommitment.from_blocks(n_slots, 0.25, afrr_pos_blocks=blocks,
                                         afrr_neg_blocks=blocks)


def test_buffer_math_matches_picasso_rules():
    rc = _afrr(96, 5.0)
    # POS aFRR raises the floor by mw * 60min; NEG lowers the ceiling likewise.
    assert rc.soc_lower(1.0)[0] == pytest.approx(1.0 + 5.0 * 1.0)   # 1 + 5 MWh
    assert rc.soc_upper(9.0)[0] == pytest.approx(9.0 - 5.0 * 1.0)   # 9 - 5 MWh
    # Reserve also eats arbitrage power headroom.
    assert rc.discharge_cap(10.0)[0] == pytest.approx(5.0)
    assert rc.charge_cap(10.0)[0] == pytest.approx(5.0)


def test_1h_battery_cannot_arbitrage_while_holding_afrr():
    """5 MW symmetric aFRR ⇒ 1h usable band [1+5, 9-5] = [6, 4] = empty ⇒ infeasible."""
    rc = _afrr(96, 5.0)
    with pytest.raises(ValueError, match="Infeasible reserve commitment"):
        solve_dispatch(PRICES, ASSET_1H, reserve_commit=rc)


def test_1h_battery_throughput_collapses_with_afrr():
    """Even a feasible aFRR level nearly freezes a 1h battery's arbitrage."""
    free = solve_dispatch(PRICES, ASSET_1H)
    held = solve_dispatch(PRICES, ASSET_1H, reserve_commit=_afrr(96, 3.5))  # band [4.5, 5.5]
    assert held.throughput_mwh < 0.20 * free.throughput_mwh


def test_2h_battery_still_cycles_while_holding_afrr():
    """The same 5 MW aFRR leaves the 2h battery a usable band [2+5, 18-5] = [7, 13]."""
    free = solve_dispatch(PRICES, ASSET_2H)
    held = solve_dispatch(PRICES, ASSET_2H, reserve_commit=_afrr(96, 5.0))
    assert held.status == "optimal"
    assert held.throughput_mwh > 0.40 * free.throughput_mwh   # retains meaningful cycling
    # SoC respects the tightened band.
    assert (held.soc_mwh >= 7.0 - 1e-6).all()
    assert (held.soc_mwh <= 13.0 + 1e-6).all()


def test_no_reserve_is_unchanged():
    """reserve_commit=None must reproduce the plain dispatch exactly (no regression)."""
    a = solve_dispatch(PRICES, ASSET_2H)
    b = solve_dispatch(PRICES, ASSET_2H, reserve_commit=None)
    assert a.revenue_eur == pytest.approx(b.revenue_eur)
