"""Tests for the value-stack co-optimisation (DA + FCR + aFRR with PICASSO)."""

from __future__ import annotations

import numpy as np
import pytest

from bessopt.market.simulator import SLOTS_PER_BLOCK, cooptimise_day
from bessopt.optimiser.lp import solve_dispatch
from bessopt.optimiser.spec import BatterySpec

ASSET_1H = BatterySpec(power_mw=10.0, energy_mwh=10.0)
ASSET_2H = BatterySpec(power_mw=10.0, energy_mwh=20.0)
PRICES = np.tile([20.0, 20, 20, 20, 120, 120, 120, 120], 12)  # 96 slots, big DA spread


def _flat(value: float, n: int = 96) -> np.ndarray:
    return np.full(n, value)


def test_zero_reserve_prices_reduce_to_pure_arbitrage():
    """With no reserve value, the co-opt LP should commit no reserve and match
    the plain day-ahead optimiser."""
    res = cooptimise_day(PRICES, _flat(0.0), _flat(0.0), _flat(0.0), ASSET_2H)
    da_only = solve_dispatch(PRICES, ASSET_2H, soc_initial_mwh=ASSET_2H.soc_min_mwh)
    assert res.revenue_by_stream["fcr"] == pytest.approx(0.0, abs=1e-6)
    assert res.revenue_by_stream["afrr_pos"] == pytest.approx(0.0, abs=1e-6)
    assert res.revenue_by_stream["day_ahead"] == pytest.approx(da_only.revenue_eur, rel=1e-4)


def test_high_reserve_prices_pull_capacity_in():
    """When reserve pays well, the optimiser commits reserve MW."""
    res = cooptimise_day(PRICES, _flat(50.0), _flat(50.0), _flat(50.0), ASSET_2H)
    assert res.r_fcr_mw.max() + res.r_afrr_pos_mw.max() + res.r_afrr_neg_mw.max() > 1.0
    assert res.revenue_by_stream["fcr"] > 0


def test_reserve_constant_within_block():
    res = cooptimise_day(PRICES, _flat(20.0), _flat(20.0), _flat(20.0), ASSET_2H)
    for arr in (res.r_fcr_mw, res.r_afrr_pos_mw, res.r_afrr_neg_mw):
        blocks = arr.reshape(-1, SLOTS_PER_BLOCK)
        assert np.allclose(blocks, blocks[:, [0]])  # each 4h block holds one MW level


def test_1h_tilts_to_reserve_more_than_2h():
    """The PICASSO story: with attractive reserve prices, the 1h battery leans on
    reserve (it can barely arbitrage), the 2h keeps a larger day-ahead share."""
    fcr, pos, neg = _flat(25.0), _flat(25.0), _flat(15.0)
    r1 = cooptimise_day(PRICES, fcr, pos, neg, ASSET_1H)
    r2 = cooptimise_day(PRICES, fcr, pos, neg, ASSET_2H)

    def reserve_share(r):
        res_rev = r.revenue_by_stream["fcr"] + r.revenue_by_stream["afrr_pos"] \
            + r.revenue_by_stream["afrr_neg"]
        return res_rev / r.total_revenue_eur

    assert reserve_share(r1) > reserve_share(r2)
    # And the 2h captures more day-ahead revenue in absolute terms.
    assert r2.revenue_by_stream["day_ahead"] > r1.revenue_by_stream["day_ahead"]


def test_soc_respects_dynamic_buffer():
    res = cooptimise_day(PRICES, _flat(20.0), _flat(20.0), _flat(20.0), ASSET_2H)
    # Floor raised by POS aFRR + FCR buffers; ceiling lowered by NEG aFRR + FCR.
    floor = ASSET_2H.soc_min_mwh + 1.0 * res.r_afrr_pos_mw + 0.25 * res.r_fcr_mw
    ceil = ASSET_2H.soc_max_mwh - 1.0 * res.r_afrr_neg_mw - 0.25 * res.r_fcr_mw
    assert (res.soc_mwh[1:] >= floor - 1e-6).all()
    assert (res.soc_mwh[1:] <= ceil + 1e-6).all()
