"""Tests for the simplified intraday (IDC) two-market value of access."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bessopt.data.loader import load_de_lu_15min
from bessopt.data.sources.intraday import intraday_index
from bessopt.market.intraday import run_intraday_value, solve_two_market_dispatch
from bessopt.optimiser.lp import solve_dispatch
from bessopt.optimiser.spec import BatterySpec

SPEC = BatterySpec(power_mw=10.0, energy_mwh=20.0)
PRICES = np.tile([20.0, 20, 20, 20, 120, 120, 120, 120], 12)


def test_intraday_index_is_da_when_spread_zero():
    assert np.allclose(intraday_index(PRICES, rms_spread_eur=0.0), PRICES)


def test_intraday_index_has_target_rms_spread():
    da = np.full(96, 50.0)
    dev = intraday_index(da, rms_spread_eur=12.0, seed=0) - da
    assert np.sqrt(np.mean(dev**2)) == pytest.approx(12.0, rel=1e-6)
    assert abs(dev.mean()) < 5.0          # roughly zero-mean (no systematic DA-ID bias)


def test_two_market_reduces_to_da_only_when_id_equals_da():
    tm = solve_two_market_dispatch(PRICES, PRICES, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    da = solve_dispatch(PRICES, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    assert tm.revenue_eur == pytest.approx(da.revenue_eur, rel=1e-4)


def test_two_market_never_worse_than_da_only():
    idp = intraday_index(PRICES, rms_spread_eur=15.0, seed=3)
    tm = solve_two_market_dispatch(PRICES, idp, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    da = solve_dispatch(PRICES, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    assert tm.revenue_eur >= da.revenue_eur - 1e-6


def test_two_market_soc_in_band():
    idp = intraday_index(PRICES, rms_spread_eur=20.0, seed=1)
    tm = solve_two_market_dispatch(PRICES, idp, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    assert (tm.soc_mwh >= SPEC.soc_min_mwh - 1e-6).all()
    assert (tm.soc_mwh <= SPEC.soc_max_mwh + 1e-6).all()


@pytest.mark.skipif(not Path("data/de_lu_15min.parquet").exists(), reason="parquet not present.")
def test_uplift_grows_with_spread_on_real_data():
    df = load_de_lu_15min()
    s, e = date(2026, 2, 1), date(2026, 2, 10)
    lo = run_intraday_value(df, SPEC, s, e, rms_spread_eur=5.0)
    hi = run_intraday_value(df, SPEC, s, e, rms_spread_eur=20.0)
    assert lo.uplift_eur_per_mw_yr >= -1.0
    assert hi.uplift_eur_per_mw_yr > lo.uplift_eur_per_mw_yr   # wider spread ⇒ more intraday value
