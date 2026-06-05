"""Tests for the rolling-horizon MPC loop."""

from __future__ import annotations

from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from bessopt.backtest.mpc import actual_horizon, run_mpc
from bessopt.backtest.oracle import run_oracle
from bessopt.data.loader import load_de_lu_15min
from bessopt.optimiser.spec import BatterySpec

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="parquet not present.")

SPEC = BatterySpec(power_mw=10.0, energy_mwh=20.0)
PRICE = "price__germany_luxembourg"
# February window — no DST transition, so 24h == 96 fixed slots aligns with Berlin days.
START, END = date(2026, 2, 1), date(2026, 2, 14)


@pytest.fixture(scope="module")
def df():
    return load_de_lu_15min(PARQUET)


def test_mpc_24h_perfect_matches_oracle(df):
    """MPC with horizon=step=24h under perfect foresight == the daily oracle."""
    act = partial(actual_horizon, price_col=PRICE)
    mpc = run_mpc(df, act, SPEC, START, END, horizon_h=24, step_h=24, price_col=PRICE)
    oracle = run_oracle(df, SPEC, START, END, price_col=PRICE)
    assert mpc.overall["rev_total"] == pytest.approx(oracle.total_revenue_eur, rel=1e-6)


def test_mpc_soc_stays_in_band(df):
    act = partial(actual_horizon, price_col=PRICE)
    mpc = run_mpc(df, act, SPEC, START, END, horizon_h=36, step_h=24, price_col=PRICE)
    soc = mpc.realised["soc_mwh"]
    assert (soc >= SPEC.soc_min_mwh - 1e-6).all()
    assert (soc <= SPEC.soc_max_mwh + 1e-6).all()


def test_mpc_bounded_by_global_optimum(df):
    """Under perfect foresight, any receding-horizon decomposition is bounded by
    the single full-window solve (the true global optimum), and should capture
    most of it."""
    import numpy as np

    from bessopt.backtest.mpc import _horizon_index
    from bessopt.optimiser.lp import solve_dispatch

    act = partial(actual_horizon, price_col=PRICE)
    static = run_mpc(df, act, SPEC, START, END, horizon_h=24, step_h=24, price_col=PRICE)
    mpc = run_mpc(df, act, SPEC, START, END, horizon_h=48, step_h=24, price_col=PRICE)

    # Global optimum: one LP over every executed slot.
    n = len(static.realised)
    t0 = pd.Timestamp(START, tz="Europe/Berlin").tz_convert("UTC")
    prices = df[PRICE].reindex(_horizon_index(t0, n)).to_numpy()
    assert not np.isnan(prices).any()
    global_opt = solve_dispatch(prices, SPEC, soc_initial_mwh=SPEC.soc_min_mwh).revenue_eur

    for r in (static, mpc):
        assert r.overall["rev_total"] <= global_opt + 1e-3        # bounded above
        assert r.overall["rev_total"] >= 0.80 * global_opt        # near-optimal (sane)
