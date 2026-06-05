"""Tests for CVaR risk-aware dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bessopt.optimiser.lp import solve_dispatch
from bessopt.optimiser.spec import BatterySpec
from bessopt.risk.cvar import (
    ScenarioSet,
    scenarios_from_quantiles,
    scenarios_sampled_from_quantiles,
    solve_cvar_dispatch,
)

SPEC = BatterySpec(power_mw=10.0, energy_mwh=20.0)


def _two_peak_quantiles() -> pd.DataFrame:
    h = np.arange(96) / 4.0
    base = (60 + 35 * np.exp(-((h - 8) ** 2) / 4) + 45 * np.exp(-((h - 19) ** 2) / 5)
            - 25 * np.exp(-((h - 13) ** 2) / 6))
    width = 6 + 30 * np.exp(-((h - 8) ** 2) / 4) + 35 * np.exp(-((h - 19) ** 2) / 5)
    return pd.DataFrame({"p10": base - width, "p50": base, "p90": base + width})


def test_single_scenario_matches_plain_dispatch():
    prices = np.tile([20.0, 20, 20, 20, 120, 120, 120, 120], 12)
    scen = ScenarioSet(paths=prices[None, :], probs=np.array([1.0]))
    cv = solve_cvar_dispatch(scen, SPEC, beta=0.0, soc_initial_mwh=SPEC.soc_min_mwh)
    det = solve_dispatch(prices, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    assert cv.expected_revenue_eur == pytest.approx(det.revenue_eur, rel=1e-4)


def test_beta_zero_is_risk_neutral_mean():
    scen = scenarios_from_quantiles(_two_peak_quantiles())
    cv = solve_cvar_dispatch(scen, SPEC, beta=0.0, soc_initial_mwh=SPEC.soc_min_mwh)
    det = solve_dispatch(scen.mean_path, SPEC, soc_initial_mwh=SPEC.soc_min_mwh)
    assert cv.expected_revenue_eur == pytest.approx(det.revenue_eur, rel=1e-3)


def test_sampled_scenarios_disagree_on_timing():
    scen = scenarios_sampled_from_quantiles(_two_peak_quantiles(), n_scenarios=20, seed=1)
    assert scen.paths.shape == (20, 96)
    # Not comonotonic: the argmax (most expensive slot) varies across scenarios.
    peak_slots = scen.paths.argmax(axis=1)
    assert len(np.unique(peak_slots)) > 1


def test_efficient_frontier_is_monotone():
    """Raising β lowers expected revenue and lowers CVaR(loss) (improves downside)."""
    scen = scenarios_sampled_from_quantiles(_two_peak_quantiles(), n_scenarios=40, seed=1)
    betas = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = [solve_cvar_dispatch(scen, SPEC, alpha=0.2, beta=b,
                                   soc_initial_mwh=SPEC.soc_min_mwh) for b in betas]
    e = [r.expected_revenue_eur for r in results]
    cvar = [r.cvar_loss_eur for r in results]
    worst = [r.scenario_revenues.min() for r in results]
    tol = 1e-3
    assert all(e[i + 1] <= e[i] + tol for i in range(len(e) - 1))         # E[rev] ↓
    assert all(cvar[i + 1] <= cvar[i] + tol for i in range(len(cvar) - 1))  # CVaR(loss) ↓ (safer)
    assert worst[-1] >= worst[0] - tol                                    # worst case improves
    # Risk aversion actually changed the dispatch (frontier is not a point).
    assert e[0] - e[-1] > tol


def test_soc_in_band_for_all_beta():
    scen = scenarios_sampled_from_quantiles(_two_peak_quantiles(), n_scenarios=20, seed=2)
    for b in (0.0, 1.0, 5.0):
        r = solve_cvar_dispatch(scen, SPEC, beta=b, soc_initial_mwh=SPEC.soc_min_mwh)
        assert (r.soc_mwh >= SPEC.soc_min_mwh - 1e-6).all()
        assert (r.soc_mwh <= SPEC.soc_max_mwh + 1e-6).all()
