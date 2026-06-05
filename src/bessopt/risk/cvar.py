"""CVaR risk-aware dispatch — max E[revenue] − β·CVaR_α(loss).

A desk optimises risk-adjusted return, not raw expected revenue. We add a
Conditional Value at Risk term over price *scenarios* drawn from the
probabilistic forecast, using the standard Rockafellar-Uryasev linearisation
(auxiliary VaR variable η and per-scenario excess-loss u_s ≥ 0), which keeps the
problem an LP:

    maximise   Σ_s p_s · rev_s  −  β·( η + (1/α)·Σ_s p_s · u_s )
    s.t.       u_s ≥ −rev_s − η ,   u_s ≥ 0
               rev_s = Σ_t (discharge[t] − charge[t]) · price_s[t] · Δt
               + the usual SoC dynamics / band / power limits on the single
                 here-and-now schedule (charge/discharge/soc).

β = 0 recovers the risk-neutral dispatch against the mean scenario. Sweeping β
traces the efficient frontier (expected revenue vs downside CVaR).

Scenarios are comonotonic quantile paths interpolated from the forecast's
P10/P50/P90 — "the day turns out generally low / median / high" — which is the
spread risk a battery actually faces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from ..optimiser.spec import BatterySpec

QUANTILE_ANCHORS = (0.10, 0.50, 0.90)   # the columns p10/p50/p90 represent


@dataclass(frozen=True)
class ScenarioSet:
    paths: np.ndarray   # (S, H) price scenarios, EUR/MWh
    probs: np.ndarray   # (S,) probabilities, sum to 1

    @property
    def mean_path(self) -> np.ndarray:
        return self.probs @ self.paths


def scenarios_from_quantiles(
    quantile_df: pd.DataFrame,
    *,
    levels: Sequence[float] = (0.10, 0.30, 0.50, 0.70, 0.90),
    probs: Sequence[float] | None = None,
) -> ScenarioSet:
    """Build comonotonic quantile-path scenarios from a (H,3) p10/p50/p90 frame.

    Each scenario fixes a single quantile level across all slots, interpolated
    per slot from the P10/P50/P90 anchors — so a path is the whole day realised
    at, say, its 30th percentile. Equal probabilities by default.
    """
    anchors = np.array(QUANTILE_ANCHORS)
    q = quantile_df[["p10", "p50", "p90"]].to_numpy()        # (H, 3)
    paths = np.stack(
        [np.array([np.interp(lv, anchors, q[t]) for t in range(q.shape[0])]) for lv in levels]
    )                                                          # (S, H)
    p = np.full(len(levels), 1.0 / len(levels)) if probs is None else np.asarray(probs, float)
    return ScenarioSet(paths=paths, probs=p / p.sum())


def scenarios_sampled_from_quantiles(
    quantile_df: pd.DataFrame,
    *,
    n_scenarios: int = 40,
    block_slots: int = 16,
    seed: int = 0,
) -> ScenarioSet:
    """Sample scenarios that disagree on *timing*, not just level.

    Each scenario draws an independent quantile level per 4h block (interpolated
    per slot from P10/P50/P90), so different scenarios make different blocks the
    expensive one. That timing disagreement is what creates a genuine risk-return
    tradeoff for a price-taker (comonotonic level-only scenarios do not — the best
    slots are best in every scenario). Modelling choice: block magnitudes are
    treated as partially independent draws from the forecast marginals.
    """
    anchors = np.array(QUANTILE_ANCHORS)
    q = quantile_df[["p10", "p50", "p90"]].to_numpy()       # (H, 3)
    H = q.shape[0]
    n_blocks = int(np.ceil(H / block_slots))
    rng = np.random.default_rng(seed)

    paths = np.empty((n_scenarios, H))
    for k in range(n_scenarios):
        block_levels = rng.uniform(0.05, 0.95, size=n_blocks)
        levels = np.repeat(block_levels, block_slots)[:H]
        paths[k] = [np.interp(levels[t], anchors, q[t]) for t in range(H)]
    return ScenarioSet(paths=paths, probs=np.full(n_scenarios, 1.0 / n_scenarios))


@dataclass
class CvarResult:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    expected_revenue_eur: float
    var_loss_eur: float          # η — Value at Risk of the loss
    cvar_loss_eur: float         # CVaR_α(loss): mean loss in the worst α tail (lower = safer)
    scenario_revenues: np.ndarray
    beta: float
    alpha: float
    status: str


def solve_cvar_dispatch(
    scenarios: ScenarioSet,
    spec: BatterySpec,
    *,
    alpha: float = 0.10,
    beta: float = 0.0,
    soc_initial_mwh: float | None = None,
    verbose: bool = False,
) -> CvarResult:
    paths = np.asarray(scenarios.paths, dtype=float)
    probs = np.asarray(scenarios.probs, dtype=float)
    S, H = paths.shape
    dt = spec.slot_hours
    P = spec.power_mw
    eta_c, eta_d = spec.eta_c, spec.eta_d
    soc_lo, soc_hi = spec.soc_min_mwh, spec.soc_max_mwh
    soc0 = soc_lo if soc_initial_mwh is None else float(soc_initial_mwh)

    t = pd.RangeIndex(H, name="t")
    s = pd.RangeIndex(S, name="s")
    paths_da = xr.DataArray(paths, coords={"s": s, "t": t}, dims=["s", "t"])
    probs_da = xr.DataArray(probs, coords={"s": s}, dims=["s"])

    m = linopy.Model()
    charge = m.add_variables(lower=0.0, upper=P, coords=[t], name="charge")
    discharge = m.add_variables(lower=0.0, upper=P, coords=[t], name="discharge")
    soc = m.add_variables(lower=soc_lo, upper=soc_hi, coords=[t], name="soc")
    eta = m.add_variables(name="eta")                       # free scalar (VaR)
    u = m.add_variables(lower=0.0, coords=[s], name="u")    # per-scenario excess loss

    # SoC dynamics on the single schedule.
    flow = (eta_c * dt) * charge - (dt / eta_d) * discharge
    interior = xr.DataArray(np.arange(H) >= 1, coords={"t": t}, dims=["t"])
    m.add_constraints(soc - soc.shift(t=1) - flow == 0.0, mask=interior, name="soc_dyn")
    m.add_constraints(soc.isel(t=0) - flow.isel(t=0) == soc0, name="soc_init")

    # Revenue per scenario and the CVaR excess-loss constraint:  u_s ≥ −rev_s − η.
    rev_s = ((discharge - charge) * paths_da * dt).sum("t")   # expression over s
    m.add_constraints(u + rev_s + eta >= 0.0, name="cvar_excess")

    expected_rev = (probs_da * rev_s).sum()
    cvar = eta + (1.0 / alpha) * (probs_da * u).sum()
    deg = (spec.deg_cost_eur_per_mwh * dt) * (charge + discharge).sum()
    m.add_objective(-(expected_rev) + beta * cvar + deg, sense="min")

    m.solve(solver_name="highs", **({"output_flag": False} if not verbose else {}))
    status = str(getattr(m, "termination_condition", "unknown"))
    if charge.solution is None or np.isnan(np.asarray(charge.solution.values)).any():
        raise RuntimeError(f"CVaR dispatch found no optimal solution (status={status}).")

    ch = np.asarray(charge.solution.values, float)
    dis = np.asarray(discharge.solution.values, float)
    soc_end = np.asarray(soc.solution.values, float)
    scen_rev = ((dis - ch)[None, :] * paths * dt).sum(axis=1)       # (S,)
    exp_rev = float(probs @ scen_rev)
    eta_val = float(np.asarray(eta.solution.values))
    loss = -scen_rev
    cvar_val = float(eta_val + (1.0 / alpha) * (probs @ np.maximum(loss - eta_val, 0.0)))

    return CvarResult(
        charge_mw=ch, discharge_mw=dis, soc_mwh=np.concatenate([[soc0], soc_end]),
        expected_revenue_eur=exp_rev, var_loss_eur=eta_val, cvar_loss_eur=cvar_val,
        scenario_revenues=scen_rev, beta=float(beta), alpha=float(alpha), status=status,
    )


__all__ = [
    "CvarResult",
    "ScenarioSet",
    "scenarios_from_quantiles",
    "scenarios_sampled_from_quantiles",
    "solve_cvar_dispatch",
]
