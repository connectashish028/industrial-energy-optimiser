"""Industrial flexibility MILP — minimise EPEX procurement cost.

Per delivery day, a mixed-integer programme (HiGHS) decides:
  - the flexible process on/off schedule (binary, must meet its daily run-hours),
  - on-site PV self-consumption,
  - battery charge/discharge (peak-shaving / arbitrage behind the meter),
to minimise Σ_t grid[t]·spot[t]·Δt, where

    grid[t] = baseline + proc_power·on[t] + charge[t] − discharge[t] − pv_used[t]

subject to grid[t] ∈ [0, grid_limit] (import only — PV surplus is curtailed),
the battery SoC dynamics, and the daily run-hours requirement. The saving is
this optimum vs running the process on a fixed day-shift with the battery idle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from ..data.loader import target_index_for
from .consumer import ConsumerSpec
from .pv import pv_available


@dataclass
class FlexResult:
    grid_mw: np.ndarray
    proc_on: np.ndarray
    pv_used_mw: np.ndarray
    pv_avail_mw: np.ndarray
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    baseline_mw: np.ndarray
    spot: np.ndarray
    cost_eur: float
    soc_end_mwh: float
    status: str


def optimise_day(
    spot: np.ndarray,
    pv_avail: np.ndarray,
    spec: ConsumerSpec,
    *,
    soc_init: float | None = None,
    verbose: bool = False,
) -> FlexResult:
    spot = np.asarray(spot, dtype=float)
    pv_avail = np.asarray(pv_avail, dtype=float)
    H = len(spot)
    dt = spec.slot_hours
    base = np.full(H, spec.baseline_load_mw)
    t = pd.RangeIndex(H, name="t")

    def A(a):
        return xr.DataArray(np.asarray(a, dtype=float), coords={"t": t}, dims=["t"])

    m = linopy.Model()
    on = m.add_variables(coords=[t], name="proc_on", binary=True)
    pv_used = m.add_variables(lower=0.0, upper=A(pv_avail), coords=[t], name="pv_used")
    grid = m.add_variables(lower=0.0, upper=spec.grid_limit_mw, coords=[t], name="grid")

    batt = spec.battery
    if batt is not None:
        soc0 = batt.soc_min_mwh if soc_init is None else float(soc_init)
        charge = m.add_variables(lower=0.0, upper=batt.power_mw, coords=[t], name="charge")
        discharge = m.add_variables(lower=0.0, upper=batt.power_mw, coords=[t], name="discharge")
        soc = m.add_variables(lower=batt.soc_min_mwh, upper=batt.soc_max_mwh, coords=[t], name="soc")
        flow = (batt.eta_c * dt) * charge - (dt / batt.eta_d) * discharge
        interior = A(np.arange(H) >= 1)
        m.add_constraints(soc - soc.shift(t=1) - flow == 0.0, mask=interior, name="soc_dyn")
        m.add_constraints(soc.isel(t=0) - flow.isel(t=0) == soc0, name="soc_init")
        # grid balance with battery
        m.add_constraints(
            grid - spec.proc_power_mw * on - charge + discharge + pv_used == A(base),
            name="balance",
        )
    else:
        soc0 = 0.0
        m.add_constraints(grid - spec.proc_power_mw * on + pv_used == A(base), name="balance")

    # Daily run-hours requirement (cost-min ⇒ it runs exactly this many of the cheapest slots).
    m.add_constraints(on.sum() >= spec.proc_slots, name="runtime")

    deg = 0.0
    if batt is not None and batt.deg_cost_eur_per_mwh > 0:
        deg = (batt.deg_cost_eur_per_mwh * dt) * (charge + discharge).sum()
    m.add_objective((grid * A(spot) * dt).sum() + deg, sense="min")
    m.solve(solver_name="highs", **({"output_flag": False} if not verbose else {}))

    def sol(v):
        return np.asarray(v.solution.values, dtype=float)

    grid_mw = sol(grid)
    on_mw = np.round(sol(on))
    pv_used_mw = sol(pv_used)
    if batt is not None:
        ch, dis, soc_end = sol(charge), sol(discharge), sol(soc)
        soc_full = np.concatenate([[soc0], soc_end])
        soc_last = soc_end[-1]
    else:
        ch = dis = np.zeros(H)
        soc_full = np.zeros(H + 1)
        soc_last = 0.0

    return FlexResult(
        grid_mw=grid_mw, proc_on=on_mw, pv_used_mw=pv_used_mw, pv_avail_mw=pv_avail,
        charge_mw=ch, discharge_mw=dis, soc_mwh=soc_full, baseline_mw=base, spot=spot,
        cost_eur=float(np.sum(grid_mw * spot * dt)), soc_end_mwh=float(soc_last),
        status=str(getattr(m, "termination_condition", "unknown")),
    )


def naive_cost(spot: np.ndarray, pv_avail: np.ndarray, spec: ConsumerSpec) -> tuple[float, np.ndarray]:
    """Cost of running the flexible process on a fixed day-shift, battery idle,
    PV passively self-consumed. Returns (cost, naive grid profile)."""
    spot = np.asarray(spot, dtype=float)
    pv_avail = np.asarray(pv_avail, dtype=float)
    H = len(spot)
    dt = spec.slot_hours
    proc = np.zeros(H)
    s = int(round(spec.naive_proc_start_hour / dt))
    proc[s:s + spec.proc_slots] = spec.proc_power_mw
    load = spec.baseline_load_mw + proc
    grid = np.maximum(load - np.minimum(pv_avail, load), 0.0)
    return float(np.sum(grid * spot * dt)), grid


@dataclass
class FlexRun:
    n_days: int
    optimised_cost_eur: float
    naive_cost_eur: float
    savings_eur: float
    savings_pct: float
    optimised_eur_per_year: float
    naive_eur_per_year: float
    savings_eur_per_year: float
    best_day: date
    best_day_result: FlexResult
    best_day_naive_grid: np.ndarray


def _daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run_flex(
    df: pd.DataFrame,
    spec: ConsumerSpec,
    start: date,
    end: date,
    *,
    price_col: str = "price__germany_luxembourg",
    days_per_year: float = 365.0,
) -> FlexRun:
    soc = spec.battery.soc_min_mwh if spec.battery is not None else 0.0
    opt_total = naive_total = 0.0
    n = 0
    best_savings = -np.inf
    best_day = None
    best_res = None
    best_naive_grid = None

    for d in _daterange(start, end):
        idx = target_index_for(d)
        spot = df[price_col].reindex(idx).to_numpy(dtype=float)
        if np.isnan(spot).any():
            continue
        pv = pv_available(df, idx, spec.pv_capacity_mwp)
        res = optimise_day(spot, pv, spec, soc_init=soc)
        nc, ngrid = naive_cost(spot, pv, spec)
        soc = res.soc_end_mwh
        opt_total += res.cost_eur
        naive_total += nc
        n += 1
        if nc - res.cost_eur > best_savings:
            best_savings, best_day, best_res, best_naive_grid = nc - res.cost_eur, d, res, ngrid

    if n == 0:
        raise RuntimeError("No flex days produced — check date range / coverage.")

    savings = naive_total - opt_total
    scale = days_per_year / n
    return FlexRun(
        n_days=n,
        optimised_cost_eur=opt_total,
        naive_cost_eur=naive_total,
        savings_eur=savings,
        savings_pct=(savings / naive_total * 100) if naive_total else float("nan"),
        optimised_eur_per_year=opt_total * scale,
        naive_eur_per_year=naive_total * scale,
        savings_eur_per_year=savings * scale,
        best_day=best_day,
        best_day_result=best_res,
        best_day_naive_grid=best_naive_grid,
    )


__all__ = ["FlexResult", "FlexRun", "naive_cost", "optimise_day", "run_flex"]
