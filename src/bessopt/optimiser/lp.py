"""The perfect-foresight dispatch LP — the heart of the optimiser.

`solve_dispatch` is a single entry point that serves:
  - the **oracle** (feed actual prices → upper-bound revenue),
  - the **forecast-driven dispatch** (feed forecast prices → schedule to settle),
  - the **MPC** receding-horizon loop (any horizon length, carried SoC),
  - and, via the `reserve_commit` / `scenarios` hooks, Phases 5 & 6 with no
    change to the variable topology.

Formulation (continuous LP; the charge/discharge mutual-exclusivity binary is
deliberately relaxed — with eta_rt < 1 the optimum never charges and discharges
simultaneously at positive prices, and the LP solves 10-100x faster):

    maximise  Σ_t (discharge[t] - charge[t]) · price[t] · Δt
              - deg_cost · Σ_t (charge[t] + discharge[t]) · Δt

    s.t.  soc[t] = soc[t-1] + η_c·charge[t]·Δt - discharge[t]·Δt/η_d   (t ≥ 1)
          soc[0] = soc_initial + η_c·charge[0]·Δt - discharge[0]·Δt/η_d
          soc_min ≤ soc[t] ≤ soc_max,   0 ≤ charge[t], discharge[t] ≤ P_max
          soc[H-1] = soc_terminal              (optional, for MPC coupling)
          Σ_t discharge[t]·Δt ≤ cycle_cap · energy   (optional, Phase 6)

`soc[t]` is the **end-of-slot** state of charge; the returned trajectory
prepends the initial SoC so it has H+1 points (soc_mwh[0] = initial,
soc_mwh[t+1] = end of slot t).
"""

from __future__ import annotations

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .spec import BatterySpec, DispatchResult


def solve_dispatch(
    prices: np.ndarray,
    spec: BatterySpec,
    *,
    soc_initial_mwh: float | None = None,
    soc_terminal_mwh: float | None = None,
    reserve_commit=None,          # ReserveCommitment (Phase 5) — tightens bounds only
    solver: str = "highs",
    index: pd.DatetimeIndex | None = None,
    verbose: bool = False,
) -> DispatchResult:
    prices = np.asarray(prices, dtype=float)
    H = int(prices.shape[0])
    if H == 0:
        raise ValueError("prices must be non-empty")

    dt = spec.slot_hours
    P = spec.power_mw
    eta_c, eta_d = spec.eta_c, spec.eta_d
    soc_lo, soc_hi = spec.soc_min_mwh, spec.soc_max_mwh

    # Per-slot bounds, tightened by any reserve commitment: committed reserve MW
    # consume arbitrage power headroom, and the PICASSO buffer shrinks the usable
    # SoC band (POS aFRR/FCR raise the floor, NEG aFRR/FCR lower the ceiling).
    charge_ub = np.full(H, P)
    discharge_ub = np.full(H, P)
    soc_lb = np.full(H, soc_lo)
    soc_ub = np.full(H, soc_hi)
    if reserve_commit is not None:
        charge_ub = np.minimum(charge_ub, reserve_commit.charge_cap(P))
        discharge_ub = np.minimum(discharge_ub, reserve_commit.discharge_cap(P))
        soc_lb = np.maximum(soc_lb, reserve_commit.soc_lower(soc_lo))
        soc_ub = np.minimum(soc_ub, reserve_commit.soc_upper(soc_hi))
        if (soc_lb > soc_ub + 1e-9).any() or (charge_ub < -1e-9).any() \
                or (discharge_ub < -1e-9).any():
            raise ValueError(
                "Infeasible reserve commitment: requested reserve MW exceed the "
                "battery's power or SoC headroom (the PICASSO buffer leaves no usable band)."
            )

    soc0 = float(soc_lb[0]) if soc_initial_mwh is None else float(soc_initial_mwh)

    t = pd.RangeIndex(H, name="t")
    price_da = xr.DataArray(prices, coords={"t": t}, dims=["t"])

    def _da(arr: np.ndarray) -> xr.DataArray:
        return xr.DataArray(np.asarray(arr, dtype=float), coords={"t": t}, dims=["t"])

    m = linopy.Model()
    charge = m.add_variables(lower=0.0, upper=_da(charge_ub), coords=[t], name="charge")
    discharge = m.add_variables(lower=0.0, upper=_da(discharge_ub), coords=[t], name="discharge")
    soc = m.add_variables(lower=_da(soc_lb), upper=_da(soc_ub), coords=[t], name="soc")

    # Per-slot energy flow into SoC (MWh): η_c·charge·Δt − discharge·Δt/η_d
    flow = (eta_c * dt) * charge - (dt / eta_d) * discharge

    # SoC recursion for t ≥ 1:  soc[t] − soc[t-1] − flow[t] == 0
    recursion = soc - soc.shift(t=1) - flow
    interior = xr.DataArray(np.arange(H) >= 1, coords={"t": t}, dims=["t"])
    m.add_constraints(recursion == 0.0, mask=interior, name="soc_dyn")

    # Initial condition:  soc[0] − flow[0] == soc0
    m.add_constraints(soc.isel(t=0) - flow.isel(t=0) == soc0, name="soc_init")

    # Optional terminal coupling (MPC / day-to-day continuity)
    if soc_terminal_mwh is not None:
        m.add_constraints(soc.isel(t=H - 1) == float(soc_terminal_mwh), name="soc_term")

    # Optional throughput / cycle cap (Phase 6)
    if spec.cycle_cap_per_day is not None:
        cap_mwh = float(spec.cycle_cap_per_day) * spec.energy_mwh * (H * dt / 24.0)
        m.add_constraints((discharge * dt).sum() <= cap_mwh, name="cycle_cap")

    # Objective: maximise revenue − degradation  ⇒  minimise the negative.
    revenue = ((discharge - charge) * price_da * dt).sum()
    degradation = (spec.deg_cost_eur_per_mwh * dt) * (charge + discharge).sum()
    m.add_objective(-revenue + degradation, sense="min")

    solve_kwargs = {}
    if solver == "highs" and not verbose:
        solve_kwargs["output_flag"] = False  # silence the per-solve HiGHS banner
    m.solve(solver_name=solver, **solve_kwargs)

    status = str(getattr(m, "termination_condition", "unknown"))
    if charge.solution is None or np.isnan(np.asarray(charge.solution.values)).any():
        raise RuntimeError(
            f"solve_dispatch found no optimal solution (status={status}); check the "
            "entry/terminal SoC against the reserve-tightened band."
        )

    ch = np.asarray(charge.solution.values, dtype=float)
    dis = np.asarray(discharge.solution.values, dtype=float)
    soc_end = np.asarray(soc.solution.values, dtype=float)
    soc_full = np.concatenate([[soc0], soc_end])  # (H+1,)

    gross = float(np.sum((dis - ch) * prices * dt))
    throughput = float(np.sum(ch + dis) * dt)
    deg = spec.deg_cost_eur_per_mwh * throughput
    revenue_eur = gross - deg

    return DispatchResult(
        charge_mw=ch,
        discharge_mw=dis,
        soc_mwh=soc_full,
        prices_used=prices,
        revenue_eur=revenue_eur,
        throughput_mwh=throughput,
        status=status,
        index=index,
    )


def settle(charge_mw: np.ndarray, discharge_mw: np.ndarray, actual_prices: np.ndarray,
           spec: BatterySpec) -> float:
    """Re-price a committed schedule at realised prices (the honest backtest step).

    Does NOT re-optimise — it settles the already-committed dispatch against the
    actual day-ahead prices, net of the linear degradation cost.
    """
    ch = np.asarray(charge_mw, dtype=float)
    dis = np.asarray(discharge_mw, dtype=float)
    actual = np.asarray(actual_prices, dtype=float)
    dt = spec.slot_hours
    gross = float(np.sum((dis - ch) * actual * dt))
    deg = spec.deg_cost_eur_per_mwh * float(np.sum(ch + dis) * dt)
    return gross - deg


__all__ = ["solve_dispatch", "settle"]
