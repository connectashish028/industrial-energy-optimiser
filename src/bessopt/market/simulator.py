"""Value-stack co-optimisation — day-ahead arbitrage + FCR + aFRR capacity.

A single deterministic LP per delivery day that co-optimises, under perfect
foresight of both day-ahead and reserve prices:

  - arbitrage charge/discharge against the day-ahead price, and
  - how much FCR / POS-aFRR / NEG-aFRR capacity (MW) to commit per 4h block,

subject to the PICASSO buffer (committed reserve consumes power headroom and
shrinks the usable SoC band — see `market.products`). The optimal *mix* differs
by duration: a 1h battery can barely cycle once it holds aFRR, so it tilts to
reserve; a 2h battery does both. That gap is derived from the constraints.

This is the perfect-foresight co-optimum (the upper bound). German desks clear
the markets *sequentially* by gate order; the report notes joint co-optimisation
rarely beats sequential commitment with safety margins, so this deterministic
joint LP is both a clean upper bound and close to realistic practice. Reserve
prices are representative unless the live feed is enabled (see
`data.sources.regelleistung`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from ..data.loader import target_index_for
from ..data.sources.regelleistung import reserve_prices_for_index
from ..optimiser.annualise import eur_per_mw_year
from ..optimiser.spec import BatterySpec
from .products import AFRR_BUFFER_H, FCR_BUFFER_H

SLOTS_PER_BLOCK = 16  # 4h / 15min

# Reserve price columns (EUR/MW/h).
FCR_COL = "reserve_fcr__capacity_price"
AFRR_POS_COL = "reserve_afrr_cap__pos__capacity_price"
AFRR_NEG_COL = "reserve_afrr_cap__neg__capacity_price"

STREAMS = ("day_ahead", "fcr", "afrr_pos", "afrr_neg")


@dataclass
class ValueStackResult:
    revenue_by_stream: dict          # EUR per stream for the day/window
    total_revenue_eur: float
    soc_end_mwh: float
    charge_mw: np.ndarray = field(default=None, repr=False)
    discharge_mw: np.ndarray = field(default=None, repr=False)
    soc_mwh: np.ndarray = field(default=None, repr=False)
    r_fcr_mw: np.ndarray = field(default=None, repr=False)
    r_afrr_pos_mw: np.ndarray = field(default=None, repr=False)
    r_afrr_neg_mw: np.ndarray = field(default=None, repr=False)


def cooptimise_day(
    da_prices: np.ndarray,
    fcr_price: np.ndarray,
    afrr_pos_price: np.ndarray,
    afrr_neg_price: np.ndarray,
    spec: BatterySpec,
    *,
    soc_initial_mwh: float | None = None,
    verbose: bool = False,
) -> ValueStackResult:
    """Co-optimise one delivery day. All price arrays are length H (per slot,
    reserve prices constant within each 4h block); reserve prices are EUR/MW/h."""
    da = np.asarray(da_prices, dtype=float)
    H = len(da)
    dt = spec.slot_hours
    P = spec.power_mw
    eta_c, eta_d = spec.eta_c, spec.eta_d
    soc_lo, soc_hi = spec.soc_min_mwh, spec.soc_max_mwh
    soc0 = soc_lo if soc_initial_mwh is None else float(soc_initial_mwh)

    t = pd.RangeIndex(H, name="t")

    def da_arr(a):
        return xr.DataArray(np.asarray(a, dtype=float), coords={"t": t}, dims=["t"])

    m = linopy.Model()
    charge = m.add_variables(lower=0.0, upper=P, coords=[t], name="charge")
    discharge = m.add_variables(lower=0.0, upper=P, coords=[t], name="discharge")
    soc = m.add_variables(lower=soc_lo, upper=soc_hi, coords=[t], name="soc")
    r_fcr = m.add_variables(lower=0.0, upper=P, coords=[t], name="r_fcr")
    r_pos = m.add_variables(lower=0.0, upper=P, coords=[t], name="r_afrr_pos")
    r_neg = m.add_variables(lower=0.0, upper=P, coords=[t], name="r_afrr_neg")

    # SoC dynamics.
    flow = (eta_c * dt) * charge - (dt / eta_d) * discharge
    interior = xr.DataArray(np.arange(H) >= 1, coords={"t": t}, dims=["t"])
    m.add_constraints(soc - soc.shift(t=1) - flow == 0.0, mask=interior, name="soc_dyn")
    m.add_constraints(soc.isel(t=0) - flow.isel(t=0) == soc0, name="soc_init")

    # Reserve is sold per 4h block ⇒ hold each reserve var constant within a block.
    non_boundary = xr.DataArray((np.arange(H) % SLOTS_PER_BLOCK) != 0, coords={"t": t}, dims=["t"])
    for r, nm in ((r_fcr, "fcr"), (r_pos, "pos"), (r_neg, "neg")):
        m.add_constraints(r - r.shift(t=1) == 0.0, mask=non_boundary, name=f"block_{nm}")

    # Power headroom: arbitrage shares the inverter with committed reserve.
    m.add_constraints(charge + r_neg + r_fcr <= P, name="charge_headroom")
    m.add_constraints(discharge + r_pos + r_fcr <= P, name="discharge_headroom")

    # PICASSO SoC buffer: POS aFRR + FCR raise the floor; NEG aFRR + FCR lower the ceiling.
    m.add_constraints(soc - AFRR_BUFFER_H * r_pos - FCR_BUFFER_H * r_fcr >= soc_lo, name="soc_floor")
    m.add_constraints(soc + AFRR_BUFFER_H * r_neg + FCR_BUFFER_H * r_fcr <= soc_hi, name="soc_ceil")

    # Objective: DA arbitrage + reserve capacity revenue − degradation.  (Maximise ⇒ minimise −.)
    rev_da = ((discharge - charge) * da_arr(da) * dt).sum()
    rev_res = (
        (r_fcr * da_arr(fcr_price) + r_pos * da_arr(afrr_pos_price) + r_neg * da_arr(afrr_neg_price))
        * dt
    ).sum()
    deg = (spec.deg_cost_eur_per_mwh * dt) * (charge + discharge).sum()
    m.add_objective(-(rev_da + rev_res) + deg, sense="min")

    solve_kwargs = {"output_flag": False} if not verbose else {}
    m.solve(solver_name="highs", **solve_kwargs)
    if charge.solution is None or np.isnan(np.asarray(charge.solution.values)).any():
        raise RuntimeError("value-stack LP found no optimal solution.")

    ch = np.asarray(charge.solution.values, dtype=float)
    dis = np.asarray(discharge.solution.values, dtype=float)
    soc_end = np.asarray(soc.solution.values, dtype=float)
    rf = np.asarray(r_fcr.solution.values, dtype=float)
    rp = np.asarray(r_pos.solution.values, dtype=float)
    rn = np.asarray(r_neg.solution.values, dtype=float)

    rev = {
        "day_ahead": float(np.sum((dis - ch) * da * dt)
                           - spec.deg_cost_eur_per_mwh * np.sum(ch + dis) * dt),
        "fcr": float(np.sum(rf * fcr_price * dt)),
        "afrr_pos": float(np.sum(rp * afrr_pos_price * dt)),
        "afrr_neg": float(np.sum(rn * afrr_neg_price * dt)),
    }
    return ValueStackResult(
        revenue_by_stream=rev,
        total_revenue_eur=float(sum(rev.values())),
        soc_end_mwh=float(soc_end[-1]),
        charge_mw=ch, discharge_mw=dis, soc_mwh=np.concatenate([[soc0], soc_end]),
        r_fcr_mw=rf, r_afrr_pos_mw=rp, r_afrr_neg_mw=rn,
    )


@dataclass
class ValueStackRun:
    per_day: pd.DataFrame                  # date + revenue per stream
    revenue_by_stream: dict                # EUR totals
    eur_per_mw_yr_by_stream: dict
    total_eur_per_mw_yr: float
    n_days: int


def _daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += pd.Timedelta(days=1)


def run_value_stack(
    df: pd.DataFrame,
    spec: BatterySpec,
    start: date,
    end: date,
    *,
    price_col: str = "price__germany_luxembourg",
    carry_soc: bool = True,
    annualise_days: float = 365.0,
    progress: bool = True,
) -> ValueStackRun:
    """Co-optimise the value stack day by day over a window (carrying SoC), and
    accumulate revenue by stream. Reserve prices come from `regelleistung`."""
    from tqdm import tqdm

    soc_state = spec.soc_min_mwh
    rows: list[dict] = []
    days = list(_daterange(start, end))
    it = tqdm(days, desc=f"value-stack:{spec.duration_h:.0f}h", unit="day") if progress else days

    for d in it:
        idx = target_index_for(d)
        da = df[price_col].reindex(idx).to_numpy(dtype=float)
        if np.isnan(da).any():
            continue
        rp = reserve_prices_for_index(idx)
        res = cooptimise_day(
            da, rp[FCR_COL].to_numpy(), rp[AFRR_POS_COL].to_numpy(), rp[AFRR_NEG_COL].to_numpy(),
            spec, soc_initial_mwh=(soc_state if carry_soc else None),
        )
        if carry_soc:
            soc_state = res.soc_end_mwh
        rows.append({"date": d, **res.revenue_by_stream})

    if not rows:
        raise RuntimeError("No value-stack days produced — check date range / coverage.")

    per_day = pd.DataFrame(rows)
    n_days = len(per_day)
    totals = {s: float(per_day[s].sum()) for s in STREAMS}
    per_mw_yr = {s: eur_per_mw_year(totals[s], n_days, spec.power_mw, annualise_days) for s in STREAMS}
    return ValueStackRun(
        per_day=per_day,
        revenue_by_stream=totals,
        eur_per_mw_yr_by_stream=per_mw_yr,
        total_eur_per_mw_yr=float(sum(per_mw_yr.values())),
        n_days=n_days,
    )


__all__ = ["STREAMS", "ValueStackResult", "ValueStackRun", "cooptimise_day", "run_value_stack"]
