"""Intraday continuous (IDC) — simplified two-market value of intraday access.

With access to *both* the day-ahead and the intraday market for the same
delivery slot, a price-taker sells each MWh into the higher of the two prices
and buys at the lower:

    effective sell[t] = max(DA[t], ID[t]),   effective buy[t] = min(DA[t], ID[t]).

`solve_two_market_dispatch` is the perfect-foresight ceiling — "re-optimise the
schedule knowing both prices". The uplift over a day-ahead-only oracle is the
**value of intraday access**. Realising it in practice needs an intraday-index
forecast and (for the full continuous book) event-driven re-optimisation — out
of scope here; this is the index-based upper bound.

ID prices are representative (see `data.sources.intraday`); the structural result
(intraday value scales with the DA-ID spread) is what matters, not the exact €.
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
from ..data.sources.intraday import DEFAULT_RMS_SPREAD_EUR, intraday_index
from ..optimiser.annualise import eur_per_mw_year
from ..optimiser.lp import solve_dispatch
from ..optimiser.spec import BatterySpec, DispatchResult


def solve_two_market_dispatch(
    da_prices: np.ndarray,
    id_prices: np.ndarray,
    spec: BatterySpec,
    *,
    soc_initial_mwh: float | None = None,
    verbose: bool = False,
) -> DispatchResult:
    """Dispatch with access to both DA and ID per slot (sell@max, buy@min)."""
    da = np.asarray(da_prices, dtype=float)
    idp = np.asarray(id_prices, dtype=float)
    sell = np.maximum(da, idp)
    buy = np.minimum(da, idp)

    H = len(da)
    dt = spec.slot_hours
    P = spec.power_mw
    eta_c, eta_d = spec.eta_c, spec.eta_d
    soc_lo, soc_hi = spec.soc_min_mwh, spec.soc_max_mwh
    soc0 = soc_lo if soc_initial_mwh is None else float(soc_initial_mwh)

    t = pd.RangeIndex(H, name="t")

    def da_arr(a):
        return xr.DataArray(np.asarray(a, float), coords={"t": t}, dims=["t"])

    m = linopy.Model()
    charge = m.add_variables(lower=0.0, upper=P, coords=[t], name="charge")
    discharge = m.add_variables(lower=0.0, upper=P, coords=[t], name="discharge")
    soc = m.add_variables(lower=soc_lo, upper=soc_hi, coords=[t], name="soc")

    flow = (eta_c * dt) * charge - (dt / eta_d) * discharge
    interior = xr.DataArray(np.arange(H) >= 1, coords={"t": t}, dims=["t"])
    m.add_constraints(soc - soc.shift(t=1) - flow == 0.0, mask=interior, name="soc_dyn")
    m.add_constraints(soc.isel(t=0) - flow.isel(t=0) == soc0, name="soc_init")

    revenue = (discharge * da_arr(sell) - charge * da_arr(buy)).sum() * dt
    deg = (spec.deg_cost_eur_per_mwh * dt) * (charge + discharge).sum()
    m.add_objective(-revenue + deg, sense="min")
    m.solve(solver_name="highs", **({"output_flag": False} if not verbose else {}))

    ch = np.asarray(charge.solution.values, float)
    dis = np.asarray(discharge.solution.values, float)
    soc_end = np.asarray(soc.solution.values, float)
    gross = float(np.sum(dis * sell - ch * buy) * dt)
    throughput = float(np.sum(ch + dis) * dt)
    return DispatchResult(
        charge_mw=ch, discharge_mw=dis, soc_mwh=np.concatenate([[soc0], soc_end]),
        prices_used=da, revenue_eur=gross - spec.deg_cost_eur_per_mwh * throughput,
        throughput_mwh=throughput, status=str(getattr(m, "termination_condition", "unknown")),
    )


@dataclass
class IntradayValueRun:
    n_days: int
    da_only_eur_per_mw_yr: float
    two_market_eur_per_mw_yr: float
    uplift_eur_per_mw_yr: float
    uplift_pct: float
    da_share_of_two_market: float       # how the two-market revenue splits DA vs ID
    id_share_of_two_market: float
    rms_spread_eur: float


def _daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run_intraday_value(
    df: pd.DataFrame,
    spec: BatterySpec,
    start: date,
    end: date,
    *,
    rms_spread_eur: float = DEFAULT_RMS_SPREAD_EUR,
    price_col: str = "price__germany_luxembourg",
    annualise_days: float = 365.0,
    seed: int = 0,
) -> IntradayValueRun:
    """Perfect-foresight DA-only vs DA+IDC over a window (each on its own SoC path)."""
    soc_da = soc_tm = spec.soc_min_mwh
    da_total = tm_total = tm_da_part = 0.0
    n = 0
    for k, d in enumerate(_daterange(start, end)):
        da = df[price_col].reindex(target_index_for(d)).to_numpy(dtype=float)
        if np.isnan(da).any():
            continue
        idp = intraday_index(da, rms_spread_eur=rms_spread_eur, seed=seed + k)

        da_only = solve_dispatch(da, spec, soc_initial_mwh=soc_da)
        tm = solve_two_market_dispatch(da, idp, spec, soc_initial_mwh=soc_tm)
        soc_da, soc_tm = da_only.soc_mwh[-1], tm.soc_mwh[-1]

        da_total += da_only.revenue_eur
        tm_total += tm.revenue_eur
        # DA-settled value of the two-market schedule (the rest is the ID routing).
        tm_da_part += float(np.sum((tm.discharge_mw - tm.charge_mw) * da) * spec.slot_hours)
        n += 1

    if n == 0:
        raise RuntimeError("No intraday-value days produced — check date range / coverage.")

    pw = spec.power_mw
    da_yr = eur_per_mw_year(da_total, n, pw, annualise_days)
    tm_yr = eur_per_mw_year(tm_total, n, pw, annualise_days)
    return IntradayValueRun(
        n_days=n,
        da_only_eur_per_mw_yr=da_yr,
        two_market_eur_per_mw_yr=tm_yr,
        uplift_eur_per_mw_yr=tm_yr - da_yr,
        uplift_pct=(tm_yr / da_yr - 1.0) * 100 if da_yr else float("nan"),
        da_share_of_two_market=tm_da_part / tm_total if tm_total else float("nan"),
        id_share_of_two_market=1.0 - (tm_da_part / tm_total) if tm_total else float("nan"),
        rms_spread_eur=rms_spread_eur,
    )


__all__ = ["IntradayValueRun", "run_intraday_value", "solve_two_market_dispatch"]
