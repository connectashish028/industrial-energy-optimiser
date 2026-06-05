"""Tiny FastAPI service exposing the industrial-flex MILP for the React demo.

Lets the React app set parameters and solve ONE day live (sub-second), proving
the React -> FastAPI -> Python/HiGHS stack. The parquet + price model load once
at startup; each /solve builds a ConsumerSpec, runs the MILP, and returns the
dispatch + cost + a solve time. Infeasible parameter sets (e.g. a grid limit
below peak load) come back as feasible=false with a plain-English reason — which
is itself a nice demonstration that the constraints bite.

    uv run python serve/flex_api.py          # -> http://127.0.0.1:8000
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bessopt.config import load_data_config
from bessopt.data.loader import issue_time_for, load_de_lu_15min, target_index_for
from bessopt.flex import ConsumerSpec, naive_cost, optimise_day
from bessopt.forecast.predict import xgboost_price_predict_full
from bessopt.flex.pv import pv_available
from bessopt.optimiser.spec import BatterySpec

_cfg = load_data_config()
_DF = load_de_lu_15min(_cfg["parquet_path"])
_PRICE_COL = _cfg["price_col"]

app = FastAPI(title="bessopt flex API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class SolveReq(BaseModel):
    date: str = "2026-01-08"
    baseline_mw: float = 8.0
    proc_mw: float = 6.0
    proc_hours: float = 8.0
    batt_mwh: float = 20.0
    grid_limit_mw: float = 25.0
    pv_mwp: float = 10.0
    batt_mw: float = 10.0
    mode: str = "perfect"  # "perfect" = actual prices; "forecast" = D-1 P50 settled at actual


@app.get("/health")
def health() -> dict:
    return {"ok": True, "window": ["2026-01-01", "2026-02-28"]}


@app.post("/solve")
def solve(req: SolveReq) -> dict:
    try:
        d = date.fromisoformat(req.date)
    except ValueError:
        return {"feasible": False, "message": f"Bad date: {req.date}"}

    idx = target_index_for(d)
    spot = _DF[_PRICE_COL].reindex(idx).to_numpy(dtype=float)
    if np.isnan(spot).any():
        return {"feasible": False,
                "message": f"No complete price data for {req.date} (try a day in Jan–Feb 2026)."}

    proc_hours = float(np.clip(req.proc_hours, 0.0, 24.0))
    pv = pv_available(_DF, idx, req.pv_mwp)
    batt = BatterySpec(power_mw=req.batt_mw, energy_mwh=req.batt_mwh) if req.batt_mw > 0 else None
    spec = ConsumerSpec(
        baseline_load_mw=req.baseline_mw, proc_power_mw=req.proc_mw,
        proc_hours_per_day=proc_hours, pv_capacity_mwp=req.pv_mwp,
        grid_limit_mw=req.grid_limit_mw, battery=batt,
    )

    infeasible_msg = ("Infeasible — the limits can't all be met. "
                      "Raise the grid limit, cut run-hours, or add a battery.")
    t0 = time.perf_counter()
    try:
        res = optimise_day(spot, pv, spec)
    except Exception as exc:  # noqa: BLE001 — surface to the UI
        m = str(exc).lower()
        if "not optim" in m or "infeasible" in m:
            return {"feasible": False, "message": infeasible_msg}
        return {"feasible": False, "message": f"Solver error: {exc}"}
    solve_ms = (time.perf_counter() - t0) * 1000.0

    if not np.isfinite(res.cost_eur):
        return {"feasible": False, "message": infeasible_msg}

    nc, ngrid = naive_cost(spot, pv, spec)
    dt = spec.slot_hours
    perfect_cost = res.cost_eur
    perfect_savings = nc - perfect_cost

    # Forecast-driven: optimise on the D-1 P50 price forecast, then settle the
    # resulting schedule at the realised (actual) price. The gap vs perfect
    # foresight is the cost of forecast error — value capture (VCR) for flexibility.
    forecast_savings: float | None = None
    vcr: float | None = None
    res_fc, fc_cost, fc_p50 = res, perfect_cost, None
    try:
        fc_p50 = (xgboost_price_predict_full(_DF, issue_time_for(d))
                  .reindex(idx)["p50"].to_numpy(dtype=float))
        res_fc = optimise_day(fc_p50, pv, spec)
        fc_cost = float(np.sum(res_fc.grid_mw * spot * dt))  # re-price the schedule at actual
        forecast_savings = nc - fc_cost
        if perfect_savings > 1e-6:
            vcr = round(forecast_savings / perfect_savings * 100, 1)
    except Exception:  # noqa: BLE001 — forecast is best-effort; perfect is still returned
        pass

    if req.mode == "forecast" and forecast_savings is not None:
        g_opt, g_proc, sel_opt, sel_sav = res_fc.grid_mw, res_fc.proc_on, fc_cost, forecast_savings
    else:
        g_opt, g_proc, sel_opt, sel_sav = res.grid_mw, res.proc_on, perfect_cost, perfect_savings

    return {
        "feasible": True,
        "date": req.date,
        "mode": req.mode,
        "peak_load_mw": spec.peak_load_mw,
        "price": [round(x, 2) for x in spot.tolist()],
        "price_forecast": [round(x, 2) for x in fc_p50.tolist()] if fc_p50 is not None else None,
        "grid_opt": [round(x, 3) for x in g_opt.tolist()],
        "grid_naive": [round(x, 3) for x in ngrid.tolist()],
        "proc_on": [int(round(x)) for x in g_proc.tolist()],
        "opt_eur": round(sel_opt, 2),
        "naive_eur": round(nc, 2),
        "savings_eur": round(sel_sav, 2),
        "savings_pct": round(sel_sav / nc * 100, 1) if nc else 0.0,
        "perfect_savings_eur": round(perfect_savings, 2),
        "forecast_savings_eur": round(forecast_savings, 2) if forecast_savings is not None else None,
        "vcr": vcr,
        "solve_ms": round(solve_ms, 1),
    }


_WIN_START, _WIN_END = date(2026, 1, 1), date(2026, 2, 28)


def _daterange(s: date, e: date):
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


class WindowReq(BaseModel):
    baseline_mw: float = 8.0
    proc_mw: float = 6.0
    proc_hours: float = 8.0
    batt_mwh: float = 20.0
    grid_limit_mw: float = 25.0
    pv_mwp: float = 10.0
    batt_mw: float = 10.0


@app.post("/window")
def window(req: WindowReq) -> dict:
    """Solve every day in the Jan–Feb window for these params (perfect foresight),
    returning the per-day savings series + annualised aggregates + the battery-free
    flexibility-only decomposition."""
    proc_hours = float(np.clip(req.proc_hours, 0.0, 24.0))
    batt = BatterySpec(power_mw=req.batt_mw, energy_mwh=req.batt_mwh) if req.batt_mw > 0 else None
    base_kw = dict(baseline_load_mw=req.baseline_mw, proc_power_mw=req.proc_mw,
                   proc_hours_per_day=proc_hours, pv_capacity_mwp=req.pv_mwp,
                   grid_limit_mw=req.grid_limit_mw)
    spec = ConsumerSpec(battery=batt, **base_kw)
    spec_nb = ConsumerSpec(battery=None, **base_kw)
    dt = spec.slot_hours

    t0 = time.perf_counter()
    daily: list[dict] = []
    opt_tot = naive_tot = naive_mwh_tot = opt_nb_tot = naive_nb_tot = 0.0
    soc = spec.battery.soc_min_mwh if spec.battery is not None else 0.0
    for d in _daterange(_WIN_START, _WIN_END):
        idx = target_index_for(d)
        spot = _DF[_PRICE_COL].reindex(idx).to_numpy(dtype=float)
        if np.isnan(spot).any():
            continue
        pv = pv_available(_DF, idx, req.pv_mwp)
        try:
            res = optimise_day(spot, pv, spec, soc_init=soc)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(res.cost_eur):
            continue
        nc, ngrid = naive_cost(spot, pv, spec)
        soc = res.soc_end_mwh
        naive_mwh = float(np.sum(ngrid * dt))
        res_nb = optimise_day(spot, pv, spec_nb)
        nc_nb, _ = naive_cost(spot, pv, spec_nb)
        daily.append({"date": d.isoformat(), "savings": round(nc - res.cost_eur, 2),
                      "naive": round(nc, 2), "naive_mwh": round(naive_mwh, 3),
                      "savings_nb": round(nc_nb - res_nb.cost_eur, 2), "naive_nb": round(nc_nb, 2)})
        opt_tot += res.cost_eur
        naive_tot += nc
        naive_mwh_tot += naive_mwh
        opt_nb_tot += res_nb.cost_eur
        naive_nb_tot += nc_nb
    solve_ms = (time.perf_counter() - t0) * 1000.0

    n = len(daily)
    if n == 0:
        return {"feasible": False, "message": "Infeasible for these parameters across the window."}
    scale = 365.0 / n
    savings_tot = naive_tot - opt_tot
    flex_only = naive_nb_tot - opt_nb_tot
    return {
        "feasible": True, "days": n, "daily": daily,
        "savings_pct": round(savings_tot / naive_tot * 100, 1),
        "savings_eur_yr": round(savings_tot * scale),
        "naive_eur_yr": round(naive_tot * scale),
        "opt_eur_yr": round(opt_tot * scale),
        "savings_eur_per_mwh": round(savings_tot / naive_mwh_tot, 2) if naive_mwh_tot else 0.0,
        "savings_keur_per_mw_yr": round(savings_tot * scale / 1000 / spec.peak_load_mw, 1),
        "flex_only_pct": round(flex_only / naive_nb_tot * 100, 1) if naive_nb_tot else 0.0,
        "peak_load_mw": spec.peak_load_mw,
        "solve_ms": round(solve_ms, 0),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
