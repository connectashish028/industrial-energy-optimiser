"""Rolling-horizon / Model Predictive Control (M4).

Wraps the SAME `solve_dispatch` in a receding-horizon loop: at each decision
point optimise the next N hours, execute only the first `step_h`, carry SoC,
re-forecast, re-solve. The uplift over the static once-daily solve comes from
*seeing past midnight* — a 36-48h horizon lets the optimiser keep charge
overnight when tomorrow morning is expensive, which a 24h solve cannot.

Two horizon price assemblers, both leak-free at the decision time t0 (Berlin
midnight of the decision day D):
  - `forecast_horizon`: XGBoost P50 for day D (issued D-1 12:00, known before t0)
    + seasonal-naive (price 7 days earlier) for the look-ahead tail beyond D.
    The naive tail is a deliberate, documented proxy — the day-ahead model is
    single-day, so the beyond-D horizon uses the only leak-free signal available.
  - `actual_horizon`: perfect foresight, to isolate the pure horizon effect from
    forecast error (the structural ceiling on MPC uplift).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..data.loader import issue_time_for
from ..optimiser.annualise import eur_per_mw_year
from ..optimiser.lp import solve_dispatch
from ..optimiser.spec import BatterySpec
from .baselines import seasonal_naive_price_predict

BERLIN = "Europe/Berlin"

# A horizon assembler: (df, t0, n_slots) -> price vector of length n_slots.
HorizonFn = Callable[[pd.DataFrame, pd.Timestamp, int], np.ndarray]


def _horizon_index(t0: pd.Timestamp, n_slots: int) -> pd.DatetimeIndex:
    return pd.date_range(t0, periods=n_slots, freq="15min")


def _leading_valid(arr: np.ndarray) -> int:
    """Number of leading non-NaN entries (how far the horizon data reaches)."""
    nan_pos = np.where(np.isnan(arr))[0]
    return int(nan_pos[0]) if len(nan_pos) else len(arr)


def actual_horizon(df: pd.DataFrame, t0: pd.Timestamp, n_slots: int,
                   *, price_col: str = "price__germany_luxembourg") -> np.ndarray:
    """Perfect-foresight prices over the horizon (for the structural ceiling)."""
    idx = _horizon_index(t0, n_slots)
    return df[price_col].reindex(idx).to_numpy(dtype=float)


def forecast_horizon(df: pd.DataFrame, t0: pd.Timestamp, n_slots: int, *, predictor,
                     price_col: str = "price__germany_luxembourg") -> np.ndarray:
    """XGBoost P50 for the decision day + seasonal-naive proxy beyond it."""
    idx = _horizon_index(t0, n_slots)
    decision_day = t0.tz_convert(BERLIN).date()
    berlin_days = pd.Series(idx.tz_convert(BERLIN).date, index=idx)

    pieces: list[pd.Series] = []
    for dk in sorted(set(berlin_days)):
        issue = issue_time_for(dk)
        if dk == decision_day:
            s = predictor(df, issue)["p50"]
        else:
            s = seasonal_naive_price_predict(df, issue, price_col=price_col)["p50"]
        pieces.append(s)
    full = pd.concat(pieces)
    full = full[~full.index.duplicated(keep="first")]
    return full.reindex(idx).to_numpy(dtype=float)


@dataclass
class MpcResult:
    realised: pd.DataFrame      # per executed slot
    overall: dict               # rev_total, eur_per_mw_yr, n_days


def run_mpc(
    df: pd.DataFrame,
    horizon_fn: HorizonFn,
    spec: BatterySpec,
    start: date,
    end: date,
    *,
    horizon_h: int = 36,
    step_h: int = 24,
    price_col: str = "price__germany_luxembourg",
    annualise_days: float = 365.0,
) -> MpcResult:
    slot_h = spec.slot_hours
    horizon_slots = int(round(horizon_h / slot_h))
    step_slots = int(round(step_h / slot_h))

    t0 = pd.Timestamp(start, tz=BERLIN).tz_convert("UTC")
    stop = pd.Timestamp(end, tz=BERLIN).tz_convert("UTC") + pd.Timedelta(days=1)

    soc_state = spec.soc_min_mwh
    rows: list[pd.DataFrame] = []

    while t0 < stop:
        # Shrink the horizon to the data available from t0 so a longer horizon
        # covers the SAME executed slots as a shorter one (a fair comparison, and
        # what a real operator does at the data edge).
        actual = actual_horizon(df, t0, horizon_slots, price_col=price_col)
        avail = _leading_valid(actual)
        if avail == 0:
            break
        prices_fc = horizon_fn(df, t0, horizon_slots)[:avail]
        avail = min(avail, _leading_valid(prices_fc))
        if avail == 0:
            break

        plan = solve_dispatch(prices_fc[:avail], spec, soc_initial_mwh=soc_state)
        k = min(step_slots, avail)
        ch, dis = plan.charge_mw[:k], plan.discharge_mw[:k]

        rows.append(pd.DataFrame({
            "target_ts": _horizon_index(t0, horizon_slots)[:k],
            "charge_mw": ch,
            "discharge_mw": dis,
            "soc_mwh": plan.soc_mwh[1:k + 1],
            "price_actual": actual[:k],
        }))
        soc_state = plan.soc_mwh[k]
        t0 = t0 + pd.Timedelta(hours=step_h)

    if not rows:
        raise RuntimeError("No MPC steps produced — check date range / coverage.")

    realised = pd.concat(rows, ignore_index=True)
    dt = spec.slot_hours
    rev_total = float(((realised["discharge_mw"] - realised["charge_mw"])
                       * realised["price_actual"] * dt).sum()
                      - spec.deg_cost_eur_per_mwh
                      * (realised["charge_mw"] + realised["discharge_mw"]).sum() * dt)
    n_days = realised.shape[0] * dt / 24.0
    overall = {
        "rev_total": rev_total,
        "eur_per_mw_yr": eur_per_mw_year(rev_total, max(n_days, 1e-9), spec.power_mw, annualise_days),
        "n_days": n_days,
        "horizon_h": horizon_h,
        "step_h": step_h,
    }
    return MpcResult(realised=realised, overall=overall)


__all__ = ["HorizonFn", "MpcResult", "actual_horizon", "forecast_horizon", "run_mpc"]
