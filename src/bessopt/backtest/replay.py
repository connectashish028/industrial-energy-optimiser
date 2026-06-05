"""The value-capture replay engine (M3) — the differentiator.

Connects forecast → optimiser → settlement and produces the headline number:
how much of the perfect-foresight revenue the forecast actually captures.

Mechanism, per delivery day D (a single SoC path that the real battery follows):
  1. Forecast prices for D (P50 via the predictor, through the as-of gate).
  2. **Forecast dispatch:** solve the LP against the forecast → a schedule.
  3. **Settle honestly:** re-price that committed schedule at the *actual* prices
     (do NOT re-optimise) → realised revenue.
  4. **Oracle:** solve the LP against the *actual* prices from the SAME entry SoC.
  5. Advance the (real) battery's SoC using the forecast schedule.

Because both the forecast and the oracle start each day from the same entry SoC,
and the forecast schedule is a feasible point of the oracle's problem, the
oracle revenue dominates the forecast revenue every day — so VCR ∈ [0, 1] and a
perfect predictor yields VCR = 1. That is the same-state "cost of forecast error"
decomposition a desk reasons about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.loader import issue_time_for, target_index_for
from ..optimiser.annualise import eur_per_mw_year
from ..optimiser.lp import settle, solve_dispatch
from ..optimiser.spec import BatterySpec
from . import metrics
from .oracle import daterange

# A predictor: (df, issue_time) -> DataFrame with columns p10/p50/p90.
PriceForecastFn = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]

QUANTILE_LEVELS = (0.10, 0.50, 0.90)


@dataclass
class ReplayResult:
    per_day: pd.DataFrame
    overall: dict
    dispatch_log: pd.DataFrame

    @property
    def vcr(self) -> float:
        return self.overall["vcr"]


def run_replay(
    df: pd.DataFrame,
    predictor: PriceForecastFn,
    spec: BatterySpec,
    start: date,
    end: date,
    *,
    price_col: str = "price__germany_luxembourg",
    carry_soc: bool = True,
    annualise_days: float = 365.0,
    progress: bool = True,
    label: str = "replay",
) -> ReplayResult:
    soc_state = spec.soc_min_mwh
    per_day_rows: list[dict] = []
    log_frames: list[pd.DataFrame] = []
    all_fc_q: list[np.ndarray] = []   # (n, 3) forecast quantiles
    all_actual: list[np.ndarray] = []

    days = list(daterange(start, end))
    it = tqdm(days, desc=label, unit="day") if progress else days

    for delivery in it:
        issue = issue_time_for(delivery)
        target_idx = target_index_for(delivery)
        actual = df[price_col].reindex(target_idx).to_numpy(dtype=float)
        if np.isnan(actual).any():
            continue
        try:
            fc = predictor(df, issue).reindex(target_idx)
        except Exception as e:  # noqa: BLE001 — surface per day, keep going
            if progress:
                tqdm.write(f"[{label}] {delivery}: predictor raised {e!r}; skipping")
            continue
        p50 = fc["p50"].to_numpy(dtype=float)
        if np.isnan(p50).any():
            continue

        entry_soc = soc_state if carry_soc else spec.soc_min_mwh

        # Forecast dispatch → settle at actuals (the honest revenue).
        plan = solve_dispatch(p50, spec, soc_initial_mwh=entry_soc, index=target_idx)
        rev_forecast = settle(plan.charge_mw, plan.discharge_mw, actual, spec)

        # Oracle from the SAME entry SoC (the day's revenue ceiling).
        oracle = solve_dispatch(actual, spec, soc_initial_mwh=entry_soc, index=target_idx)
        rev_oracle = oracle.revenue_eur

        if carry_soc:
            soc_state = plan.soc_mwh[-1]  # the real battery follows the forecast schedule

        per_day_rows.append({
            "date": delivery,
            "rev_oracle": rev_oracle,
            "rev_forecast": rev_forecast,
            "vcr_day": metrics.vcr(rev_forecast, rev_oracle),
            "spread": float(actual.max() - actual.min()),
            "soc_end": plan.soc_mwh[-1],
            "tau_day": metrics.kendall_tau(p50, actual),
            "mae_day": float(np.mean(np.abs(p50 - actual))),
        })
        log_frames.append(pd.DataFrame({
            "target_ts": target_idx,
            "charge_mw": plan.charge_mw,
            "discharge_mw": plan.discharge_mw,
            "soc_mwh": plan.soc_mwh[1:],
            "price_fc": p50,
            "price_actual": actual,
        }))
        all_fc_q.append(fc[["p10", "p50", "p90"]].to_numpy(dtype=float))
        all_actual.append(actual)

    if not per_day_rows:
        raise RuntimeError("No replay days produced — check date range / coverage.")

    per_day = pd.DataFrame(per_day_rows)
    dispatch_log = pd.concat(log_frames, ignore_index=True)

    fc_q = np.vstack(all_fc_q)        # (N, 3)
    actual_all = np.concatenate(all_actual)
    p50_all = fc_q[:, 1]

    rev_oracle_total = float(per_day["rev_oracle"].sum())
    rev_forecast_total = float(per_day["rev_forecast"].sum())
    n_days = len(per_day)
    pw = spec.power_mw

    overall = {
        "asset": f"{spec.duration_h:.0f}h",
        "n_days": n_days,
        "rev_oracle_total": rev_oracle_total,
        "rev_forecast_total": rev_forecast_total,
        "vcr": metrics.vcr(rev_forecast_total, rev_oracle_total),
        "eur_per_mw_yr_oracle": eur_per_mw_year(rev_oracle_total, n_days, pw, annualise_days),
        "eur_per_mw_yr_forecast": eur_per_mw_year(rev_forecast_total, n_days, pw, annualise_days),
        "mae": float(np.mean(np.abs(p50_all - actual_all))),
        "kendall_tau": metrics.kendall_tau(p50_all, actual_all),
        "pinball_p50": metrics.pinball(actual_all, p50_all, 0.50),
        "crps": metrics.crps_from_quantiles(
            actual_all, QUANTILE_LEVELS, [fc_q[:, 0], fc_q[:, 1], fc_q[:, 2]]
        ),
    }
    return ReplayResult(per_day=per_day, overall=overall, dispatch_log=dispatch_log)


__all__ = ["PriceForecastFn", "ReplayResult", "run_replay"]
