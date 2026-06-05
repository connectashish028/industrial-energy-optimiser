"""Perfect-foresight oracle over a window — the upper bound on revenue.

Feeds *actual* prices into the LP for each delivery day (carrying SoC across
days), and reports total revenue, €/MW/year, and the per-day breakdown. This is
the yardstick every forecast is measured against (the oracle in VCR = R_fc / R_oracle).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.loader import target_index_for
from ..optimiser.annualise import eur_per_mw_year
from ..optimiser.lp import solve_dispatch
from ..optimiser.spec import BatterySpec, DispatchResult


def daterange(start: date, end: date, step_days: int = 1) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step_days)


def day_prices(df: pd.DataFrame, delivery: date, price_col: str) -> np.ndarray | None:
    """Actual price vector for the delivery day, or None if incomplete."""
    idx = target_index_for(delivery)
    s = df[price_col].reindex(idx)
    if s.isna().any():
        return None
    return s.to_numpy(dtype=float)


@dataclass
class OracleResult:
    per_day: pd.DataFrame          # date, revenue, spread, cycles, soc_end
    total_revenue_eur: float
    eur_per_mw_year: float
    n_days: int
    best_day: date                 # highest-spread day (for the SoC plot)
    best_day_dispatch: DispatchResult


def run_oracle(
    df: pd.DataFrame,
    spec: BatterySpec,
    start: date,
    end: date,
    *,
    price_col: str = "price__germany_luxembourg",
    carry_soc: bool = True,
    annualise_days: float = 365.0,
) -> OracleResult:
    soc_state = spec.soc_min_mwh
    rows = []
    best_spread = -np.inf
    best_day = None
    best_dispatch = None

    for delivery in daterange(start, end):
        prices = day_prices(df, delivery, price_col)
        if prices is None:
            continue
        res = solve_dispatch(
            prices, spec, soc_initial_mwh=(soc_state if carry_soc else None),
            index=target_index_for(delivery),
        )
        spread = float(prices.max() - prices.min())
        rows.append({
            "date": delivery,
            "revenue": res.revenue_eur,
            "spread": spread,
            "cycles": res.cycles(spec),
            "soc_end": res.soc_mwh[-1],
        })
        if carry_soc:
            soc_state = res.soc_mwh[-1]
        if spread > best_spread:
            best_spread, best_day, best_dispatch = spread, delivery, res

    if not rows:
        raise RuntimeError("No oracle days produced — check date range / price coverage.")

    per_day = pd.DataFrame(rows)
    total = float(per_day["revenue"].sum())
    n_days = len(per_day)
    return OracleResult(
        per_day=per_day,
        total_revenue_eur=total,
        eur_per_mw_year=eur_per_mw_year(total, n_days, spec.power_mw, annualise_days),
        n_days=n_days,
        best_day=best_day,
        best_day_dispatch=best_dispatch,
    )


__all__ = ["OracleResult", "daterange", "day_prices", "run_oracle"]
