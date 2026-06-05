"""Interactive what-if simulation for a user-defined BESS.

Given any `BatterySpec` (power, energy, efficiency, SoC band, degradation cost),
run the perfect-foresight oracle and the co-optimised value stack over a recent
window and return the economics — the engine behind the dashboard's "Simulate
your BESS" tab. Kept out of the Streamlit file so it is testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .backtest.oracle import run_oracle
from .backtest.replay import run_replay
from .data.loader import target_index_for
from .forecast.predict import xgboost_price_predict_full
from .market.simulator import run_value_stack
from .optimiser.spec import BatterySpec, DispatchResult

PRICE_COL = "price__germany_luxembourg"


def last_complete_day(df: pd.DataFrame, price_col: str = PRICE_COL) -> date:
    """Most recent delivery day whose actual prices are all present."""
    d = df.index.max().tz_convert("Europe/Berlin").date()
    for _ in range(15):
        if not df[price_col].reindex(target_index_for(d)).isna().any():
            return d
        d -= timedelta(days=1)
    raise RuntimeError("No complete actual delivery day found near the data edge.")


def default_window(df: pd.DataFrame, window_days: int, price_col: str = PRICE_COL) -> tuple[date, date]:
    end = last_complete_day(df, price_col)
    return end - timedelta(days=window_days), end


@dataclass
class SimResult:
    spec: BatterySpec
    start: date
    end: date
    n_days: int
    oracle_eur_per_mw_yr: float
    avg_cycles_per_day: float
    best_day: date
    best_day_dispatch: DispatchResult
    value_stack_eur_per_mw_yr: dict | None      # stream -> EUR/MW/yr (or None)
    value_stack_total_eur_per_mw_yr: float | None
    forecast_eur_per_mw_yr: float | None = None  # realistic DA-only forecast-driven (or None)
    vcr: float | None = None                     # value-capture ratio (forecast / oracle)


def simulate_asset(
    spec: BatterySpec,
    df: pd.DataFrame,
    start: date,
    end: date,
    *,
    price_col: str = PRICE_COL,
    annualise_days: float = 365.0,
    with_value_stack: bool = True,
    with_forecast: bool = False,
) -> SimResult:
    """Oracle economics (+ optional value stack + optional forecast-driven VCR).

    Both the oracle and the value stack respect `spec.deg_cost_eur_per_mwh`, so
    moving the degradation cost changes cycling and revenue as a desk would model
    it. `with_forecast` runs the honest forecast→optimise→settle replay (slower).
    """
    oracle = run_oracle(df, spec, start, end, price_col=price_col, annualise_days=annualise_days)

    vs_dict = vs_total = None
    if with_value_stack:
        vs = run_value_stack(df, spec, start, end, price_col=price_col,
                             annualise_days=annualise_days, progress=False)
        vs_dict = vs.eur_per_mw_yr_by_stream
        vs_total = vs.total_eur_per_mw_yr

    fc_yr = vcr = None
    if with_forecast:
        rep = run_replay(df, xgboost_price_predict_full, spec, start, end, price_col=price_col,
                         annualise_days=annualise_days, progress=False, label="sim")
        fc_yr = rep.overall["eur_per_mw_yr_forecast"]
        vcr = rep.overall["vcr"]

    return SimResult(
        spec=spec,
        start=start,
        end=end,
        n_days=oracle.n_days,
        oracle_eur_per_mw_yr=oracle.eur_per_mw_year,
        avg_cycles_per_day=float(oracle.per_day["cycles"].mean()),
        best_day=oracle.best_day,
        best_day_dispatch=oracle.best_day_dispatch,
        value_stack_eur_per_mw_yr=vs_dict,
        value_stack_total_eur_per_mw_yr=vs_total,
        forecast_eur_per_mw_yr=fc_yr,
        vcr=vcr,
    )


__all__ = ["SimResult", "default_window", "last_complete_day", "simulate_asset"]
