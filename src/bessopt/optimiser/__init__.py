"""SoC-aware dispatch optimiser (linopy + HiGHS) - the heart of bessopt."""

from .annualise import eur_per_mw_year
from .lp import settle, solve_dispatch
from .spec import BatterySpec, DispatchResult

__all__ = [
    "BatterySpec",
    "DispatchResult",
    "eur_per_mw_year",
    "settle",
    "solve_dispatch",
]
