"""Industrial flexibility — demand-side MILP cost optimisation (industrial cost-min)."""

from .consumer import ConsumerSpec
from .optimiser import FlexResult, FlexRun, naive_cost, optimise_day, run_flex
from .pv import pv_available

__all__ = [
    "ConsumerSpec",
    "FlexResult",
    "FlexRun",
    "naive_cost",
    "optimise_day",
    "pv_available",
    "run_flex",
]
