"""Risk + degradation overlay — CVaR risk-aware dispatch, linear throughput cost."""

from .cvar import (
    CvarResult,
    ScenarioSet,
    scenarios_from_quantiles,
    scenarios_sampled_from_quantiles,
    solve_cvar_dispatch,
)
from .degradation import (
    DEFAULT_DEG_COST_EUR_PER_MWH,
    WARRANTY_CYCLES_PER_YEAR,
    annual_cycles_to_per_day,
    full_equivalent_cycles,
)

__all__ = [
    "DEFAULT_DEG_COST_EUR_PER_MWH",
    "WARRANTY_CYCLES_PER_YEAR",
    "CvarResult",
    "ScenarioSet",
    "annual_cycles_to_per_day",
    "full_equivalent_cycles",
    "scenarios_from_quantiles",
    "scenarios_sampled_from_quantiles",
    "solve_cvar_dispatch",
]
