"""Degradation overlay — deliberately simple, because that *is* the standard.

Almost every commercial optimiser models battery degradation as a **linear
throughput cost** (a few €/MWh) plus a **cycle cap**, not rainflow counting. For
a daily-cycling merchant asset the linear cost captures the economics and keeps
the problem an LP; rainflow is more accurate in theory but the gain is small and
it complicates the model. We considered it and chose the linear cost on purpose.

Both levers already live on `BatterySpec` and in `solve_dispatch`:
  - `deg_cost_eur_per_mwh` — subtracted per MWh of throughput in the objective.
  - `cycle_cap_per_day`    — caps discharge throughput per solve.
This module just holds the documented defaults and a couple of helpers.
"""

from __future__ import annotations

# LFP defaults. Throughput cost in the widely-cited 2-4 EUR/MWh band; warranty
# cycles ~600-730 full-equivalent cycles/year.
DEFAULT_DEG_COST_EUR_PER_MWH = 3.0
WARRANTY_CYCLES_PER_YEAR = 700.0


def annual_cycles_to_per_day(cycles_per_year: float = WARRANTY_CYCLES_PER_YEAR) -> float:
    """Convert an annual full-equivalent cycle warranty to a per-day cap."""
    return cycles_per_year / 365.0


def full_equivalent_cycles(throughput_mwh: float, energy_mwh: float) -> float:
    """Full-equivalent cycles = total (charge+discharge) throughput / (2 · usable energy)."""
    return throughput_mwh / (2.0 * energy_mwh)


__all__ = [
    "DEFAULT_DEG_COST_EUR_PER_MWH",
    "WARRANTY_CYCLES_PER_YEAR",
    "annual_cycles_to_per_day",
    "full_equivalent_cycles",
]
