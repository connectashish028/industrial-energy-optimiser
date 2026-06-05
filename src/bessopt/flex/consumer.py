"""Industrial consumer specification — the demand-side asset to optimise.

Models a large industrial site that buys power from the grid at the EPEX
day-ahead price and has flexibility to move when it consumes:

  - a flat **inflexible baseline** load (the process that must always run),
  - a **flexible on/off process** that must accumulate a number of run-hours per
    day but can be scheduled to the cheapest / sunniest hours (the MILP knob),
  - on-site **PV** (self-consumed or curtailed; no export by default),
  - an optional on-site **battery** for peak-shaving / price arbitrage,
  - a **grid-connection limit**.

The optimiser minimises annual procurement cost; the saving vs running the
flexible process on a fixed day-shift is the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..optimiser.spec import BatterySpec


def _default_battery() -> BatterySpec:
    return BatterySpec(power_mw=10.0, energy_mwh=20.0)


@dataclass(frozen=True)
class ConsumerSpec:
    baseline_load_mw: float = 8.0          # inflexible base load (always on)
    proc_power_mw: float = 6.0             # flexible process rated power (on/off)
    proc_hours_per_day: float = 8.0        # required run-hours per day
    pv_capacity_mwp: float = 10.0          # on-site PV
    grid_limit_mw: float = 25.0            # grid-connection import limit
    battery: BatterySpec | None = field(default_factory=_default_battery)
    naive_proc_start_hour: float = 8.0     # baseline: fixed day-shift start (Berlin)
    slot_hours: float = 0.25

    @property
    def proc_slots(self) -> int:
        return int(round(self.proc_hours_per_day / self.slot_hours))

    @property
    def peak_load_mw(self) -> float:
        return self.baseline_load_mw + self.proc_power_mw


__all__ = ["ConsumerSpec"]
