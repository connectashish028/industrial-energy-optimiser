"""Battery specification + dispatch result containers.

The 1h and 2h assets differ **only** in `energy_mwh` (10 vs 20). Duration =
`energy_mwh / power_mw`. Running the identical optimiser at both capacities is
the cleanest way to surface the duration economics — and, once the PICASSO
reserve buffer lands (Phase 5), to derive the 1h-vs-2h revenue gap from the
constraint rather than asserting it.

Efficiency convention: the round-trip efficiency `eta_rt` is split symmetrically
into charge and discharge legs, `eta_c = eta_d = sqrt(eta_rt)`. With
`eta_rt = 0.90` (a fine LFP default), each leg is ~0.9487.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatterySpec:
    power_mw: float = 10.0
    energy_mwh: float = 20.0          # duration = energy_mwh / power_mw (1h: 10, 2h: 20)
    eta_rt: float = 0.90             # round-trip efficiency; split eta_c = eta_d = sqrt(eta_rt)
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    deg_cost_eur_per_mwh: float = 0.0   # Phase 6 hook: linear throughput cost (2-4 for LFP)
    cycle_cap_per_day: float | None = None   # Phase 6 hook: e.g. 2.0 caps ~730 cycles/yr
    slot_hours: float = 0.25            # 15-minute market time unit

    # --- derived ---
    @property
    def eta_c(self) -> float:
        return self.eta_rt ** 0.5

    @property
    def eta_d(self) -> float:
        return self.eta_rt ** 0.5

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def usable_mwh(self) -> float:
        return (self.soc_max_frac - self.soc_min_frac) * self.energy_mwh


@dataclass
class DispatchResult:
    charge_mw: np.ndarray        # (H,)   grid draw per slot, MW
    discharge_mw: np.ndarray     # (H,)   grid inject per slot, MW
    soc_mwh: np.ndarray          # (H+1,) trajectory incl. initial and terminal, MWh
    prices_used: np.ndarray      # (H,)   the price vector optimised against, EUR/MWh
    revenue_eur: float           # net: gross arbitrage minus degradation cost
    throughput_mwh: float        # sum(charge + discharge) * dt
    status: str                  # solver termination condition ("optimal", ...)
    index: object | None = None  # optional pd.DatetimeIndex of the H target slots

    def cycles(self, spec: BatterySpec) -> float:
        """Full-equivalent cycles = total throughput / (2 * usable energy span)."""
        return self.throughput_mwh / (2.0 * spec.energy_mwh)


__all__ = ["BatterySpec", "DispatchResult"]
