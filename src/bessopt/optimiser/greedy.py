"""Greedy ranking dispatch — the documented baseline the LP must beat.

Vendored from the `loadforecast` project's `dispatch.py`. It ranks slots by
price and charges the cheapest / discharges the most expensive, subject only to
a daily throughput budget. It has **no SoC trajectory and no inter-temporal
feasibility** — that is precisely the gap the LP (`solve_dispatch`) closes, so
keeping the greedy as a baseline makes the LP's value legible.

Quantile twist: when a probabilistic forecast is available, pass P10 prices as
the charge signal and P90 as the discharge signal (cheap-tail / expensive-tail
estimates) rather than the median for both.
"""

from __future__ import annotations

import numpy as np

from .spec import BatterySpec


def _slots_per_direction(spec: BatterySpec, cycles_per_day: float, horizon_slots: int) -> int:
    """How many slots may we charge (or discharge), given a cycles/day budget?"""
    energy_per_slot = spec.power_mw * spec.slot_hours
    days = horizon_slots * spec.slot_hours / 24.0
    budget_mwh = cycles_per_day * spec.energy_mwh * max(days, spec.slot_hours / 24.0)
    return int(budget_mwh / energy_per_slot)


def dispatch_pnl(
    charge_signal: np.ndarray,
    discharge_signal: np.ndarray,
    actual_prices: np.ndarray,
    spec: BatterySpec,
    *,
    cycles_per_day: float = 1.5,
) -> dict:
    """Greedy dispatch + realised P&L against ``actual_prices``.

    Returns a dict with charge/discharge slot indices, costs/revenue, net P&L,
    and realised cycles. Use as a baseline comparison for the LP.
    """
    charge_signal = np.asarray(charge_signal, dtype=float)
    discharge_signal = np.asarray(discharge_signal, dtype=float)
    actual_prices = np.asarray(actual_prices, dtype=float)
    n = _slots_per_direction(spec, cycles_per_day, len(actual_prices))

    charge_idx = set(np.argsort(charge_signal)[:n].tolist())
    discharge_idx = set(np.argsort(discharge_signal)[-n:].tolist())

    # A slot in both lists is dropped from both — we'd cycle against ourselves.
    overlap = charge_idx & discharge_idx
    charge_idx = sorted(charge_idx - overlap)
    discharge_idx = sorted(discharge_idx - overlap)

    e = spec.power_mw * spec.slot_hours          # MWh per full-power slot
    rte = spec.eta_rt
    cost = float(sum(actual_prices[i] * e for i in charge_idx))
    revenue = float(sum(actual_prices[i] * e * rte for i in discharge_idx))
    net = revenue - cost
    n_cycles = (len(discharge_idx) * e) / spec.energy_mwh

    return {
        "charge_slots": charge_idx,
        "discharge_slots": discharge_idx,
        "charge_cost": cost,
        "discharge_revenue": revenue,
        "net_pnl": net,
        "n_cycles_realised": float(n_cycles),
    }


__all__ = ["dispatch_pnl"]
