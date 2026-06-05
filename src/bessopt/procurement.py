"""Procurement strategy — spot vs fixed vs hedged, cost and risk.

Buying energy is a risk-adjusted decision, not just a cost one. An industrial
consumer can buy on the **spot** market (cheapest on average, but the monthly
bill swings with prices), lock a **fixed-price** contract (certain, but priced
at a forward premium over expected spot), or **hedge** a fraction and leave the
rest on spot. This module quantifies the trade-off over a historical window:

  - annual cost at each hedge ratio (0% = all spot … 100% = all fixed), and
  - risk = volatility of the weekly bill (zero when fully fixed).

It produces a cost-vs-risk frontier — the same risk-adjusted lens as the
battery's CVaR frontier, applied to procurement. Representative: the fixed price
is `avg_spot × (1 + premium)`; a real desk would use the traded forward curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

PRICE_COL = "price__germany_luxembourg"


@dataclass
class ProcurementResult:
    n_days: int
    volume_mwh: float
    avg_spot_eur_mwh: float
    fixed_price_eur_mwh: float
    spot_eur_per_year: float
    fixed_eur_per_year: float
    ratios: np.ndarray              # hedge ratios 0..1
    cost_per_year: np.ndarray       # annual cost at each ratio
    risk_eur: np.ndarray            # weekly-bill std at each ratio
    recommended_ratio: float
    recommended_cost_per_year: float


def procurement_analysis(
    df: pd.DataFrame,
    load_mw: float,
    start: date,
    end: date,
    *,
    price_col: str = PRICE_COL,
    fixed_premium: float = 0.06,
    days_per_year: float = 365.0,
    slot_hours: float = 0.25,
) -> ProcurementResult:
    """Spot/fixed/hedged procurement economics for a flat `load_mw` over [start, end]."""
    s = df[price_col].loc[str(start):str(end)].dropna()
    if s.empty:
        raise RuntimeError("No spot prices in the requested window.")

    dt = slot_hours
    n_slots = len(s)
    n_days = int(round(n_slots * dt / 24))
    volume = load_mw * dt * n_slots                     # MWh over the window
    avg_spot = float(s.mean())                          # flat load ⇒ time-weighted = mean
    fixed_price = avg_spot * (1.0 + fixed_premium)
    scale = days_per_year / max(n_days, 1)

    # Weekly economics. Cost is the total bill; RISK is the volatility of the
    # weekly *unit* price (€/MWh) — so a fully-fixed contract has zero risk
    # (volume variation across partial weeks doesn't masquerade as price risk).
    weekly_spot_bill = (load_mw * s * dt).resample("W").sum()
    weekly_vol = pd.Series(load_mw * dt, index=s.index).resample("W").sum()
    weekly_spot_unit = (weekly_spot_bill / weekly_vol).to_numpy()      # €/MWh per week

    ratios = np.round(np.linspace(0.0, 1.0, 11), 2)
    spot_cost = float(weekly_spot_bill.sum())
    fixed_cost = fixed_price * volume
    cost_year, risk = [], []
    for r in ratios:
        cost_year.append((r * fixed_cost + (1.0 - r) * spot_cost) * scale)
        hedged_unit = r * fixed_price + (1.0 - r) * weekly_spot_unit
        risk.append(float(np.std(hedged_unit)))
    cost_year = np.array(cost_year)
    risk = np.array(risk)

    # Recommendation: the smallest hedge that cuts weekly price volatility to ≤25%
    # of the all-spot level — a realistic industrial budget-certainty target.
    target = 0.25 * risk[0]
    below = np.where(risk <= target + 1e-9)[0]
    rec_i = int(below[0]) if len(below) else len(ratios) - 1

    return ProcurementResult(
        n_days=n_days,
        volume_mwh=volume,
        avg_spot_eur_mwh=avg_spot,
        fixed_price_eur_mwh=fixed_price,
        spot_eur_per_year=float((load_mw * s * dt).sum()) * scale,
        fixed_eur_per_year=fixed_price * volume * scale,
        ratios=ratios,
        cost_per_year=cost_year,
        risk_eur=risk,
        recommended_ratio=float(ratios[rec_i]),
        recommended_cost_per_year=float(cost_year[rec_i]),
    )


__all__ = ["ProcurementResult", "procurement_analysis"]
