"""Annualisation helpers — turn a backtest-window revenue into €/MW/year.

Every reported number must state its scope. €/MW/year scales a window's revenue
to a full year and normalises by rated power, so the 1h and 2h assets (same
power, different energy) are compared on the metric a desk quotes.
"""

from __future__ import annotations


def eur_per_mw_year(
    total_revenue_eur: float,
    n_days: int,
    power_mw: float,
    days_per_year: float = 365.0,
) -> float:
    """Annualise a window's total revenue to €/MW/year.

    total_revenue / n_days  → €/day, × days_per_year → €/year, / power → €/MW/year.
    """
    if n_days <= 0 or power_mw <= 0:
        raise ValueError("n_days and power_mw must be positive")
    return total_revenue_eur / n_days * days_per_year / power_mw


__all__ = ["eur_per_mw_year"]
