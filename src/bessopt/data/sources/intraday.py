"""Intraday continuous (IDC) index price — representative, with a real-API stub.

The IDC market is a continuous order book trading until ~5 min before delivery;
its published *index* prices (ID1 / ID3 = volume-weighted averages of the last
1h / 3h of trading) are the simplified signal we use — not the full order book.

DATA STATUS — read before trusting magnitudes:
  The full IDC trade tape is a paid EPEX feed, and the no-token sources
  (SMARD / Energy-Charts) expose day-ahead, not the continuous index, at a
  resolution we can rely on here. So `intraday_index` returns a **representative**
  series: the day-ahead price plus a mean-reverting deviation calibrated to a
  configurable DA-ID RMS spread (~10-15 EUR/MWh is typical for DE). This is
  deterministic and clearly synthetic. The *mechanism* (a second price per
  delivery slot the battery can route to) and the *structural* result (intraday
  value scales with the DA-ID spread) are real; the absolute € is illustrative.
  `_fetch_live` sketches the real SMARD/EPEX path; verify and wire it to replace
  the representative series.

Intraday index prices for a delivery slot are set near delivery, so they are NOT
known at the D-1 12:00 day-ahead gate — a forecaster must not use day-D intraday
prices as features. The value-of-intraday-access oracle uses them with perfect
foresight (an explicit upper bound), which is fine because it is an oracle.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Typical DE day-ahead vs intraday-continuous index RMS spread (EUR/MWh).
DEFAULT_RMS_SPREAD_EUR = 12.0
_AR1_PHI = 0.8   # within-day persistence of the DA-ID deviation


def intraday_index(
    da_prices: np.ndarray,
    *,
    rms_spread_eur: float = DEFAULT_RMS_SPREAD_EUR,
    seed: int = 0,
) -> np.ndarray:
    """Representative intraday-index prices for one delivery day.

    ID[t] = DA[t] + d[t], where d is a zero-mean AR(1) deviation with the given
    RMS spread (so on average ID ≈ DA, but each slot can deviate — the source of
    intraday arbitrage). Deterministic given (da_prices, rms_spread, seed).
    """
    if os.environ.get("BESSOPT_INTRADAY_LIVE") == "1":  # pragma: no cover
        return _fetch_live(da_prices)

    da = np.asarray(da_prices, dtype=float)
    H = len(da)
    if rms_spread_eur <= 0:
        return da.copy()
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal(H)
    d = np.empty(H)
    d[0] = innov[0]
    for t in range(1, H):
        d[t] = _AR1_PHI * d[t - 1] + np.sqrt(1 - _AR1_PHI**2) * innov[t]
    # Scale to the target RMS spread.
    d *= rms_spread_eur / (np.sqrt(np.mean(d**2)) + 1e-9)
    return da + d


def intraday_index_for_index(
    da_series: pd.Series,
    *,
    rms_spread_eur: float = DEFAULT_RMS_SPREAD_EUR,
    seed: int = 0,
) -> pd.Series:
    """As `intraday_index`, aligned to a DA price Series' index."""
    vals = intraday_index(da_series.to_numpy(), rms_spread_eur=rms_spread_eur, seed=seed)
    return pd.Series(vals, index=da_series.index, name="intraday_id__price")


def _fetch_live(da_prices: np.ndarray) -> np.ndarray:  # pragma: no cover - needs data feed
    """Real intraday-index client — STUB. Wire SMARD's intraday-continuous index
    (ID1/ID3) filter or an EPEX feed, align to the delivery slots, and return it."""
    raise NotImplementedError(
        "Live intraday-index ingestion is not wired. Add the SMARD/EPEX intraday "
        "filter, then set BESSOPT_INTRADAY_LIVE=1."
    )


__all__ = ["DEFAULT_RMS_SPREAD_EUR", "intraday_index", "intraday_index_for_index"]
