"""On-site PV availability from irradiance.

A simple, transparent PV model: available power = capacity × (irradiance / STC) ×
performance ratio, clipped to the inverter rating. Irradiance is the real
Open-Meteo shortwave-radiation column already in the dataset, so the PV shape is
driven by actual weather, not a synthetic curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STC_IRRADIANCE = 1000.0  # W/m^2 standard test condition
IRRADIANCE_COL = "weather__shortwave_radiation"


def pv_available(
    df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    capacity_mwp: float,
    *,
    performance_ratio: float = 0.80,
    irradiance_col: str = IRRADIANCE_COL,
) -> np.ndarray:
    """Available on-site PV power (MW) over `idx` for a `capacity_mwp` plant."""
    if capacity_mwp <= 0 or irradiance_col not in df.columns:
        return np.zeros(len(idx))
    irr = df[irradiance_col].reindex(idx).to_numpy(dtype=float)
    irr = np.nan_to_num(irr, nan=0.0)
    pv = capacity_mwp * np.clip(irr / STC_IRRADIANCE, 0.0, None) * performance_ratio
    return np.clip(pv, 0.0, capacity_mwp)


__all__ = ["pv_available"]
