"""Reserve-market capacity prices (FCR + aFRR) — regelleistung.net.

Provides the three reserve capacity-price columns the value-stack needs:
  reserve_fcr__capacity_price            (symmetric ±, EUR/MW/h)
  reserve_afrr_cap__pos__capacity_price  (POS, EUR/MW/h)
  reserve_afrr_cap__neg__capacity_price  (NEG, EUR/MW/h)

These clear in 6 × 4h blocks per day at the D-1 morning gates (FCR ~08:00,
aFRR ~09:00 CET), so the cleared prices for delivery day D are known before the
D-1 12:00 day-ahead issue time — i.e. usable as features/inputs at decision time.

DATA SOURCE STATUS — read this before trusting the numbers:
  The live regelleistung.net API (the "datacenter" tender-results endpoint) is
  format-volatile and partly gated, and product definitions change frequently.
  `_fetch_live` sketches the real client; it is OFF by default. Unless the env
  var BESSOPT_REGELLEISTUNG_LIVE=1 is set (and the endpoint verified), `fetch`
  returns a **representative** price series — deterministic, calibrated to
  typical 2024-2026 German levels with a realistic time-of-day block shape, but
  NOT the real cleared prices. Every value-stack number built on it must be
  labelled "representative reserve prices" until the live feed is wired and
  verified.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..schema import Column

BLOCK_HOURS = 4.0
N_BLOCKS = 6                       # 6 × 4h blocks per day
BERLIN = "Europe/Berlin"

# Representative capacity-price levels (EUR/MW/h), base + per-block multipliers
# over the 6 daily blocks (00-04, 04-08, 08-12, 12-16, 16-20, 20-24 local).
# Calibrated to typical 2024-2026 German magnitudes; see module docstring caveat.
_LEVELS = {
    "reserve_fcr__capacity_price":           (8.0,  [0.8, 0.9, 1.1, 1.0, 1.2, 1.0]),
    "reserve_afrr_cap__pos__capacity_price": (10.0, [0.7, 0.9, 1.2, 1.0, 1.3, 1.1]),
    "reserve_afrr_cap__neg__capacity_price": (6.0,  [1.2, 1.1, 0.9, 0.8, 0.9, 1.1]),
}

RESERVE_COLUMNS = tuple(_LEVELS.keys())


def _block_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """4h block-start timestamps (Berlin-aligned) covering [start, end), in UTC."""
    s = pd.Timestamp(start).tz_convert(BERLIN).normalize()
    e = pd.Timestamp(end).tz_convert(BERLIN)
    idx = pd.date_range(s, e, freq=f"{int(BLOCK_HOURS)}h", inclusive="left")
    return idx.tz_convert("UTC")


def _representative_prices(col: str, idx: pd.DatetimeIndex) -> np.ndarray:
    """Representative price for `col` at every timestamp in `idx` (constant within
    each Berlin 4h block — works on a block index or a fine 15-min index alike)."""
    base, block_mult = _LEVELS[col]
    local = idx.tz_convert(BERLIN)
    block_of_day = (local.hour // int(BLOCK_HOURS)).to_numpy()
    mult = np.array(block_mult)[block_of_day]
    # Deterministic day-to-day variation from the date ordinal (no RNG state).
    day_ord = local.normalize().asi8 // (24 * 3600 * 10**9)
    season = 1.0 + 0.15 * np.sin(2 * np.pi * (day_ord % 365) / 365.0)
    jitter = 1.0 + 0.10 * np.sin(day_ord.astype(float) * 12.9898)  # pseudo-random, deterministic
    weekend = np.where(local.dayofweek.to_numpy() >= 5, 0.9, 1.0)
    return np.clip(base * mult * season * jitter * weekend, 0.0, None)


def reserve_prices_for_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Reserve capacity prices (EUR/MW/h) aligned to an arbitrary 15-min index.

    This is the primary API for the value-stack simulator. Representative unless
    the live feed is enabled (see module docstring).
    """
    if os.environ.get("BESSOPT_REGELLEISTUNG_LIVE") == "1":  # pragma: no cover
        blocks = _fetch_live(idx)
        return blocks.reindex(idx, method="ffill")
    return pd.DataFrame({c: _representative_prices(c, idx) for c in RESERVE_COLUMNS}, index=idx)


def reserve_block_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Per-4h-block reserve capacity prices (EUR/MW/h), indexed by block-start (UTC)."""
    return reserve_prices_for_index(_block_index(start, end))


def fetch(column: Column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Source-interface fetch: a 15-min Series for one reserve column."""
    idx15 = pd.date_range(pd.Timestamp(start).floor("15min"), end, freq="15min", inclusive="left")
    return reserve_prices_for_index(idx15)[column.name].rename(column.name)


def _fetch_live(blocks: pd.DatetimeIndex) -> pd.DataFrame:  # pragma: no cover - needs network
    """Real regelleistung.net datacenter client — STUB, verify before enabling.

    The current API exposes tender results per product/date; the exact endpoint,
    auth, and JSON schema must be checked against live docs before relying on it
    (it has changed repeatedly). Wire it here, map cleared marginal prices to the
    three RESERVE_COLUMNS in EUR/MW/h, and return a per-block DataFrame.
    """
    raise NotImplementedError(
        "Live regelleistung.net ingestion is not wired. Verify the datacenter API "
        "endpoint/schema, implement here, and set BESSOPT_REGELLEISTUNG_LIVE=1."
    )


__all__ = [
    "RESERVE_COLUMNS",
    "fetch",
    "reserve_block_prices",
    "reserve_prices_for_index",
]
