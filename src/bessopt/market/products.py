"""German balancing-market products and the PICASSO state-of-charge buffer.

The PICASSO buffer is the constraint that drives the whole 1h-vs-2h story:

  - **aFRR** (capacity) requires holding a **60-minute** energy buffer on the
    relevant end of SoC — POS aFRR (discharge on command) needs energy *above*
    soc_min; NEG aFRR (charge on command) needs room *below* soc_max.
  - **FCR** requires a **15-minute** buffer, symmetric (both ends), because the
    product is symmetric ±.

Committing reserve also consumes *power* headroom: a slot offering r MW of POS
aFRR can only arbitrage-discharge up to P_max − r (it must keep r MW free to
answer activation), and likewise for NEG aFRR / FCR on the charge leg.

A 10 MW / 10 MWh (1h) battery committing ~4 MW symmetric aFRR consumes its whole
usable energy band as buffer and cannot simultaneously arbitrage. A 10 MW /
20 MWh (2h) battery reserves the same MW *and* still has a usable band to cycle.
That gap emerges endogenously from these bounds — it is not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PICASSO energy-buffer durations (hours of full-power activation that must be
# reservable as SoC headroom), per product.
AFRR_BUFFER_H = 1.0    # 60 minutes
FCR_BUFFER_H = 0.25    # 15 minutes


@dataclass(frozen=True)
class ProductSpec:
    """Static description of a balancing product (for the gate-closure simulator)."""
    name: str
    block_hours: float          # FCR / aFRR clear in 4h blocks; DA/intraday in 0.25h
    symmetric: bool             # FCR is symmetric ±; aFRR POS/NEG clear separately
    buffer_h: float             # PICASSO energy buffer this product demands
    pricing: str                # "pay_as_cleared" | "pay_as_bid"


FCR = ProductSpec("FCR", block_hours=4.0, symmetric=True, buffer_h=FCR_BUFFER_H,
                  pricing="pay_as_cleared")
AFRR_CAP = ProductSpec("aFRR_capacity", block_hours=4.0, symmetric=False,
                       buffer_h=AFRR_BUFFER_H, pricing="pay_as_bid")


def expand_blocks(block_values: np.ndarray, n_slots: int, slot_hours: float,
                  block_hours: float = 4.0) -> np.ndarray:
    """Expand per-4h-block MW commitments to a per-slot array of length n_slots."""
    slots_per_block = int(round(block_hours / slot_hours))
    per_slot = np.repeat(np.asarray(block_values, dtype=float), slots_per_block)
    if len(per_slot) < n_slots:                      # pad a short tail (DST / ragged)
        per_slot = np.concatenate([per_slot, np.full(n_slots - len(per_slot), per_slot[-1])])
    return per_slot[:n_slots]


@dataclass(frozen=True)
class ReserveCommitment:
    """Per-slot committed reserve MW (POS/NEG aFRR + symmetric FCR) and the
    PICASSO buffers they imply. All arrays are length H (one per dispatch slot)."""
    afrr_pos_mw: np.ndarray
    afrr_neg_mw: np.ndarray
    fcr_mw: np.ndarray
    afrr_buffer_h: float = AFRR_BUFFER_H
    fcr_buffer_h: float = FCR_BUFFER_H

    @classmethod
    def from_blocks(
        cls,
        n_slots: int,
        slot_hours: float,
        *,
        afrr_pos_blocks: np.ndarray | None = None,
        afrr_neg_blocks: np.ndarray | None = None,
        fcr_blocks: np.ndarray | None = None,
        block_hours: float = 4.0,
    ) -> ReserveCommitment:
        z = np.zeros(n_slots)

        def exp(b):
            return z if b is None else expand_blocks(b, n_slots, slot_hours, block_hours)

        return cls(exp(afrr_pos_blocks), exp(afrr_neg_blocks), exp(fcr_blocks))

    # --- the headroom these commitments consume (per slot) ---
    def discharge_cap(self, power_mw: float) -> np.ndarray:
        """Arbitrage-discharge power left after POS aFRR + FCR."""
        return power_mw - self.afrr_pos_mw - self.fcr_mw

    def charge_cap(self, power_mw: float) -> np.ndarray:
        """Arbitrage-charge power left after NEG aFRR + FCR."""
        return power_mw - self.afrr_neg_mw - self.fcr_mw

    def soc_lower(self, soc_min_mwh: float) -> np.ndarray:
        """SoC floor raised by POS aFRR (60-min) + FCR (15-min) energy buffers."""
        return soc_min_mwh + self.afrr_pos_mw * self.afrr_buffer_h + self.fcr_mw * self.fcr_buffer_h

    def soc_upper(self, soc_max_mwh: float) -> np.ndarray:
        """SoC ceiling lowered by NEG aFRR (60-min) + FCR (15-min) energy buffers."""
        return soc_max_mwh - self.afrr_neg_mw * self.afrr_buffer_h - self.fcr_mw * self.fcr_buffer_h


__all__ = [
    "AFRR_BUFFER_H",
    "AFRR_CAP",
    "FCR",
    "FCR_BUFFER_H",
    "ProductSpec",
    "ReserveCommitment",
    "expand_blocks",
]
