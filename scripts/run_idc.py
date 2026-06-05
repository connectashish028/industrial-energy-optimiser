"""Intraday continuous (IDC) — value of intraday access, simplified.

Compares a day-ahead-only oracle to a DA+IDC oracle that routes each slot to the
better of the two markets (sell@max(DA,ID), buy@min(DA,ID)) — the perfect-
foresight ceiling on intraday value. Reports the uplift at a typical DA-ID spread
and its sensitivity to the spread (the structural result).

    python scripts/run_idc.py
    python scripts/run_idc.py --start 2025-10-01 --end 2026-03-31

NOTE: intraday-index prices are REPRESENTATIVE (DA + a calibrated mean-reverting
spread — see data/sources/intraday.py); the full IDC tape is a paid EPEX feed.
The €/MW/yr scales with the assumed DA-ID spread, so read the sensitivity, not a
single number. This is index-based and perfect-foresight — not the continuous
order book or event-driven re-optimisation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

import pandas as pd

from bessopt.config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.data.sources.intraday import DEFAULT_RMS_SPREAD_EUR
from bessopt.market.intraday import run_intraday_value
from bessopt.reporting.charts import plot_intraday_value
from bessopt.risk.degradation import DEFAULT_DEG_COST_EUR_PER_MWH

SPREADS = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Intraday (IDC) value of access (simplified).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date("2026-01-01"))
    ap.add_argument("--end", type=_parse_date, default=_parse_date("2026-03-31"))
    ap.add_argument("--deg-cost", type=float, default=DEFAULT_DEG_COST_EUR_PER_MWH,
                    help="throughput cost (EUR/MWh) applied to BOTH sides — filters "
                         "unrealistic micro-cycling on tiny intraday wiggles")
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    # Apply a degradation cost to both DA-only and DA+IDC so the perfect-foresight
    # battery doesn't micro-cycle on every intraday wiggle (a fair, credible ceiling).
    assets = {n: replace(s, deg_cost_eur_per_mwh=args.deg_cost) for n, s in load_assets().items()}
    price_col = data_cfg["price_col"]
    annual = bt["annualise_days"]
    out_dir = REPO_ROOT / "outputs"

    print(f"\nIntraday (IDC) value of access: {args.start} -> {args.end}  "
          f"(deg cost {args.deg_cost:g} EUR/MWh, both sides)")
    print("Intraday prices are REPRESENTATIVE; read the spread sensitivity, not one number.\n")
    print(f"At the typical DA-ID spread (~{DEFAULT_RMS_SPREAD_EUR:g} EUR/MWh RMS):")
    print(f"{'asset':<8}{'DA-only EUR/MW/yr':>20}{'DA+IDC':>12}{'uplift':>10}{'ID share':>10}")
    print("-" * 60)

    sensitivity: dict[str, list] = {}
    for name, spec in assets.items():
        label = f"{spec.duration_h:.0f}h"
        pts = []
        at_default = None
        for sp in SPREADS:
            r = run_intraday_value(df, spec, args.start, args.end, rms_spread_eur=sp,
                                   price_col=price_col, annualise_days=annual)
            pts.append((sp, r.uplift_eur_per_mw_yr))
            if sp == DEFAULT_RMS_SPREAD_EUR:
                at_default = r
        sensitivity[label] = pts
        r = at_default
        print(f"{name:<8}{r.da_only_eur_per_mw_yr:>20,.0f}{r.two_market_eur_per_mw_yr:>12,.0f}"
              f"{r.uplift_pct:>9.1f}%{r.id_share_of_two_market*100:>9.1f}%")

    chart = plot_intraday_value(sensitivity, out_dir / "intraday_value.png",
                                typical_spread=DEFAULT_RMS_SPREAD_EUR)
    print(f"\nSpread-sensitivity chart: {chart.relative_to(REPO_ROOT)}")
    print("\nIntraday access adds value roughly linearly in the DA-ID spread. Capturing it "
          "for real\nneeds an intraday-index forecast (and, for the full book, event-driven "
          "re-optimisation).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
