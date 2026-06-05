"""Industrial flexibility — cost-optimal scheduling of a >10 MW consumer.

Minimises EPEX day-ahead procurement cost for an industrial site (flat baseline +
flexible on/off process + on-site PV + battery) via a MILP, and reports the
annual saving vs running the flexible process on a fixed day-shift. Saves a
schedule chart for the highest-saving day.

    python scripts/run_flex.py
    python scripts/run_flex.py --start 2026-01-01 --end 2026-02-28 --proc-mw 6 --proc-hours 8
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from bessopt.config import REPO_ROOT, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.flex import ConsumerSpec, run_flex
from bessopt.optimiser.spec import BatterySpec
from bessopt.reporting.charts import plot_flex_schedule


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    data_cfg = load_data_config()
    ap = argparse.ArgumentParser(description="Industrial flexibility cost optimiser (MILP).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date("2026-01-01"))
    ap.add_argument("--end", type=_parse_date, default=_parse_date("2026-02-28"))
    ap.add_argument("--baseline-mw", type=float, default=8.0)
    ap.add_argument("--proc-mw", type=float, default=6.0)
    ap.add_argument("--proc-hours", type=float, default=8.0)
    ap.add_argument("--pv-mwp", type=float, default=10.0)
    ap.add_argument("--batt-mw", type=float, default=10.0)
    ap.add_argument("--batt-mwh", type=float, default=20.0)
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    battery = BatterySpec(power_mw=args.batt_mw, energy_mwh=args.batt_mwh) if args.batt_mw > 0 else None
    spec = ConsumerSpec(baseline_load_mw=args.baseline_mw, proc_power_mw=args.proc_mw,
                        proc_hours_per_day=args.proc_hours, pv_capacity_mwp=args.pv_mwp,
                        battery=battery)
    run = run_flex(df, spec, args.start, args.end, price_col=data_cfg["price_col"])

    site = (f"{spec.peak_load_mw:.0f} MW peak ({spec.baseline_load_mw:.0f} base + "
            f"{spec.proc_power_mw:.0f} flex×{spec.proc_hours_per_day:.0f}h), "
            f"{spec.pv_capacity_mwp:.0f} MWp PV"
            + (f", {args.batt_mw:.0f} MW/{args.batt_mwh:.0f} MWh battery" if battery else ""))
    print(f"\nIndustrial flex — {site}")
    print(f"Window {args.start} → {args.end} ({run.n_days} days), DA spot procurement.\n")
    print(f"  {'':<18}{'cost (window)':>16}{'€/year':>16}")
    print("  " + "-" * 50)
    print(f"  {'Naive (fixed shift)':<18}{run.naive_cost_eur:>16,.0f}{run.naive_eur_per_year:>16,.0f}")
    print(f"  {'Optimised (MILP)':<18}{run.optimised_cost_eur:>16,.0f}{run.optimised_eur_per_year:>16,.0f}")
    print(f"  {'SAVINGS':<18}{run.savings_eur:>16,.0f}{run.savings_eur_per_year:>16,.0f}")
    print(f"\n  → {run.savings_pct:.1f}% of procurement cost — "
          f"€{run.savings_eur_per_year:,.0f}/year for this site.")

    r = run.best_day_result
    naive_day_cost = float((run.best_day_naive_grid * r.spot * spec.slot_hours).sum())
    chart = plot_flex_schedule(
        r, run.best_day_naive_grid, spec,
        title=(f"Industrial flex — {run.best_day} — optimised €{r.cost_eur:,.0f} vs "
               f"naive €{naive_day_cost:,.0f} (saved €{naive_day_cost - r.cost_eur:,.0f})"),
        out_path=REPO_ROOT / "outputs" / "flex_schedule.png",
    )
    print(f"  Schedule chart: {chart.relative_to(REPO_ROOT)}")
    print("\nDA spot only; PV from real irradiance; representative consumer. The flexible "
          "process\nis scheduled into the cheapest/sunniest hours, the battery peak-shaves.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
