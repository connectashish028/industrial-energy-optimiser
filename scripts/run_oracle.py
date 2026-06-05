"""Phase 1 / M1 deliverable — perfect-foresight oracle.

Runs the SoC-aware LP against ACTUAL day-ahead prices over the evaluation
window for both the 1h and 2h asset, prints perfect-foresight €/MW/year, and
saves a SoC-over-time plot for the highest-spread day (where you can see the
battery charge cheap and discharge into the spike).

    python scripts/run_oracle.py
    python scripts/run_oracle.py --start 2025-10-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from bessopt.backtest.oracle import run_oracle
from bessopt.config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.reporting.charts import plot_soc_dispatch


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Perfect-foresight oracle (M1).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date(bt["test_start"]))
    ap.add_argument("--end", type=_parse_date, default=_parse_date(bt["test_end"]))
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    assets = load_assets()
    price_col = data_cfg["price_col"]
    annual_days = bt["annualise_days"]
    out_dir = REPO_ROOT / "outputs"

    print(f"\nOracle window: {args.start} -> {args.end}  (price = {price_col})\n")
    print(f"{'asset':<10}{'days':>6}{'EUR total':>14}{'EUR/MW/yr':>14}{'avg cyc':>10}")
    print("-" * 54)

    for name, spec in assets.items():
        res = run_oracle(
            df, spec, args.start, args.end,
            price_col=price_col, carry_soc=bt["replay"]["carry_soc"],
            annualise_days=annual_days,
        )
        avg_cycles = res.per_day["cycles"].mean()
        print(f"{name:<10}{res.n_days:>6}{res.total_revenue_eur:>14,.0f}"
              f"{res.eur_per_mw_year:>14,.0f}{avg_cycles:>10.2f}")

        plot_path = out_dir / f"oracle_soc_{name}_{res.best_day}.png"
        plot_soc_dispatch(
            res.best_day_dispatch, spec,
            title=f"Oracle dispatch — {name} ({spec.duration_h:.0f}h, {spec.energy_mwh:.0f} MWh) "
                  f"— {res.best_day} (highest-spread day)",
            out_path=plot_path,
        )
        print(f"           SoC plot: {plot_path.relative_to(REPO_ROOT)}")

    print("\nNote: day-ahead arbitrage only, price-taker, perfect fills. "
          "Full-stack (aFRR/intraday) numbers come in M5.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
