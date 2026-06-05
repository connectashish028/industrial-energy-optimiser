"""Phase 5 / M5 deliverable — the value stack: revenue by market, 1h vs 2h.

Co-optimises day-ahead arbitrage + FCR + aFRR capacity under the PICASSO SoC
buffer, day by day over the window, for both assets. Prints the revenue split
and saves the stacked revenue-by-market bar chart. The 1h-vs-2h gap emerges
endogenously from the buffer: the 1h battery leans on reserve (it can barely
arbitrage once it holds aFRR), the 2h stacks both.

    python scripts/run_valuestack.py
    python scripts/run_valuestack.py --start 2025-10-01 --end 2026-03-31

NOTE: reserve prices are REPRESENTATIVE (see data/sources/regelleistung.py) unless
the live regelleistung.net feed is wired and BESSOPT_REGELLEISTUNG_LIVE=1 is set.
Treat the reserve revenue magnitudes as illustrative of the mechanism, not actuals.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from bessopt.config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.market.simulator import run_value_stack
from bessopt.reporting.charts import plot_revenue_by_market


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Value-stack revenue-by-market (M5).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date(bt["test_start"]))
    ap.add_argument("--end", type=_parse_date, default=_parse_date(bt["test_end"]))
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    assets = load_assets()
    price_col = data_cfg["price_col"]
    annual_days = bt["annualise_days"]
    out_dir = REPO_ROOT / "outputs"

    print(f"\nValue stack (co-optimised DA + FCR + aFRR): {args.start} -> {args.end}")
    print("Reserve prices are REPRESENTATIVE (mechanism demo), not live regelleistung.net.\n")
    cols = ["day_ahead", "fcr", "afrr_pos", "afrr_neg", "TOTAL"]
    print(f"{'asset':<8}" + "".join(f"{c:>14}" for c in cols) + f"{'resShare':>10}")
    print("-" * (8 + 14 * 5 + 10))

    summary: dict[str, dict] = {}
    for name, spec in assets.items():
        run = run_value_stack(df, spec, args.start, args.end, price_col=price_col,
                              carry_soc=bt["replay"]["carry_soc"], annualise_days=annual_days,
                              progress=True)
        py = run.eur_per_mw_yr_by_stream
        label = f"{spec.duration_h:.0f}h"
        summary[label] = py
        total = run.total_eur_per_mw_yr
        res_share = (total - py["day_ahead"]) / total if total else float("nan")
        row = [py["day_ahead"], py["fcr"], py["afrr_pos"], py["afrr_neg"], total]
        print(f"{name:<8}" + "".join(f"{v:>14,.0f}" for v in row) + f"{res_share*100:>9.1f}%")

    chart = plot_revenue_by_market(summary, out_dir / "value_stack_by_market.png")
    print(f"\nRevenue-by-market chart: {chart.relative_to(REPO_ROOT)}")
    print("\nThe reserve share is higher for the 1h asset — the PICASSO 60-min aFRR buffer "
          "consumes\nits usable energy, so it tilts to reserve while the 2h asset stacks both.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
