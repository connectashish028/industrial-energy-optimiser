"""Phase 3 / M3 deliverable — the headline Value-Capture chart.

For each asset (1h, 2h): replay the XGBoost price forecast through the SoC-aware
optimiser, settle against actual prices, and compare to the perfect-foresight
oracle. Produces the money chart (oracle vs forecast-driven €/MW/yr, VCR
annotated) and prints VCR / Kendall tau / MAE, with the seasonal-naive baseline
for context.

    python scripts/run_vcr.py
    python scripts/run_vcr.py --start 2025-10-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from bessopt.backtest.baselines import seasonal_naive_price_predict
from bessopt.backtest.replay import run_replay
from bessopt.config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.forecast.predict import xgboost_price_predict_full
from bessopt.reporting.charts import plot_headline_vcr


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Headline value-capture chart (M3).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date(bt["test_start"]))
    ap.add_argument("--end", type=_parse_date, default=_parse_date(bt["test_end"]))
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    assets = load_assets()
    price_col = data_cfg["price_col"]
    annual_days = bt["annualise_days"]
    carry = bt["replay"]["carry_soc"]
    out_dir = REPO_ROOT / "outputs"

    print(f"\nValue-capture replay: {args.start} -> {args.end}  (price = {price_col})\n")
    header = f"{'asset':<8}{'oracle €/MW/yr':>16}{'forecast':>12}{'VCR':>8}{'naiveVCR':>10}{'tau':>7}{'MAE':>8}"
    print(header.replace("€", "EUR"))
    print("-" * len(header))

    summary: dict[str, dict] = {}
    for name, spec in assets.items():
        fc = run_replay(df, xgboost_price_predict_full, spec, args.start, args.end,
                        price_col=price_col, carry_soc=carry, annualise_days=annual_days,
                        progress=True, label=f"xgb:{name}")
        nv = run_replay(df, seasonal_naive_price_predict, spec, args.start, args.end,
                        price_col=price_col, carry_soc=carry, annualise_days=annual_days,
                        progress=True, label=f"naive:{name}")
        o = fc.overall
        label = f"{spec.duration_h:.0f}h"
        summary[label] = {
            "oracle": o["eur_per_mw_yr_oracle"],
            "forecast": o["eur_per_mw_yr_forecast"],
            "vcr": o["vcr"],
        }
        print(f"{name:<8}{o['eur_per_mw_yr_oracle']:>16,.0f}{o['eur_per_mw_yr_forecast']:>12,.0f}"
              f"{o['vcr']*100:>7.1f}%{nv.vcr*100:>9.1f}%{o['kendall_tau']:>7.3f}{o['mae']:>8.2f}")

    chart = plot_headline_vcr(summary, out_dir / "headline_vcr.png")
    print(f"\nHeadline chart: {chart.relative_to(REPO_ROOT)}")
    print("\nKPI note: VCR (value captured vs perfect foresight) and Kendall tau are the\n"
          "business metrics — not MAE. Day-ahead arbitrage only, price-taker.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
