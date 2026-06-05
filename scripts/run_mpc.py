"""Phase 4 / M4 deliverable — rolling-horizon MPC vs the static daily solve.

The static solve optimises one delivery day at a time (24h horizon = 24h step).
MPC optimises a longer horizon (e.g. 36h) but executes only the first 24h, so it
can see past midnight and hold charge overnight for tomorrow's morning peak.

Two comparisons on the same window/spec:
  - perfect foresight (actual prices): isolates the pure HORIZON effect — the
    structural ceiling on MPC uplift, free of forecast error.
  - real forecast (XGBoost for the day + seasonal-naive look-ahead tail).

    python scripts/run_mpc.py
    python scripts/run_mpc.py --horizon 48 --start 2026-01-01 --end 2026-02-28
"""

from __future__ import annotations

import argparse
from datetime import date
from functools import partial

import pandas as pd

from bessopt.backtest.mpc import actual_horizon, forecast_horizon, run_mpc
from bessopt.config import load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.forecast.predict import xgboost_price_predict_full


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Rolling-horizon MPC vs static (M4).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date(bt["test_start"]))
    ap.add_argument("--end", type=_parse_date, default=_parse_date(bt["test_end"]))
    ap.add_argument("--horizon", type=int, default=bt["mpc"]["horizon_h"])
    ap.add_argument("--step", type=int, default=bt["mpc"]["step_h"])
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    assets = load_assets()
    price_col = data_cfg["price_col"]
    annual_days = bt["annualise_days"]

    fc_fn = partial(forecast_horizon, predictor=xgboost_price_predict_full, price_col=price_col)
    act_fn = partial(actual_horizon, price_col=price_col)

    print(f"\nMPC vs static: {args.start} -> {args.end}  "
          f"(static=24h horizon, MPC={args.horizon}h horizon, step={args.step}h)\n")
    print(f"{'asset':<10}{'signal':<10}{'static EUR/MW/yr':>18}{'MPC EUR/MW/yr':>16}{'uplift':>9}")
    print("-" * 63)

    def _mpc(spec, horizon_fn, horizon_h):
        return run_mpc(df, horizon_fn, spec, args.start, args.end, horizon_h=horizon_h,
                       step_h=args.step, price_col=price_col, annualise_days=annual_days)

    for name, spec in assets.items():
        for sig, hfn in (("perfect", act_fn), ("forecast", fc_fn)):
            static = _mpc(spec, hfn, 24)
            mpc = _mpc(spec, hfn, args.horizon)
            s = static.overall["eur_per_mw_yr"]
            m = mpc.overall["eur_per_mw_yr"]
            uplift = (m / s - 1.0) * 100 if s else float("nan")
            print(f"{name:<10}{sig:<10}{s:>18,.0f}{m:>16,.0f}{uplift:>8.1f}%")

    print("\nNote: 'perfect' isolates the horizon effect (no forecast error); 'forecast' "
          "uses the\nXGBoost day + seasonal-naive look-ahead tail. Day-ahead arbitrage only.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
