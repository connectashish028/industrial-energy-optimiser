"""Phase 6 / M6 deliverable — risk (CVaR) + degradation overlay.

Part A — CVaR efficient frontier: for a sample of days, build timing-uncertain
price scenarios from the probabilistic forecast and sweep the risk-aversion β.
Shows the desk's tradeoff — give up a little expected revenue to lift the
worst-case (the efficient frontier).

Part B — degradation sensitivity: sweep the linear throughput cost ν_deg and
show how cycling and net revenue respond (the industry-standard degradation
lever, already in the LP objective).

    python scripts/run_risk.py
    python scripts/run_risk.py --days 12 --asset asset_2h
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from bessopt.backtest.oracle import run_oracle
from bessopt.config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from bessopt.data.loader import issue_time_for, load_de_lu_15min
from bessopt.forecast.predict import xgboost_price_predict_full
from bessopt.reporting.charts import plot_efficient_frontier
from bessopt.risk.cvar import scenarios_sampled_from_quantiles, solve_cvar_dispatch

ALPHA = 0.20
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
DEG_COSTS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def _sample_days(start: date, end: date, n: int) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=int(round(i * span / (n - 1)))) for i in range(n)]


def cvar_frontier(df, spec, days, price_col):
    """Per-day-average expected revenue and CVaR(loss) at each β."""
    quantile_frames = []
    for d in days:
        try:
            q = xgboost_price_predict_full(df, issue_time_for(d))
        except Exception:  # noqa: BLE001
            continue
        if q["p50"].isna().any():
            continue
        quantile_frames.append(q)

    frontier = []
    for b in BETAS:
        exp, cvar = [], []
        for q in quantile_frames:
            scen = scenarios_sampled_from_quantiles(q, n_scenarios=40, seed=1)
            r = solve_cvar_dispatch(scen, spec, alpha=ALPHA, beta=b,
                                    soc_initial_mwh=spec.soc_min_mwh)
            exp.append(r.expected_revenue_eur)
            cvar.append(r.cvar_loss_eur)
        frontier.append({"beta": b, "expected": float(np.mean(exp)),
                         "cvar_loss": float(np.mean(cvar)), "n_days": len(exp)})
    return frontier


def degradation_sweep(df, spec, start, end, price_col, annual_days):
    rows = []
    for nu in DEG_COSTS:
        res = run_oracle(df, replace(spec, deg_cost_eur_per_mwh=nu), start, end,
                         price_col=price_col, annualise_days=annual_days)
        rows.append({"deg_cost": nu, "eur_per_mw_yr": res.eur_per_mw_year,
                     "avg_cycles": float(res.per_day["cycles"].mean())})
    return rows


def main() -> int:
    bt = load_backtest_config()
    data_cfg = load_data_config()

    ap = argparse.ArgumentParser(description="Risk (CVaR) + degradation overlay (M6).")
    ap.add_argument("--start", type=_parse_date, default=_parse_date(bt["test_start"]))
    ap.add_argument("--end", type=_parse_date, default=_parse_date(bt["test_end"]))
    ap.add_argument("--asset", default="asset_2h")
    ap.add_argument("--days", type=int, default=12, help="days sampled for the CVaR frontier")
    ap.add_argument("--parquet", default=data_cfg["parquet_path"])
    args = ap.parse_args()

    df = load_de_lu_15min(args.parquet)
    spec = load_assets()[args.asset]
    price_col = data_cfg["price_col"]
    out_dir = REPO_ROOT / "outputs"

    # --- Part A: CVaR efficient frontier ---
    print(f"\nCVaR efficient frontier ({args.asset}, α={ALPHA}, "
          f"{args.days} sampled days {args.start}..{args.end})\n")
    days = _sample_days(args.start, args.end, args.days)
    frontier = cvar_frontier(df, spec, days, price_col)
    print(f"{'beta':>6}{'E[rev] €/day':>16}{'CVaR(loss) €/day':>20}")
    print("-" * 42)
    for f in frontier:
        print(f"{f['beta']:>6.2f}{f['expected']:>16,.1f}{f['cvar_loss']:>20,.1f}")
    base, safe = frontier[0], frontier[-1]
    give_up = (base["expected"] - safe["expected"]) / base["expected"] * 100
    print(f"\n  β: 0 → {safe['beta']:g} sacrifices {give_up:.1f}% expected revenue to cut "
          f"downside CVaR from {base['cvar_loss']:,.0f} to {safe['cvar_loss']:,.0f} €/day.")
    chart = plot_efficient_frontier(frontier, out_dir / "cvar_frontier.png")
    print(f"  Frontier chart: {chart.relative_to(REPO_ROOT)}")

    # --- Part B: degradation sensitivity ---
    print(f"\nDegradation sensitivity ({args.asset}, perfect-foresight oracle "
          f"{args.start}..{args.end}):\n")
    print(f"{'ν_deg €/MWh':>12}{'€/MW/yr':>14}{'avg cycles/day':>16}")
    print("-" * 42)
    for r in degradation_sweep(df, spec, args.start, args.end, price_col, bt["annualise_days"]):
        print(f"{r['deg_cost']:>12.1f}{r['eur_per_mw_yr']:>14,.0f}{r['avg_cycles']:>16.2f}")
    print("\nHigher throughput cost ⇒ the optimiser cycles less and only on wider spreads "
          "(degradation internalised). Linear cost is the documented industry standard.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
