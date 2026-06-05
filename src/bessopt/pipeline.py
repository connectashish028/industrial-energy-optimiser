"""Daily pipeline — ingest → forecast → optimise → settle → log.

The lightweight productionisation path (no Airflow): a single orchestration
function a GitHub Actions cron calls each day. It is gate-closure paced, not
HFT — correctness around the D-1 12:00 gate matters, latency does not.

Steps:
  1. Load the (refreshed) DE-LU parquet; find the last delivery day with complete
     actuals.
  2. Score the model over a trailing window: VCR / Kendall τ / MAE / €/MW/yr for
     the 1h and 2h asset (forecast-driven vs perfect-foresight oracle).
  3. Produce the next delivery day's forecast + optimal dispatch (the operational
     output — no settlement yet).
  4. Log params + KPIs + chart artifacts to MLflow (local file store; degrades
     gracefully if unavailable).
  5. Write results.json + refresh the dashboard charts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from .backtest.replay import run_replay
from .config import REPO_ROOT, load_assets, load_backtest_config, load_data_config
from .data.loader import issue_time_for, load_de_lu_15min, target_index_for
from .forecast.predict import DEFAULT_XGB_PRICE_DIR, xgboost_price_predict_full
from .optimiser.lp import solve_dispatch
from .reporting.charts import plot_headline_vcr, plot_soc_dispatch

OUT_DIR = REPO_ROOT / "outputs"
RESULTS_JSON = OUT_DIR / "results.json"


def _last_complete_actual_day(df: pd.DataFrame, price_col: str):
    """Most recent delivery day whose 96 actual prices are all present."""
    d = df.index.max().tz_convert("Europe/Berlin").date()
    for _ in range(15):
        if not df[price_col].reindex(target_index_for(d)).isna().any():
            return d
        d -= timedelta(days=1)
    raise RuntimeError("No complete actual delivery day found near the data edge.")


def _next_forecastable_day(df: pd.DataFrame, after, price_col: str):
    """Next delivery day after `after` whose forecast can be built (issue ≤ data end)."""
    last_ts = df.index.max()
    d = after + timedelta(days=1)
    for _ in range(5):
        if issue_time_for(d) <= last_ts:
            fc = xgboost_price_predict_full(df, issue_time_for(d))
            if not fc["p50"].isna().any():
                return d, fc
        d += timedelta(days=1)
    return None, None


def _model_version() -> str:
    meta = DEFAULT_XGB_PRICE_DIR / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("model", "xgboost_price")
    return "unknown"


def run_daily_pipeline(
    *,
    window_days: int = 60,
    parquet: str | None = None,
    use_mlflow: bool = True,
    now_utc: datetime | None = None,
) -> dict:
    bt = load_backtest_config()
    data_cfg = load_data_config()
    price_col = data_cfg["price_col"]
    annual_days = bt["annualise_days"]
    assets = load_assets()

    df = load_de_lu_15min(parquet or data_cfg["parquet_path"])
    now = now_utc or datetime.now(UTC)

    last_day = _last_complete_actual_day(df, price_col)
    start = last_day - timedelta(days=window_days)

    # 2. Score the model over the trailing window.
    asset_kpis: dict[str, dict] = {}
    vcr_summary: dict[str, dict] = {}
    for name, spec in assets.items():
        rep = run_replay(df, xgboost_price_predict_full, spec, start, last_day,
                         price_col=price_col, carry_soc=bt["replay"]["carry_soc"],
                         annualise_days=annual_days, progress=False, label=name)
        o = rep.overall
        asset_kpis[name] = {
            "vcr": o["vcr"], "kendall_tau": o["kendall_tau"], "mae": o["mae"],
            "eur_per_mw_yr_oracle": o["eur_per_mw_yr_oracle"],
            "eur_per_mw_yr_forecast": o["eur_per_mw_yr_forecast"], "n_days": o["n_days"],
        }
        vcr_summary[f"{spec.duration_h:.0f}h"] = {
            "oracle": o["eur_per_mw_yr_oracle"], "forecast": o["eur_per_mw_yr_forecast"],
            "vcr": o["vcr"],
        }

    # 3. Next-day operational forecast + dispatch (2h asset).
    spec2 = assets["asset_2h"]
    nd_date, fc = _next_forecastable_day(df, last_day, price_col)
    next_day = None
    if nd_date is not None:
        plan = solve_dispatch(fc["p50"].to_numpy(), spec2,
                              soc_initial_mwh=spec2.soc_min_mwh, index=fc.index)
        plot_soc_dispatch(plan, spec2, title=f"Next-day dispatch — asset_2h — {nd_date}",
                          out_path=OUT_DIR / "next_day_dispatch_2h.png")
        next_day = {
            "date": nd_date.isoformat(),
            "asset": "asset_2h",
            "expected_arbitrage_eur": float(plan.revenue_eur),  # vs forecast prices
            "n_charge_slots": int((plan.charge_mw > 1e-6).sum()),
            "n_discharge_slots": int((plan.discharge_mw > 1e-6).sum()),
            "cycles": float(plan.cycles(spec2)),
        }

    # 5. Charts + results.json.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_headline_vcr(vcr_summary, OUT_DIR / "headline_vcr.png")
    results = {
        "run_time_utc": now.isoformat(),
        "data_last_ts": df.index.max().isoformat(),
        "data_lag_hours": round((now - df[price_col].dropna().index.max()).total_seconds() / 3600, 1),
        "model_version": _model_version(),
        "window": {"start": start.isoformat(), "end": last_day.isoformat(), "days": window_days},
        "assets": asset_kpis,
        "next_day": next_day,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    # 4. MLflow.
    if use_mlflow:
        _log_to_mlflow(results, now)

    return results


def _log_to_mlflow(results: dict, now: datetime) -> None:
    try:
        import mlflow

        # SQLite tracking backend (file store is deprecated in recent MLflow);
        # artifacts go to a local mlartifacts/ dir. Both are gitignored.
        mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
        exp_name = "bessopt-daily"
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=(REPO_ROOT / "mlartifacts").as_uri())
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=f"daily-{now:%Y%m%d-%H%M}"):
            mlflow.log_params({
                "model_version": results["model_version"],
                "window_days": results["window"]["days"],
                "window_end": results["window"]["end"],
            })
            metrics = {"data_lag_hours": results["data_lag_hours"]}
            for name, k in results["assets"].items():
                for key in ("vcr", "kendall_tau", "mae", "eur_per_mw_yr_forecast",
                            "eur_per_mw_yr_oracle"):
                    metrics[f"{name}.{key}"] = float(k[key])
            mlflow.log_metrics(metrics)
            for png in ("headline_vcr.png", "next_day_dispatch_2h.png"):
                p = OUT_DIR / png
                if p.exists():
                    mlflow.log_artifact(str(p))
            mlflow.log_artifact(str(RESULTS_JSON))
    except Exception as e:  # noqa: BLE001 — logging must never break the pipeline
        print(f"[pipeline] MLflow logging skipped: {e!r}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="bessopt daily pipeline (M7).")
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    res = run_daily_pipeline(window_days=args.window_days, use_mlflow=not args.no_mlflow)
    print(f"\nDaily pipeline complete. Window {res['window']['start']}..{res['window']['end']}"
          f"  (data lag {res['data_lag_hours']}h)")
    for name, k in res["assets"].items():
        print(f"  {name}: VCR {k['vcr']*100:.1f}%  τ {k['kendall_tau']:.3f}  "
              f"forecast €{k['eur_per_mw_yr_forecast']:,.0f}/MW/yr")
    if res["next_day"]:
        nd = res["next_day"]
        print(f"  next-day dispatch ({nd['date']}, 2h): {nd['n_charge_slots']} charge / "
              f"{nd['n_discharge_slots']} discharge slots, {nd['cycles']:.2f} cycles")
    print(f"  → {RESULTS_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_daily_pipeline"]
