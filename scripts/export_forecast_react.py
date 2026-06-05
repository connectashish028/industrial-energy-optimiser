"""Export a February forecast-vs-actual band for the React Forecasting tab.

Runs the XGBoost quantile price model over every February day (point-in-time,
issue = D-1 12:00) and writes the P10/P50/P90 band + realised actual as an
hourly series, plus a February MAE. Keeps the existing VCR cards.

    uv run python scripts/export_forecast_react.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np

from bessopt.config import REPO_ROOT, load_data_config
from bessopt.data.loader import issue_time_for, load_de_lu_15min, target_index_for
from bessopt.forecast.predict import xgboost_price_predict_full

START = date(2026, 2, 1)
END = date(2026, 2, 28)


def _daterange(s: date, e: date):
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def main() -> int:
    cfg = load_data_config()
    df = load_de_lu_15min(cfg["parquet_path"])
    price_col = cfg["price_col"]

    rows = []  # full 15-min resolution (for MAE); downsampled to hourly for the chart
    n_days = 0
    for d in _daterange(START, END):
        idx = target_index_for(d)
        actual = df[price_col].reindex(idx).to_numpy(dtype=float)
        if np.isnan(actual).any():
            continue
        fc = xgboost_price_predict_full(df, issue_time_for(d)).reindex(idx)
        p10 = fc["p10"].to_numpy(dtype=float)
        p50 = fc["p50"].to_numpy(dtype=float)
        p90 = fc["p90"].to_numpy(dtype=float)
        if np.isnan(p50).any():
            continue
        n_days += 1
        for i, ts in enumerate(idx):
            rows.append((ts, actual[i], p10[i], p50[i], p90[i]))

    if not rows:
        raise RuntimeError("No February forecast days produced — check data/model coverage.")

    a = np.array([r[1] for r in rows])
    p50 = np.array([r[3] for r in rows])
    mae = float(np.mean(np.abs(a - p50)))

    # hourly sample (every 4th 15-min slot; 96 slots/day ⇒ clean hour-on-hour)
    series = [
        {"t": ts.strftime("%Y-%m-%dT%H:%M"), "actual": round(av, 1),
         "p10": round(lo, 1), "p50": round(md, 1), "p90": round(hi, 1)}
        for (ts, av, lo, md, hi) in rows[::4]
    ]

    fpath = REPO_ROOT / "react-demo" / "src" / "data" / "forecast.json"
    existing = json.loads(fpath.read_text(encoding="utf-8"))
    out = {
        "assets": existing["assets"],
        "mae": round(mae, 1),
        "period": {"start": START.isoformat(), "end": END.isoformat(),
                   "label": f"February {START.year} ({n_days} days, hourly)"},
        "series": series,
    }
    fpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {fpath}")
    print(f"  {n_days} days · {len(series)} hourly points · MAE €{round(mae, 1)}/MWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
