"""Smoke test for the daily pipeline (M7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bessopt.pipeline import run_daily_pipeline

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="parquet not present.")


def test_pipeline_runs_and_produces_results():
    # MLflow off for speed / no DB side effects; fixed clock for determinism.
    res = run_daily_pipeline(window_days=8, use_mlflow=False,
                             now_utc=datetime(2026, 6, 4, 9, 0, tzinfo=UTC))

    assert set(res) >= {"run_time_utc", "data_last_ts", "window", "assets", "next_day"}
    assert set(res["assets"]) == {"asset_1h", "asset_2h"}
    for k in res["assets"].values():
        assert 0.0 <= k["vcr"] <= 1.0
        assert -1.0 <= k["kendall_tau"] <= 1.0
        assert k["eur_per_mw_yr_oracle"] >= k["eur_per_mw_yr_forecast"] - 1.0
        assert k["n_days"] > 0
    assert res["window"]["days"] == 8
    # results.json written.
    from bessopt.pipeline import RESULTS_JSON
    assert RESULTS_JSON.exists()
