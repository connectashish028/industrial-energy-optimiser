"""Cardinal invariants for the value-capture replay engine.

The most important: perfect foresight is an upper bound, so the oracle revenue
must dominate the forecast revenue, VCR ∈ [0, 1], and a perfect predictor (one
that returns the actual prices) must yield VCR == 1.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bessopt.backtest.baselines import seasonal_naive_price_predict
from bessopt.backtest.replay import run_replay
from bessopt.data.loader import load_de_lu_15min, target_index_for
from bessopt.optimiser.spec import BatterySpec

PARQUET = Path("data/de_lu_15min.parquet")
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="parquet not present.")

SPEC = BatterySpec(power_mw=10.0, energy_mwh=20.0)
START, END = date(2026, 2, 1), date(2026, 2, 7)
PRICE = "price__germany_luxembourg"


@pytest.fixture(scope="module")
def df():
    return load_de_lu_15min(PARQUET)


def perfect_predictor(frame: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """A predictor that 'knows' the actual delivery-day prices (for the VCR=1 test)."""
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    idx = target_index_for(delivery_local.date())
    actual = frame[PRICE].reindex(idx).to_numpy()
    return pd.DataFrame({"p10": actual, "p50": actual, "p90": actual}, index=idx)


def test_oracle_dominates_forecast_every_day(df):
    res = run_replay(df, seasonal_naive_price_predict, SPEC, START, END,
                     price_col=PRICE, progress=False)
    # Same-state construction ⇒ oracle ≥ forecast each day (small numerical slack).
    assert (res.per_day["rev_oracle"] >= res.per_day["rev_forecast"] - 1e-6).all()
    assert res.overall["rev_oracle_total"] >= res.overall["rev_forecast_total"] - 1e-6


def test_vcr_in_unit_interval(df):
    res = run_replay(df, seasonal_naive_price_predict, SPEC, START, END,
                     price_col=PRICE, progress=False)
    assert -1e-9 <= res.vcr <= 1.0 + 1e-9


def test_perfect_predictor_gives_vcr_one(df):
    res = run_replay(df, perfect_predictor, SPEC, START, END, price_col=PRICE, progress=False)
    assert res.vcr == pytest.approx(1.0, abs=1e-6)
    # And Kendall tau of a perfect forecast is 1.
    assert res.overall["kendall_tau"] == pytest.approx(1.0, abs=1e-9)


def test_real_forecast_beats_naive(df):
    """The XGBoost forecast should capture more value than seasonal-naive."""
    from bessopt.forecast.predict import DEFAULT_XGB_PRICE_DIR, xgboost_price_predict_full

    if not DEFAULT_XGB_PRICE_DIR.exists():
        pytest.skip("price model checkpoint not present")
    fc = run_replay(df, xgboost_price_predict_full, SPEC, START, END,
                    price_col=PRICE, progress=False)
    naive = run_replay(df, seasonal_naive_price_predict, SPEC, START, END,
                       price_col=PRICE, progress=False)
    assert fc.vcr >= naive.vcr
