"""Tests for the value-capture metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bessopt.backtest import metrics


def test_kendall_tau_monotone_is_one():
    a = np.array([1.0, 2, 3, 4, 5, 6])
    assert metrics.kendall_tau(a, 3 * a + 7) == pytest.approx(1.0)
    assert metrics.kendall_tau(a, -a) == pytest.approx(-1.0)


def test_kendall_tau_handles_nan():
    a = np.array([1.0, 2, np.nan, 4])
    b = np.array([1.0, 2, 3, 4])
    assert metrics.kendall_tau(a, b) == pytest.approx(1.0)


def test_vcr_arithmetic():
    assert metrics.vcr(80.0, 100.0) == pytest.approx(0.8)
    assert np.isnan(metrics.vcr(10.0, 0.0))


def test_pinball_median_is_half_mae():
    y = np.array([10.0, 20, 30])
    yhat = np.array([12.0, 18, 33])
    # Pinball at q=0.5 equals 0.5 * MAE.
    expected = 0.5 * np.mean(np.abs(y - yhat))
    assert metrics.pinball(y, yhat, 0.5) == pytest.approx(expected)


def test_pinball_asymmetry():
    """At q=0.9 under-prediction is penalised ~9x more than over-prediction."""
    y = np.array([100.0])
    under = metrics.pinball(y, np.array([90.0]), 0.9)   # pred below actual
    over = metrics.pinball(y, np.array([110.0]), 0.9)   # pred above actual
    assert under == pytest.approx(9.0)
    assert over == pytest.approx(1.0)


def test_crps_nonnegative_and_zero_at_perfect():
    y = np.array([50.0, 60, 70])
    perfect = [y, y, y]
    assert metrics.crps_from_quantiles(y, (0.1, 0.5, 0.9), perfect) == pytest.approx(0.0)
    noisy = [y - 10, y, y + 10]
    assert metrics.crps_from_quantiles(y, (0.1, 0.5, 0.9), noisy) > 0
