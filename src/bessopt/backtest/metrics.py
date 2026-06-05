"""Forecast + value-capture metrics.

Point error (MAE/RMSE/MAPE) and skill score are vendored. The additions are the
metrics that actually matter for a storage forecaster:

  - **Kendall tau** — rank correlation between forecast and realised prices. The
    report's key finding: tau (not MAE) predicts dispatch value. tau ~0.85-0.95
    captures 97-100% of perfect-foresight revenue; persistence (tau ~0) ~33%.
  - **VCR** — Value Capture Ratio = R_forecast / R_oracle, the primary business
    metric ("my forecast captures X% of theoretical revenue").
  - **Pinball / CRPS** — proper scores for the probabilistic (quantile) forecast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error in percent. Skips zero/NaN denominators."""
    mask = y_true.abs() > 1e-9
    if not mask.any():
        return float("nan")
    return float(((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()).mean() * 100)


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """1 - MAE_model / MAE_baseline. >0 means the model beats the baseline."""
    if mae_baseline <= 0:
        return float("nan")
    return 1.0 - mae_model / mae_baseline


# --- value-capture metrics (the ones that matter for storage) ---

def kendall_tau(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation between forecast and realised prices — the KPI that
    predicts dispatch value (battery dispatch is a ranking problem)."""
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = ~(np.isnan(f) | np.isnan(a))
    if mask.sum() < 2:
        return float("nan")
    return float(kendalltau(f[mask], a[mask]).statistic)


def vcr(rev_forecast: float, rev_oracle: float) -> float:
    """Value Capture Ratio = R_forecast / R_oracle. Reported at portfolio level
    (sum then divide) to stay robust when a single day's oracle revenue ~ 0."""
    if rev_oracle == 0:
        return float("nan")
    return float(rev_forecast / rev_oracle)


def pinball(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level q in (0, 1)."""
    y = np.asarray(y_true, dtype=float)
    yq = np.asarray(y_pred_q, dtype=float)
    diff = y - yq
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.nanmean(loss))


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_levels: Sequence[float],
    quantile_preds: Sequence[np.ndarray],
) -> float:
    """Discrete-quantile approximation of CRPS = 2 · mean over levels of pinball loss.

    With a finite set of quantile levels this is the standard CRPS estimator
    (twice the average pinball loss across the reported quantiles).
    """
    levels = list(quantile_levels)
    losses = [pinball(y_true, qp, q) for q, qp in zip(levels, quantile_preds, strict=True)]
    return float(2.0 * np.mean(losses))
