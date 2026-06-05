"""Probabilistic day-ahead price forecaster (vendored XGBoost quantile model).

Production model from the sibling `loadforecast` project: one XGBoost regressor
per quantile (P10/P50/P90), targeting the raw EPEX DE-LU clearing price. It
already captures ~97% of perfect-foresight battery P&L on a 10 MW / 20 MWh
asset. We reuse it as-is; LightGBM / LEAR are optional later swaps.

Every prediction routes through `build_target_day_features`, which masks
future-leaking values via the point-in-time `availability` rules — so the
forecast only ever sees data that existed at the D-1 12:00 issue time.

The returned frame has one row per target quarter-hour (92/96/100 on DST days)
and columns p10/p50/p90 in EUR/MWh, sorted per row so p10 <= p50 <= p90.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import build_target_day_features

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XGB_PRICE_DIR = _REPO_ROOT / "model_checkpoints" / "xgboost_price_v1"
XGB_QUANTILES = (0.10, 0.50, 0.90)
PRICE_VRE_FC_COL = "fc_gen__photovoltaics_and_wind"

_XGB_CACHE: dict[str, dict] = {}


def _load_xgb(model_dir: Path) -> dict:
    """Load and cache the three quantile regressors + meta for a checkpoint."""
    import xgboost as xgb

    key = str(model_dir.resolve())
    if key in _XGB_CACHE:
        return _XGB_CACHE[key]
    models: dict = {}
    for q in XGB_QUANTILES:
        reg = xgb.XGBRegressor()
        reg.load_model(model_dir / f"xgb_q{int(q * 100):02d}.json")
        models[q] = reg
    meta = json.loads((model_dir / "meta.json").read_text())
    bundle = {"models": models, "meta": meta}
    _XGB_CACHE[key] = bundle
    return bundle


def _add_engineered_vre_features(features: pd.DataFrame, df: pd.DataFrame,
                                 issue_time: pd.Timestamp) -> pd.DataFrame:
    """The 3 engineered VRE features present at the price model's training time."""
    vre_fc = features["tso_vre_fc"]
    features["tso_vre_fc_present"] = (~vre_fc.isna()).astype(np.float32)
    features["tso_vre_fc"] = vre_fc.fillna(0.0)
    load_fc = features["tso_load_fc"]
    safe_load = load_fc.where(load_fc > 0, 1.0)
    features["vre_to_load_ratio"] = (features["tso_vre_fc"] / safe_load).astype(np.float32)
    ref = df[PRICE_VRE_FC_COL].loc[issue_time - pd.Timedelta(days=90):issue_time].dropna()
    q90 = float(ref.quantile(0.90)) if len(ref) > 100 else 1.0
    features["vre_percentile"] = (features["tso_vre_fc"] / max(q90, 1.0)).astype(np.float32)
    return features


def xgboost_price_predict_full(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_XGB_PRICE_DIR,
) -> pd.DataFrame:
    """Probabilistic day-ahead price forecast → (H, 3) DataFrame [p10, p50, p90] in EUR/MWh."""
    bundle = _load_xgb(Path(model_dir))
    models, meta = bundle["models"], bundle["meta"]

    features = build_target_day_features(df, issue_time)
    features = _add_engineered_vre_features(features, df, issue_time)

    # Reindex to the model's exact training feature order (robust to column drift).
    feature_cols = meta.get("feature_cols")
    if feature_cols is not None:
        features = features.reindex(columns=feature_cols)

    X = features.to_numpy(dtype=np.float32)
    preds = np.stack([models[q].predict(X) for q in XGB_QUANTILES], axis=1)  # (H, 3)
    preds.sort(axis=1)  # enforce p10 <= p50 <= p90 (independent quantile models can cross)

    out = pd.DataFrame(
        {"p10": preds[:, 0], "p50": preds[:, 1], "p90": preds[:, 2]},
        index=features.index,
    )
    out.index.name = "target_ts"
    return out


def price_p50(df: pd.DataFrame, issue_time: pd.Timestamp, **kw) -> pd.Series:
    """Point forecast: the median (p50) price, as a Series."""
    return xgboost_price_predict_full(df, issue_time, **kw)["p50"].rename("price_p50")


__all__ = [
    "DEFAULT_XGB_PRICE_DIR",
    "XGB_QUANTILES",
    "price_p50",
    "xgboost_price_predict_full",
]
