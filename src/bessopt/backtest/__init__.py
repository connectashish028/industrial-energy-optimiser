"""Backtest — oracle, value-capture replay engine, and rolling-horizon MPC."""

from .baselines import seasonal_naive_price_predict
from .mpc import MpcResult, actual_horizon, forecast_horizon, run_mpc
from .oracle import OracleResult, run_oracle
from .replay import PriceForecastFn, ReplayResult, run_replay

__all__ = [
    "MpcResult",
    "OracleResult",
    "PriceForecastFn",
    "ReplayResult",
    "actual_horizon",
    "forecast_horizon",
    "run_mpc",
    "run_oracle",
    "run_replay",
    "seasonal_naive_price_predict",
]
