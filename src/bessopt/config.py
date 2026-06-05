"""Config loading — single place that reads configs/*.yaml and builds objects.

Paths resolve relative to the repository root (three levels up from this file),
so scripts work regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .optimiser.spec import BatterySpec

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_assets() -> dict[str, BatterySpec]:
    """Return {asset_name: BatterySpec} from configs/assets.yaml."""
    raw = _load_yaml("assets.yaml")
    return {name: BatterySpec(**params) for name, params in raw.items()}


def load_data_config() -> dict:
    cfg = _load_yaml("data.yaml")
    # Resolve the parquet path against the repo root.
    cfg["parquet_path"] = str((REPO_ROOT / cfg["parquet_path"]).resolve())
    return cfg


def load_backtest_config() -> dict:
    return _load_yaml("backtest.yaml")


__all__ = [
    "CONFIG_DIR",
    "REPO_ROOT",
    "load_assets",
    "load_backtest_config",
    "load_data_config",
]
