"""
config_loader.py
-----------------
Loads `config/petrophysics_config.yaml` and `config/seismic_config.yaml`,
resolving per-well / per-dataset parameter overrides against each file's
field-wide `defaults` block.

This is intentionally the *only* place that reads these YAML files, so the
rest of the codebase (petrophysics.py, seismic_attributes.py, routers,
etc.) can just call `get_well_config(well_id)` / `get_seismic_config(dataset_id)`
and receive a fully-merged, ready-to-use dict.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any

import yaml

_PETROPHYSICS_CONFIG_PATH = (
    Path(__file__).parent / "config" / "petrophysics_config.yaml"
)
_SEISMIC_CONFIG_PATH = Path(__file__).parent / "config" / "seismic_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` on top of `base`, without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache(maxsize=1)
def _load_raw_petrophysics_config() -> dict[str, Any]:
    return _load_yaml(_PETROPHYSICS_CONFIG_PATH)


@functools.lru_cache(maxsize=1)
def _load_raw_seismic_config() -> dict[str, Any]:
    return _load_yaml(_SEISMIC_CONFIG_PATH)


def reload_config() -> None:
    """Clear cached config so the next access re-reads both YAML files.

    Useful for tests or after an SME edits either config file live.
    """
    _load_raw_petrophysics_config.cache_clear()
    _load_raw_seismic_config.cache_clear()


def get_well_config(well_id: str | None = None) -> dict[str, Any]:
    """Return the fully-merged petrophysics config dict for a given well.

    Falls back entirely to `defaults` when `well_id` is None or has no
    override block defined in the YAML file.
    """
    raw = _load_raw_petrophysics_config()
    defaults = raw.get("defaults", {})
    wells = raw.get("wells", {}) or {}

    if well_id is None:
        return copy.deepcopy(defaults)

    override = wells.get(well_id, {}) or {}
    return _deep_merge(defaults, override)


def get_seismic_config(dataset_id: str | None = None) -> dict[str, Any]:
    """Return the fully-merged seismic attributes config dict for a given
    dataset. Falls back entirely to `defaults` when `dataset_id` is None or
    has no override block defined in seismic_config.yaml.
    """
    raw = _load_raw_seismic_config()
    defaults = raw.get("defaults", {})
    datasets = raw.get("datasets", {}) or {}

    if dataset_id is None:
        return copy.deepcopy(defaults)

    override = datasets.get(dataset_id, {}) or {}
    return _deep_merge(defaults, override)
