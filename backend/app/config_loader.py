"""
config_loader.py
-----------------
Loads `config/petrophysics_config.yaml` and resolves per-well parameter
overrides against the field-wide `defaults` block.

This is intentionally the *only* place that reads the YAML file, so the
rest of the codebase (petrophysics.py, routers, etc.) can just call
`get_well_config(well_id)` and receive a fully-merged, ready-to-use dict.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config" / "petrophysics_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` on top of `base`, without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@functools.lru_cache(maxsize=1)
def _load_raw_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Petrophysics config not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def reload_config() -> None:
    """Clear the cached config so the next access re-reads the YAML file.

    Useful for tests or after an SME edits the config live.
    """
    _load_raw_config.cache_clear()


def get_well_config(well_id: str | None = None) -> dict[str, Any]:
    """Return the fully-merged config dict for a given well.

    Falls back entirely to `defaults` when `well_id` is None or has no
    override block defined in the YAML file.
    """
    raw = _load_raw_config()
    defaults = raw.get("defaults", {})
    wells = raw.get("wells", {}) or {}

    if well_id is None:
        return copy.deepcopy(defaults)

    override = wells.get(well_id, {}) or {}
    return _deep_merge(defaults, override)
