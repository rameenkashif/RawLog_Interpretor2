"""
coordinate_tie_override_repository.py
-----------------------------------------
Storage for manual well->trace tie-point overrides (see
coordinate_calibration.py's module docstring for why these exist: the
per-axis linear well/seismic coordinate fit is a working default, not a
real CRS reprojection, and no algorithm can recover a true correspondence
from genuinely ambiguous coordinate data alone -- a manual override is the
real fix path for a well the calibration can't resolve with confidence).

One JSON file per well, same file-per-entity pattern as
synthetic_tie_repository.py / repository.py. An override, once saved,
takes priority over the calibrated/algorithmic tie for that well
everywhere a well needs to be located on the seismic survey.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
COORDINATE_OVERRIDE_DIR = DATA_DIR / "coordinate_overrides"


@dataclass
class WellTraceOverride:
    """A user-confirmed well -> trace mapping, bypassing the calibrated
    coordinate transform entirely for this well."""

    well_id: str
    inline: int
    crossline: int
    note: str = ""


class CoordinateTieOverrideRepository(ABC):
    @abstractmethod
    def save_override(self, override: WellTraceOverride) -> None: ...

    @abstractmethod
    def get_override(self, well_id: str) -> WellTraceOverride | None: ...

    @abstractmethod
    def list_overrides(self) -> list[WellTraceOverride]: ...

    @abstractmethod
    def delete_override(self, well_id: str) -> bool: ...


class FileCoordinateTieOverrideRepository(CoordinateTieOverrideRepository):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or COORDINATE_OVERRIDE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.override.json"

    def save_override(self, override: WellTraceOverride) -> None:
        with open(self._path(override.well_id), "w", encoding="utf-8") as f:
            json.dump(asdict(override), f, indent=2)

    def get_override(self, well_id: str) -> WellTraceOverride | None:
        path = self._path(well_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WellTraceOverride(**data)

    def list_overrides(self) -> list[WellTraceOverride]:
        overrides = []
        for path in sorted(self.base_dir.glob("*.override.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            overrides.append(WellTraceOverride(**data))
        return overrides

    def delete_override(self, well_id: str) -> bool:
        path = self._path(well_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed


_repository: CoordinateTieOverrideRepository | None = None


def get_coordinate_tie_override_repository() -> CoordinateTieOverrideRepository:
    """FastAPI dependency-injectable accessor for the active repository."""
    global _repository
    if _repository is None:
        _repository = FileCoordinateTieOverrideRepository()
    return _repository
