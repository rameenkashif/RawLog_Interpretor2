"""
coordinate_calibration_repository.py
-----------------------------------------
Persists the fitted per-axis well<->seismic coordinate calibration
(coordinate_calibration.py) so it has a STABLE reference set of
calibration wells across requests. Refitting from whatever wells happen
to exist right now, every time a report is requested, would silently
absorb a newly added/bad well into defining its own "valid" range --
making the extrapolation flag structurally unable to ever catch it,
exactly the false confidence fix #5 exists to prevent (the real-world
case: a calibration fit from 6-7 wells, then an 8th well added later that
should be checked AGAINST that existing fit, not folded into recomputing
a new one that trivially includes it).

A calibration is only (re)computed when explicitly requested via
coordinate_calibration_service.fit_and_store_calibration() -- normal
report/resolve calls read whatever is currently stored (auto-bootstrapping
once, the first time, from whatever wells exist then).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from app.coordinate_calibration import AxisCalibration

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CALIBRATION_PATH = DATA_DIR / "coordinate_overrides" / "calibration.json"


class CoordinateCalibrationRepository(ABC):
    @abstractmethod
    def save(
        self, calibration: AxisCalibration, well_ids: list[str], bin_spacing_m: float, segy_filename: str
    ) -> None: ...

    @abstractmethod
    def load(self) -> tuple[AxisCalibration, list[str], float, str] | None: ...


class FileCoordinateCalibrationRepository(CoordinateCalibrationRepository):
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_CALIBRATION_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self, calibration: AxisCalibration, well_ids: list[str], bin_spacing_m: float, segy_filename: str
    ) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "calibration": asdict(calibration),
                    "well_ids": well_ids,
                    "bin_spacing_m": bin_spacing_m,
                    "segy_filename": segy_filename,
                },
                f,
                indent=2,
            )

    def load(self) -> tuple[AxisCalibration, list[str], float, str] | None:
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cal_data = data["calibration"]
        cal = AxisCalibration(
            a=cal_data["a"],
            b=cal_data["b"],
            c=cal_data["c"],
            d=cal_data["d"],
            well_x_range=tuple(cal_data["well_x_range"]),
            well_y_range=tuple(cal_data["well_y_range"]),
            seismic_x_range=tuple(cal_data["seismic_x_range"]),
            seismic_y_range=tuple(cal_data["seismic_y_range"]),
            n_wells_used=cal_data["n_wells_used"],
        )
        return cal, data["well_ids"], data["bin_spacing_m"], data["segy_filename"]


_repository: CoordinateCalibrationRepository | None = None


def get_coordinate_calibration_repository() -> CoordinateCalibrationRepository:
    """FastAPI dependency-injectable accessor for the active repository."""
    global _repository
    if _repository is None:
        _repository = FileCoordinateCalibrationRepository()
    return _repository
