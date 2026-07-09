"""
synthetic_tie_repository.py
------------------------------
Storage for manual well-tie stretch/squeeze control points, one JSON file
per well, so adjustments persist across sessions instead of being
recomputed from scratch every time the synthetic seismogram module is
opened. Same file-per-entity JSON pattern as repository.py /
seismic_repository.py.

Keyed by well_id only (not well_id + dataset_id): the synthetic seismogram
module ties against seismic_processor.SegyVolume's single active SEG-Y
volume (see seismic_processor.py), not the separate multi-dataset upload
pipeline, so there is only ever one "current" survey to tie against at a
time. segy_filename is still recorded on the saved tie for informational
QC (so a stale tie made against a since-replaced SEG-Y file is visible),
but is not part of the storage key.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SYNTHETIC_PROCESSED_DIR = DATA_DIR / "synthetic_processed"


@dataclass
class TiePoint:
    """One manual stretch/squeeze control point."""

    md_m: float  # measured depth, meters, on the well's own MD axis
    time_shift_ms: float  # manual time shift applied at this control point


@dataclass
class TiePointSet:
    """A well's full set of manual tie adjustments, plus which wavelet
    settings they were made with (stretch/squeeze is wavelet-dependent, so
    persisting this alongside the points avoids silently reapplying old
    adjustments to a differently-generated synthetic)."""

    well_id: str
    points: list[TiePoint] = field(default_factory=list)
    wavelet_method: str = "ricker"
    wavelet_freq_hz: float = 25.0
    segy_filename: str | None = None


class SyntheticTieRepository(ABC):
    @abstractmethod
    def save_tie_points(self, tie: TiePointSet) -> None: ...

    @abstractmethod
    def get_tie_points(self, well_id: str) -> TiePointSet | None: ...

    @abstractmethod
    def delete_tie_points(self, well_id: str) -> bool: ...


class FileSyntheticTieRepository(SyntheticTieRepository):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or SYNTHETIC_PROCESSED_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.tie.json"

    def save_tie_points(self, tie: TiePointSet) -> None:
        with open(self._path(tie.well_id), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "well_id": tie.well_id,
                    "points": [asdict(p) for p in tie.points],
                    "wavelet_method": tie.wavelet_method,
                    "wavelet_freq_hz": tie.wavelet_freq_hz,
                    "segy_filename": tie.segy_filename,
                },
                f,
                indent=2,
            )

    def get_tie_points(self, well_id: str) -> TiePointSet | None:
        path = self._path(well_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TiePointSet(
            well_id=data["well_id"],
            points=[TiePoint(**p) for p in data.get("points", [])],
            wavelet_method=data.get("wavelet_method", "ricker"),
            wavelet_freq_hz=data.get("wavelet_freq_hz", 25.0),
            segy_filename=data.get("segy_filename"),
        )

    def delete_tie_points(self, well_id: str) -> bool:
        path = self._path(well_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed


_repository: SyntheticTieRepository | None = None


def get_synthetic_tie_repository() -> SyntheticTieRepository:
    """FastAPI dependency-injectable accessor for the active repository."""
    global _repository
    if _repository is None:
        _repository = FileSyntheticTieRepository()
    return _repository
