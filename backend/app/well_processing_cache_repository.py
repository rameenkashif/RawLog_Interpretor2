"""
well_processing_cache_repository.py
--------------------------------------
Disk-persisted, well_id-keyed cache of the dashboard-upload background
pipeline's results (tie/synthetic/spectral summaries + processing status).

Unlike seismic_processor.py's _spectral_cache/_volume_cache (in-memory
only, cleared on every process restart or SEG-Y refresh), this cache
survives restarts and is what the new agent tools + the upload status
endpoint read from -- see services/dashboard_upload_service.py.

Scalars only: the well-to-seismic tie and synthetic seismogram results
carry large arrays (time series, spectra) that every page already fetches
fresh from the existing live endpoints via React Query. Persisting those
arrays a second time here would just be a stale duplicate; this cache only
needs to answer "is it ready, and how good is it" cheaply.

Same file-per-entity JSON pattern as synthetic_tie_repository.py /
coordinate_calibration_repository.py: ABC -> File impl -> module-global
singleton -> get_*_repository() accessor.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
WELL_PROCESSING_CACHE_DIR = DATA_DIR / "well_processing_cache"


@dataclass
class WellProcessingCacheRecord:
    well_id: str
    status: str  # "processing" | "ready" | "failed"
    run_token: str
    created_at: str
    updated_at: str
    dataset_id: str | None = None
    segy_filename: str | None = None
    error: str | None = None

    tie_available: bool = False
    tie_error: str | None = None
    tie_correlation: float | None = None
    tie_boundary_pinned: bool | None = None
    tie_low_confidence: bool | None = None
    tie_best_freq_hz: float | None = None
    tie_polarity: int | None = None
    tie_bulk_shift_ms: float | None = None
    tie_distance_m: float | None = None
    tie_trace_index: int | None = None
    tie_inline: int | None = None
    tie_crossline: int | None = None

    synthetic_available: bool = False
    synthetic_error: str | None = None
    synthetic_correlation: float | None = None
    synthetic_boundary_pinned: bool | None = None
    synthetic_low_confidence: bool | None = None
    synthetic_datum_check_plausible: bool | None = None
    synthetic_washout_count: int | None = None
    synthetic_polarity: int | None = None
    synthetic_best_shift_ms: float | None = None

    spectral_available: bool = False
    spectral_error: str | None = None
    spectral_inline: int | None = None
    spectral_dominant_freq_hz: float | None = None
    spectral_bandwidth_hz: float | None = None
    spectral_snr_proxy: float | None = None


class WellProcessingCacheRepository(ABC):
    @abstractmethod
    def save(self, record: WellProcessingCacheRecord) -> None: ...

    @abstractmethod
    def get(self, well_id: str) -> WellProcessingCacheRecord | None: ...

    @abstractmethod
    def list_all(self) -> list[WellProcessingCacheRecord]: ...


class FileWellProcessingCacheRepository(WellProcessingCacheRepository):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or WELL_PROCESSING_CACHE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.json"

    def save(self, record: WellProcessingCacheRecord) -> None:
        with open(self._path(record.well_id), "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

    def get(self, well_id: str) -> WellProcessingCacheRecord | None:
        path = self._path(well_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WellProcessingCacheRecord(**data)

    def list_all(self) -> list[WellProcessingCacheRecord]:
        records = []
        for path in sorted(self.base_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                records.append(WellProcessingCacheRecord(**json.load(f)))
        return records


_repository: WellProcessingCacheRepository | None = None


def get_well_processing_cache_repository() -> WellProcessingCacheRepository:
    """FastAPI dependency-injectable accessor for the active repository."""
    global _repository
    if _repository is None:
        _repository = FileWellProcessingCacheRepository()
    return _repository
