"""
repository.py
--------------
Storage layer for processed well data.

The brief asks for an in-memory/local-file cache to start, architected so
it can be swapped for Postgres later without touching routers/services.
`WellRepository` is the abstract interface; `FileWellRepository` is the
concrete implementation used today (Parquet for curve data + JSON sidecar
for metadata, both under backend/data/processed/).

To swap in Postgres later: implement a `PostgresWellRepository` with the
same method signatures and swap the instance created in `get_repository()`
below -- no router or service code needs to change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.las_loader import WellMetadata

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"


class WellRepository(ABC):
    @abstractmethod
    def save_well(self, metadata: WellMetadata, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def get_well(self, well_id: str) -> tuple[WellMetadata, pd.DataFrame] | None: ...

    @abstractmethod
    def list_wells(self) -> list[WellMetadata]: ...

    @abstractmethod
    def delete_well(self, well_id: str) -> bool: ...

    @abstractmethod
    def well_exists(self, well_id: str) -> bool: ...


class FileWellRepository(WellRepository):
    """Local-disk repository: one Parquet file (curve data) + one JSON
    sidecar (metadata) per well, under `backend/data/processed/`.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or PROCESSED_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _curve_path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.parquet"

    def _meta_path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.meta.json"

    def save_well(self, metadata: WellMetadata, df: pd.DataFrame) -> None:
        df.to_parquet(self._curve_path(metadata.well_id), index=False)
        with open(self._meta_path(metadata.well_id), "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    def get_well(self, well_id: str) -> tuple[WellMetadata, pd.DataFrame] | None:
        curve_path = self._curve_path(well_id)
        meta_path = self._meta_path(well_id)
        if not curve_path.exists() or not meta_path.exists():
            return None

        df = pd.read_parquet(curve_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = WellMetadata(**meta_dict)
        return metadata, df

    def list_wells(self) -> list[WellMetadata]:
        metas = []
        for meta_path in sorted(self.base_dir.glob("*.meta.json")):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            metas.append(WellMetadata(**meta_dict))
        return metas

    def delete_well(self, well_id: str) -> bool:
        curve_path = self._curve_path(well_id)
        meta_path = self._meta_path(well_id)
        existed = curve_path.exists() or meta_path.exists()
        curve_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def well_exists(self, well_id: str) -> bool:
        return self._curve_path(well_id).exists() and self._meta_path(well_id).exists()


_repository: WellRepository | None = None


def get_repository() -> WellRepository:
    """FastAPI dependency-injectable accessor for the active repository.

    Swap `FileWellRepository()` for a future `PostgresWellRepository()`
    here -- this is the single point of change needed to migrate storage
    backends.
    """
    global _repository
    if _repository is None:
        _repository = FileWellRepository()
    return _repository
