"""
seismic_repository.py
-----------------------
Storage layer for processed seismic datasets, mirroring repository.py's
pattern for wells: an abstract interface + a local-disk implementation
(NumPy .npz for the trace matrix, Parquet for per-trace attributes, JSON
for metadata), so it can be swapped for a database-backed implementation
later without touching routers/services.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.segy_loader import SegyMetadata

DATA_DIR = Path(__file__).parent.parent / "data"
SEISMIC_PROCESSED_DIR = DATA_DIR / "seismic_processed"


class SeismicRepository(ABC):
    @abstractmethod
    def save_dataset(
        self,
        metadata: SegyMetadata,
        traces: np.ndarray,
        twt_axis_ms: np.ndarray,
        trace_x: np.ndarray,
        trace_y: np.ndarray,
        trace_inline: np.ndarray,
        trace_crossline: np.ndarray,
        attributes: pd.DataFrame,
    ) -> None: ...

    @abstractmethod
    def get_dataset(
        self, dataset_id: str
    ) -> (
        tuple[SegyMetadata, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
        | None
    ): ...

    @abstractmethod
    def list_datasets(self) -> list[SegyMetadata]: ...

    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> bool: ...


class FileSeismicRepository(SeismicRepository):
    """Local-disk repository: one .npz (traces + time axis), one Parquet
    (per-trace attributes), and one JSON sidecar (metadata) per dataset,
    under `backend/data/seismic_processed/`.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or SEISMIC_PROCESSED_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _traces_path(self, dataset_id: str) -> Path:
        return self.base_dir / f"{dataset_id}.npz"

    def _attrs_path(self, dataset_id: str) -> Path:
        return self.base_dir / f"{dataset_id}.attrs.parquet"

    def _meta_path(self, dataset_id: str) -> Path:
        return self.base_dir / f"{dataset_id}.meta.json"

    def save_dataset(
        self,
        metadata: SegyMetadata,
        traces: np.ndarray,
        twt_axis_ms: np.ndarray,
        trace_x: np.ndarray,
        trace_y: np.ndarray,
        trace_inline: np.ndarray,
        trace_crossline: np.ndarray,
        attributes: pd.DataFrame,
    ) -> None:
        np.savez_compressed(
            self._traces_path(metadata.dataset_id),
            traces=traces,
            twt_axis_ms=twt_axis_ms,
            trace_x=trace_x,
            trace_y=trace_y,
            trace_inline=trace_inline,
            trace_crossline=trace_crossline,
        )
        attributes.to_parquet(self._attrs_path(metadata.dataset_id), index=False)
        with open(self._meta_path(metadata.dataset_id), "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    def get_dataset(
        self, dataset_id: str
    ) -> (
        tuple[SegyMetadata, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
        | None
    ):
        traces_path = self._traces_path(dataset_id)
        attrs_path = self._attrs_path(dataset_id)
        meta_path = self._meta_path(dataset_id)
        if not (traces_path.exists() and attrs_path.exists() and meta_path.exists()):
            return None

        npz = np.load(traces_path)
        attributes = pd.read_parquet(attrs_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = SegyMetadata(**meta_dict)

        n_traces = npz["traces"].shape[0]
        # Older cached datasets (processed before per-trace coordinates/
        # geometry were added) won't have these arrays in their .npz --
        # fall back to NaN rather than erroring, same as segy_loader does
        # when a raw file has no coordinate/geometry headers.
        trace_x = npz["trace_x"] if "trace_x" in npz.files else np.full(n_traces, np.nan)
        trace_y = npz["trace_y"] if "trace_y" in npz.files else np.full(n_traces, np.nan)
        trace_inline = npz["trace_inline"] if "trace_inline" in npz.files else np.full(n_traces, np.nan)
        trace_crossline = npz["trace_crossline"] if "trace_crossline" in npz.files else np.full(n_traces, np.nan)

        return (
            metadata,
            npz["traces"],
            npz["twt_axis_ms"],
            trace_x,
            trace_y,
            trace_inline,
            trace_crossline,
            attributes,
        )

    def list_datasets(self) -> list[SegyMetadata]:
        metas = []
        for meta_path in sorted(self.base_dir.glob("*.meta.json")):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
            metas.append(SegyMetadata(**meta_dict))
        return metas

    def delete_dataset(self, dataset_id: str) -> bool:
        traces_path = self._traces_path(dataset_id)
        attrs_path = self._attrs_path(dataset_id)
        meta_path = self._meta_path(dataset_id)
        existed = traces_path.exists() or attrs_path.exists() or meta_path.exists()
        traces_path.unlink(missing_ok=True)
        attrs_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed


_seismic_repository: SeismicRepository | None = None


def get_seismic_repository() -> SeismicRepository:
    """FastAPI dependency-injectable accessor for the active seismic repository."""
    global _seismic_repository
    if _seismic_repository is None:
        _seismic_repository = FileSeismicRepository()
    return _seismic_repository
