"""
checkshot_repository.py
--------------------------
Disk-persisted, well_id-keyed store of real checkshot / time-depth-survey
points (depth_m, twt_ms), uploaded once per field as a workbook (one sheet
per well) via services/checkshot_service.py.

Same file-per-entity JSON pattern as well_processing_cache_repository.py /
synthetic_tie_repository.py / coordinate_calibration_repository.py: ABC ->
File impl -> module-global singleton -> get_*_repository() accessor.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKSHOT_DIR = DATA_DIR / "checkshots"


@dataclass
class CheckshotRecord:
    well_id: str
    # Sorted ascending by depth_m at save time -- callers (well_seismic_tie.
    # checkshot_anchor_shift) rely on the shallowest point being points[0].
    points: list[list[float]]  # [[depth_m, twt_ms], ...]


class CheckshotRepository(ABC):
    @abstractmethod
    def save(self, record: CheckshotRecord) -> None: ...

    @abstractmethod
    def get(self, well_id: str) -> CheckshotRecord | None: ...

    @abstractmethod
    def list_all(self) -> list[CheckshotRecord]: ...


class FileCheckshotRepository(CheckshotRepository):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or CHECKSHOT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, well_id: str) -> Path:
        return self.base_dir / f"{well_id}.json"

    def save(self, record: CheckshotRecord) -> None:
        with open(self._path(record.well_id), "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

    def get(self, well_id: str) -> CheckshotRecord | None:
        path = self._path(well_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return CheckshotRecord(**json.load(f))

    def list_all(self) -> list[CheckshotRecord]:
        records = []
        for path in sorted(self.base_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                records.append(CheckshotRecord(**json.load(f)))
        return records


_repository: CheckshotRepository | None = None


def get_checkshot_repository() -> CheckshotRepository:
    global _repository
    if _repository is None:
        _repository = FileCheckshotRepository()
    return _repository
