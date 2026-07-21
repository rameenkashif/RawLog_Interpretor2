"""
test_well_processing_cache_repository.py
-------------------------------------------
Unit tests for app/well_processing_cache_repository.py -- the disk-persisted,
well_id-keyed cache the dashboard-upload background pipeline writes into.
"""

from __future__ import annotations

from app.well_processing_cache_repository import (
    FileWellProcessingCacheRepository,
    WellProcessingCacheRecord,
)


def _record(well_id: str = "Z-02", status: str = "processing") -> WellProcessingCacheRecord:
    return WellProcessingCacheRecord(
        well_id=well_id,
        status=status,
        run_token="token-1",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class TestFileWellProcessingCacheRepository:
    def test_save_and_get_round_trip(self, tmp_path):
        repo = FileWellProcessingCacheRepository(base_dir=tmp_path)
        record = _record(status="ready")
        record.tie_available = True
        record.tie_correlation = 0.82
        record.tie_low_confidence = False
        repo.save(record)

        loaded = repo.get("Z-02")
        assert loaded is not None
        assert loaded.well_id == "Z-02"
        assert loaded.status == "ready"
        assert loaded.tie_available is True
        assert loaded.tie_correlation == 0.82
        assert loaded.tie_low_confidence is False

    def test_missing_well_returns_none(self, tmp_path):
        repo = FileWellProcessingCacheRepository(base_dir=tmp_path)
        assert repo.get("DOES_NOT_EXIST") is None

    def test_save_overwrites_previous(self, tmp_path):
        repo = FileWellProcessingCacheRepository(base_dir=tmp_path)
        repo.save(_record(status="processing"))
        repo.save(_record(status="ready"))
        assert repo.get("Z-02").status == "ready"

    def test_wells_are_independent(self, tmp_path):
        repo = FileWellProcessingCacheRepository(base_dir=tmp_path)
        repo.save(_record(well_id="Z-02", status="ready"))
        repo.save(_record(well_id="Z-03", status="failed"))
        assert repo.get("Z-02").status == "ready"
        assert repo.get("Z-03").status == "failed"

    def test_list_all(self, tmp_path):
        repo = FileWellProcessingCacheRepository(base_dir=tmp_path)
        repo.save(_record(well_id="Z-02"))
        repo.save(_record(well_id="Z-03"))
        well_ids = {r.well_id for r in repo.list_all()}
        assert well_ids == {"Z-02", "Z-03"}
