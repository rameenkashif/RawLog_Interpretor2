"""
test_checkshot_repository.py
--------------------------------
Unit tests for app/checkshot_repository.py -- the disk-persisted,
well_id-keyed store of uploaded checkshot points.
"""

from __future__ import annotations

from app.checkshot_repository import CheckshotRecord, FileCheckshotRepository


class TestFileCheckshotRepository:
    def test_save_and_get_round_trip(self, tmp_path):
        repo = FileCheckshotRepository(base_dir=tmp_path)
        record = CheckshotRecord(well_id="Z-02", points=[[3354.8, 2045.0], [3409.01, 2101.0]])
        repo.save(record)

        loaded = repo.get("Z-02")
        assert loaded is not None
        assert loaded.well_id == "Z-02"
        assert loaded.points == [[3354.8, 2045.0], [3409.01, 2101.0]]

    def test_get_missing_well_returns_none(self, tmp_path):
        repo = FileCheckshotRepository(base_dir=tmp_path)
        assert repo.get("Z-99") is None

    def test_save_overwrites_existing_record(self, tmp_path):
        repo = FileCheckshotRepository(base_dir=tmp_path)
        repo.save(CheckshotRecord(well_id="Z-02", points=[[3000.0, 2000.0]]))
        repo.save(CheckshotRecord(well_id="Z-02", points=[[3000.0, 2000.0], [3100.0, 2050.0]]))

        loaded = repo.get("Z-02")
        assert len(loaded.points) == 2

    def test_list_all_returns_every_stored_well(self, tmp_path):
        repo = FileCheckshotRepository(base_dir=tmp_path)
        repo.save(CheckshotRecord(well_id="Z-02", points=[[3000.0, 2000.0]]))
        repo.save(CheckshotRecord(well_id="Z-03", points=[[3100.0, 2050.0]]))

        records = repo.list_all()
        assert {r.well_id for r in records} == {"Z-02", "Z-03"}

    def test_base_dir_created_if_missing(self, tmp_path):
        base = tmp_path / "nested" / "checkshots"
        assert not base.exists()
        FileCheckshotRepository(base_dir=base)
        assert base.exists()
