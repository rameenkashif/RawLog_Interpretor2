"""
test_coordinate_tie_override_repository.py
-----------------------------------------------
Tests for app/coordinate_tie_override_repository.py: persistence of
manual well->trace tie-point overrides (fix #5 in the calibration audit).
"""

from __future__ import annotations

import pytest

from app.coordinate_tie_override_repository import (
    FileCoordinateTieOverrideRepository,
    WellTraceOverride,
)


@pytest.fixture
def repo(tmp_path):
    return FileCoordinateTieOverrideRepository(base_dir=tmp_path / "overrides")


class TestFileCoordinateTieOverrideRepository:
    def test_save_and_get_round_trip(self, repo):
        repo.save_override(WellTraceOverride(well_id="Z-08", inline=420, crossline=130, note="confirmed by geologist"))
        loaded = repo.get_override("Z-08")
        assert loaded is not None
        assert loaded.inline == 420
        assert loaded.crossline == 130
        assert loaded.note == "confirmed by geologist"

    def test_get_missing_returns_none(self, repo):
        assert repo.get_override("NOPE") is None

    def test_list_overrides_returns_all(self, repo):
        repo.save_override(WellTraceOverride(well_id="Z-02", inline=400, crossline=100))
        repo.save_override(WellTraceOverride(well_id="Z-08", inline=420, crossline=130))
        overrides = repo.list_overrides()
        assert {o.well_id for o in overrides} == {"Z-02", "Z-08"}

    def test_delete_override(self, repo):
        repo.save_override(WellTraceOverride(well_id="Z-02", inline=400, crossline=100))
        assert repo.delete_override("Z-02") is True
        assert repo.get_override("Z-02") is None
        assert repo.delete_override("Z-02") is False

    def test_save_overwrites_existing(self, repo):
        repo.save_override(WellTraceOverride(well_id="Z-02", inline=400, crossline=100))
        repo.save_override(WellTraceOverride(well_id="Z-02", inline=401, crossline=101))
        loaded = repo.get_override("Z-02")
        assert loaded.inline == 401
        assert loaded.crossline == 101
