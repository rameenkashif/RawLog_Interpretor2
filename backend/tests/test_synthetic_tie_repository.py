"""
test_synthetic_tie_repository.py
-----------------------------------
Unit tests for app/synthetic_tie_repository.py -- manual well-tie
stretch/squeeze control point persistence.
"""

from __future__ import annotations

from app.synthetic_tie_repository import FileSyntheticTieRepository, TiePoint, TiePointSet


class TestFileSyntheticTieRepository:
    def test_save_and_load_round_trip(self, tmp_path):
        repo = FileSyntheticTieRepository(base_dir=tmp_path)
        tie = TiePointSet(
            well_id="Z-02_RAW",
            points=[TiePoint(md_m=3500.0, time_shift_ms=4.5), TiePoint(md_m=3600.0, time_shift_ms=-2.0)],
            wavelet_method="statistical",
            wavelet_freq_hz=30.0,
            segy_filename="origional.segy",
        )
        repo.save_tie_points(tie)

        loaded = repo.get_tie_points("Z-02_RAW")
        assert loaded is not None
        assert loaded.well_id == "Z-02_RAW"
        assert len(loaded.points) == 2
        assert loaded.points[0].md_m == 3500.0
        assert loaded.points[0].time_shift_ms == 4.5
        assert loaded.wavelet_method == "statistical"
        assert loaded.wavelet_freq_hz == 30.0
        assert loaded.segy_filename == "origional.segy"

    def test_missing_well_returns_none(self, tmp_path):
        repo = FileSyntheticTieRepository(base_dir=tmp_path)
        assert repo.get_tie_points("DOES_NOT_EXIST") is None

    def test_save_overwrites_previous(self, tmp_path):
        repo = FileSyntheticTieRepository(base_dir=tmp_path)
        repo.save_tie_points(TiePointSet(well_id="Z-02_RAW", points=[TiePoint(md_m=1.0, time_shift_ms=1.0)]))
        repo.save_tie_points(TiePointSet(well_id="Z-02_RAW", points=[]))
        loaded = repo.get_tie_points("Z-02_RAW")
        assert loaded.points == []

    def test_delete(self, tmp_path):
        repo = FileSyntheticTieRepository(base_dir=tmp_path)
        repo.save_tie_points(TiePointSet(well_id="Z-02_RAW"))
        assert repo.delete_tie_points("Z-02_RAW") is True
        assert repo.get_tie_points("Z-02_RAW") is None
        assert repo.delete_tie_points("Z-02_RAW") is False

    def test_wells_are_independent(self, tmp_path):
        repo = FileSyntheticTieRepository(base_dir=tmp_path)
        repo.save_tie_points(TiePointSet(well_id="Z-02_RAW", points=[TiePoint(md_m=1.0, time_shift_ms=1.0)]))
        repo.save_tie_points(TiePointSet(well_id="Z-03_RAW", points=[]))
        assert len(repo.get_tie_points("Z-02_RAW").points) == 1
        assert len(repo.get_tie_points("Z-03_RAW").points) == 0
