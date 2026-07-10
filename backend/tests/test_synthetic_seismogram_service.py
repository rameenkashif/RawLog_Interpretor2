"""
test_synthetic_seismogram_service.py
---------------------------------------
Integration tests for app/services/synthetic_seismogram_service.py (the
"Synthetic Seismogram" module) and its router (app/routers/synthetic.py).

Reuses test_seismic_processor.py's synthetic-SEG-Y helper (same
non-standard header layout as the real file) and coordinate-patching
helpers, isolated FileWellRepository/FileSeismicRepository instances, and
monkeypatched well_service/seismic_processor module functions -- same
conventions already established for the well-tie tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app import well_seismic_tie as wst
from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import synthetic_seismogram_service as sss
from app.services import well_service
from app.synthetic_tie_repository import FileSyntheticTieRepository, TiePoint, TiePointSet
from tests.test_seismic_processor import (
    CROSSLINES,
    INLINES,
    _set_las_coordinate_meters,
    _write_synthetic_segy,
)

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"

# Center of _write_synthetic_segy's default SourceX/Y range (source_x_base=
# 363000 + il*10 for il in 382..386, source_y_base=2949800 + xl*10 for xl in
# 46..49) -- same values used by test_seismic_processor.py's aligned-well tests.
ALIGNED_X = 366840.0
ALIGNED_Y = 2950275.0


@pytest.fixture
def well_repo(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells")


@pytest.fixture
def tie_repo(tmp_path):
    return FileSyntheticTieRepository(base_dir=tmp_path / "ties")


@pytest.fixture
def aligned_well(well_repo):
    """Real Z-02 curve data, with X/Y patched (in feet, so unit
    standardization still applies) so the converted meters land inside the
    synthetic test survey's coordinate extent."""
    las_text = Z02_PATH.read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", ALIGNED_X)
    las_text = _set_las_coordinate_meters(las_text, "Y", ALIGNED_Y)
    return well_service.process_and_store_las_bytes(las_text.encode(), "Z-02_raw.las", repo=well_repo)


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path, n_samples=80, interval_ms=2)
    return sp.SegyVolume(path)


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch, well_repo, tie_repo, volume):
    monkeypatch.setattr(
        well_service, "get_well_summary",
        lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_curves",
        lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(sp, "get_segy_volume", lambda refresh=False: volume)
    # synthetic_seismogram_service imports get_synthetic_tie_repository
    # directly (`from ... import get_synthetic_tie_repository`), so the name
    # to patch is the one bound in ITS namespace, not the origin module's.
    monkeypatch.setattr(sss, "get_synthetic_tie_repository", lambda: tie_repo)


class TestGenerate:
    def test_full_pipeline_succeeds(self, aligned_well):
        result = sss.generate(aligned_well.well_id, wavelet_method="ricker", wavelet_freq_hz=25.0, density_method="rhob")
        assert result["well_id"] == aligned_well.well_id
        assert result["density_method"] == "rhob"
        assert result["well_header"]["coordinate_unit_detected"] == "feet"
        assert result["well_header"]["unit_conversion_applied"] is True
        assert len(result["washout_flag"]) == len(result["washout_depth_m"])
        assert len(result["seismic_twt_ms"]) == len(result["synthetic"]) == len(result["real_trace"])
        assert len(result["wavelet_spectrum_freq_hz"]) == len(result["wavelet_spectrum_amplitude"])
        assert result["applied_tie_points"] == []
        assert "sonic" in result["time_depth_note"].lower()
        assert "vertical" in result["vertical_assumption_note"].lower()
        # This test's synthetic SEG-Y uses a real (non-zero) recording delay
        # (DELAY_MS=2030), matching the actual production survey -- the
        # synthetic trace must be anchored onto that same absolute time
        # window, not silently come out all-zero from a 0-based sonic
        # integration curve having no overlap with it (see well_seismic_tie
        # depth_to_twt/build_synthetic's t0_ms anchoring).
        assert max(abs(v) for v in result["synthetic"]) > 0.0
        assert result["reflectivity_twt_ms"][0] == pytest.approx(2030.0, abs=1.0)
        # Fixes #7/#8/#9: datum plausibility check, bulk-shift search
        # range, and boundary-pinned reliability flag are all surfaced,
        # not just the raw correlation number.
        assert result["max_shift_ms"] == wst.DEFAULT_MAX_SHIFT_MS
        assert isinstance(result["boundary_pinned"], bool)
        assert result["datum_check"]["delay_ms"] == pytest.approx(2030.0)
        assert isinstance(result["datum_check"]["plausible"], bool)

    def test_generate_accepts_custom_max_shift_ms(self, aligned_well):
        result = sss.generate(aligned_well.well_id, max_shift_ms=50.0)
        assert result["max_shift_ms"] == 50.0
        assert abs(result["best_shift_ms"]) <= 50.0 + 1e-6

    def test_statistical_wavelet(self, aligned_well):
        result = sss.generate(aligned_well.well_id, wavelet_method="statistical", density_method="rhob")
        assert result["wavelet_method"] == "statistical"
        assert len(result["wavelet_amplitude"]) > 0

    def test_gardner_density_calibrates_with_real_rhob(self, aligned_well):
        result = sss.generate(aligned_well.well_id, density_method="gardner")
        assert result["density_method"] == "gardner"
        assert result["gardner_coefficients"] is not None
        assert result["gardner_coefficients"]["calibrated"] is True

    def test_rock_physics_density_succeeds(self, aligned_well):
        # VSH/PHIE are computed as part of the normal LAS processing
        # pipeline (petrophysics.py), so they're present for any well
        # loaded through process_and_store_las_bytes.
        result = sss.generate(aligned_well.well_id, density_method="rock_physics")
        assert result["density_method"] == "rock_physics"
        assert result["gardner_coefficients"] is None

    def test_unknown_density_method_raises(self, aligned_well):
        with pytest.raises(sss.SyntheticSeismogramError):
            sss.generate(aligned_well.well_id, density_method="bogus")

    def test_unknown_wavelet_method_raises(self, aligned_well):
        with pytest.raises(sss.SyntheticSeismogramError):
            sss.generate(aligned_well.well_id, wavelet_method="bogus")

    def test_crs_mismatch_still_raises(self, well_repo):
        # Unmodified real Z-02 -- its converted coordinates are far outside
        # this test's narrow synthetic survey extent.
        las_bytes = Z02_PATH.read_bytes()
        result = well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        with pytest.raises(sp.CrsMismatchError):
            sss.generate(result.well_id)

    def test_missing_dt_raises_missing_curve(self, aligned_well, well_repo):
        metadata, df = well_repo.get_well(aligned_well.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)
        with pytest.raises(sss.MissingCurveError) as exc_info:
            sss.generate(aligned_well.well_id)
        assert exc_info.value.curve == "DT"

    def test_missing_rhob_with_rhob_density_method_raises(self, aligned_well, well_repo):
        metadata, df = well_repo.get_well(aligned_well.well_id)
        df["RHOB"] = np.nan
        well_repo.save_well(metadata, df)
        with pytest.raises(sss.MissingCurveError) as exc_info:
            sss.generate(aligned_well.well_id, density_method="rhob")
        assert exc_info.value.curve == "RHOB"

    def test_applied_saved_tie_point_shifts_reflectivity_time_axis(self, aligned_well, tie_repo):
        baseline = sss.generate(aligned_well.well_id, apply_saved_tie=False)
        tie_repo.save_tie_points(
            TiePointSet(
                well_id=aligned_well.well_id,
                points=[TiePoint(md_m=baseline["reflectivity_depth_m"][0], time_shift_ms=50.0)],
            )
        )
        shifted = sss.generate(aligned_well.well_id, apply_saved_tie=True)
        assert len(shifted["applied_tie_points"]) == 1
        # A uniform +50ms shift at every control point (only one here) should
        # shift the whole reflectivity time axis by +50ms.
        np.testing.assert_allclose(
            np.array(shifted["reflectivity_twt_ms"]) - np.array(baseline["reflectivity_twt_ms"]),
            50.0,
            atol=1e-6,
        )

    def test_apply_saved_tie_false_ignores_saved_points(self, aligned_well, tie_repo):
        tie_repo.save_tie_points(
            TiePointSet(well_id=aligned_well.well_id, points=[TiePoint(md_m=3500.0, time_shift_ms=999.0)])
        )
        result = sss.generate(aligned_well.well_id, apply_saved_tie=False)
        assert result["applied_tie_points"] == []


class TestNearestTrace:
    def test_returns_geometry(self, aligned_well):
        result = sss.nearest_trace(aligned_well.well_id)
        assert result["well_id"] == aligned_well.well_id
        assert result["distance_m"] >= 0
        assert result["inline"] in INLINES
        assert result["crossline"] in CROSSLINES

    def test_crs_mismatch_raises(self, well_repo):
        las_bytes = Z02_PATH.read_bytes()
        result = well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        with pytest.raises(sp.CrsMismatchError):
            sss.nearest_trace(result.well_id)


class TestSaveAndGetTiePoints:
    def test_round_trip(self, aligned_well, tie_repo):
        saved = sss.save_tie_points(
            aligned_well.well_id,
            [{"md_m": 3500.0, "time_shift_ms": 4.0}, {"md_m": 3600.0, "time_shift_ms": -2.0}],
            "statistical",
            30.0,
        )
        assert len(saved.points) == 2
        assert saved.segy_filename is not None

        loaded = sss.get_tie_points(aligned_well.well_id)
        assert loaded is not None
        assert len(loaded.points) == 2
        assert loaded.wavelet_freq_hz == 30.0

    def test_no_saved_points_returns_none(self, aligned_well):
        assert sss.get_tie_points(aligned_well.well_id) is None
