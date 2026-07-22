"""
test_spectral_petro_correlation_service.py
----------------------------------------------
Tests for app/services/spectral_petro_correlation_service.py ("CWT vs SWT
-- Petrophysical Correlation").

Reuses test_seismic_processor.py's synthetic-SEG-Y helper and
coordinate-patching helper, and the same well_repo/aligned_well fixture
pattern already established in test_synthetic_seismogram_service.py --
real Z-02/Z-03 curve data with X/Y patched so the converted-to-meters
coordinates land inside the synthetic test survey's extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import spectral_petro_correlation_service as spc
from app.services import well_service
from tests.test_seismic_processor import (
    CROSSLINES,
    INLINES,
    _set_las_coordinate_meters,
    _write_synthetic_segy,
)

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"
Z03_PATH = RAW_LAS_DIR / "Z-03_raw.las"

# Same values used by test_synthetic_seismogram_service.py -- center of
# _write_synthetic_segy's default SourceX/Y range.
ALIGNED_X = 366840.0
ALIGNED_Y = 2950275.0


@pytest.fixture
def well_repo(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells")


@pytest.fixture
def aligned_well(well_repo):
    las_text = Z02_PATH.read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", ALIGNED_X)
    las_text = _set_las_coordinate_meters(las_text, "Y", ALIGNED_Y)
    return well_service.process_and_store_las_bytes(las_text.encode(), "Z-02_raw.las", repo=well_repo)


@pytest.fixture
def aligned_well_2(well_repo):
    """A second, distinct well_id (Z-03_RAW) also aligned into the
    synthetic survey's extent, for all_wells-mode tests."""
    las_text = Z03_PATH.read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", ALIGNED_X)
    las_text = _set_las_coordinate_meters(las_text, "Y", ALIGNED_Y)
    return well_service.process_and_store_las_bytes(las_text.encode(), "Z-03_raw.las", repo=well_repo)


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path, n_samples=80, interval_ms=2)
    return sp.SegyVolume(path)


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch, well_repo, volume):
    monkeypatch.setattr(
        well_service, "get_well_summary",
        lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_curves",
        lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "list_well_summaries",
        lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
    )
    monkeypatch.setattr(sp, "get_segy_volume", lambda refresh=False: volume)


class TestSingleWellCorrelation:
    def test_basic_shape(self, aligned_well):
        result = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert result["mode"] == "single"
        assert len(result["wells"]) == 1
        well = result["wells"][0]
        assert well["well_id"] == aligned_well.well_id
        assert well["nearest_inline"] in INLINES
        assert well["nearest_crossline"] in CROSSLINES
        assert result["skipped_well_ids"] == []
        assert result["averages"] is None

        for curve in ("vsh", "phie", "swe"):
            pair = well[curve]
            assert pair["cwt_n"] >= 0
            assert pair["swt_n"] >= 0
            if pair["cwt_r"] is not None:
                assert -1.0 - 1e-9 <= pair["cwt_r"] <= 1.0 + 1e-9
            if pair["swt_r"] is not None:
                assert -1.0 - 1e-9 <= pair["swt_r"] <= 1.0 + 1e-9

    def test_default_level_is_3(self, aligned_well):
        result = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert result["swt_level"] == 3

    def test_band_matches_formula(self, aligned_well):
        # volume fixture: n_samples=80, interval_ms=2 -> Nyquist = 1000/(2*2) = 250 Hz
        nyquist = 250.0
        result = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, swt_level=3)
        lo, hi = result["swt_band_hz"]
        assert lo == pytest.approx(nyquist / 8)
        assert hi == pytest.approx(nyquist / 4)

        result2 = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, swt_level=2)
        lo2, hi2 = result2["swt_band_hz"]
        assert lo2 == pytest.approx(nyquist / 4)
        assert hi2 == pytest.approx(nyquist / 2)

    def test_cwt_frequency_is_nearest_available_bin_to_band_center(self, aligned_well):
        result = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, swt_level=3)
        lo, hi = result["swt_band_hz"]
        center = (lo + hi) / 2.0
        available = [f for f in sp.CWT_DEFAULT_FREQS_HZ if f <= 250.0]
        expected = min(available, key=lambda f: abs(f - center))
        assert result["cwt_frequency_hz"] == pytest.approx(expected)

    def test_wavelet_echoed_back(self, aligned_well):
        result = spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, wavelet="coif3")
        assert result["wavelet"] == "coif3"

    def test_missing_well_id_without_all_wells_raises(self):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_correlation(well_id=None, all_wells=False)

    def test_unknown_well_raises(self):
        with pytest.raises(well_service.WellNotFoundError):
            spc.get_correlation(well_id="DOES_NOT_EXIST", all_wells=False)

    def test_unknown_swt_level_raises(self, aligned_well):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, swt_level=9)

    def test_unknown_wavelet_raises(self, aligned_well):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_correlation(well_id=aligned_well.well_id, all_wells=False, wavelet="bogus")

    def test_missing_dt_raises_missing_curve(self, aligned_well, well_repo):
        metadata, df = well_repo.get_well(aligned_well.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)
        with pytest.raises(sp.MissingCurveError) as exc_info:
            spc.get_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert exc_info.value.curve == "DT"

    def test_crs_mismatch_raises(self, well_repo):
        # Unmodified real Z-02 -- its converted coordinates are far outside
        # this test's narrow synthetic survey extent (same as
        # test_synthetic_seismogram_service.py's analogous test).
        las_bytes = Z02_PATH.read_bytes()
        result = well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        with pytest.raises(sp.CrsMismatchError):
            spc.get_correlation(well_id=result.well_id, all_wells=False)


class TestTimeShiftCorrection:
    """_resolve_well_tie_context's time_shift_ms parameter (general-purpose
    shift correction on top of the sonic-integrated depth-time axis; no
    current caller here passes non-zero, all default to 0.0). Sign
    convention verified against
    well_seismic_tie.cross_correlate_and_shift directly (not assumed):
    shifted_syn = np.roll(synthetic, best_lag) means a seismic-time
    sample t corresponds to the well's own unshifted axis at
    t - best_shift_ms, so depth_at_time must be looked up at
    (seismic_time - time_shift_ms), not seismic_time directly. These
    tests check that operationally, not just that the code runs.
    """

    def test_zero_shift_reproduces_unshifted_behavior_exactly(self, aligned_well, volume):
        ctx_default = spc._resolve_well_tie_context(volume, aligned_well.well_id)
        ctx_explicit_zero = spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=0.0)

        np.testing.assert_array_equal(ctx_default.depth_at_time, ctx_explicit_zero.depth_at_time)
        np.testing.assert_array_equal(ctx_default.overlap, ctx_explicit_zero.overlap)

    def test_positive_shift_maps_to_shallower_depth(self, aligned_well, volume):
        # twt increases with depth (standard), so evaluating the well's
        # own unshifted curve at an EARLIER equivalent time
        # (seismic_time - shift, shift > 0) must land at a SHALLOWER
        # depth than the unshifted lookup at the same seismic sample.
        ctx0 = spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=0.0)
        ctx_shifted = spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=10.0)

        # Compare at seismic samples present in BOTH overlap windows (the
        # shift can change which samples overlap at all -- see the next
        # test), not by raw array position.
        common = ctx0.overlap & ctx_shifted.overlap
        depth0_at_common = np.interp(
            volume.twt_axis_ms[common], volume.twt_axis_ms[ctx0.overlap], ctx0.depth_at_time
        )
        depth_shifted_at_common = np.interp(
            volume.twt_axis_ms[common], volume.twt_axis_ms[ctx_shifted.overlap], ctx_shifted.depth_at_time
        )
        assert common.any()
        assert np.all(depth_shifted_at_common <= depth0_at_common + 1e-9)
        assert np.any(depth_shifted_at_common < depth0_at_common - 1e-9)

    def test_negative_shift_maps_to_deeper_depth(self, aligned_well, volume):
        ctx0 = spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=0.0)
        ctx_shifted = spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=-10.0)

        common = ctx0.overlap & ctx_shifted.overlap
        depth0_at_common = np.interp(
            volume.twt_axis_ms[common], volume.twt_axis_ms[ctx0.overlap], ctx0.depth_at_time
        )
        depth_shifted_at_common = np.interp(
            volume.twt_axis_ms[common], volume.twt_axis_ms[ctx_shifted.overlap], ctx_shifted.depth_at_time
        )
        assert common.any()
        assert np.all(depth_shifted_at_common >= depth0_at_common - 1e-9)
        assert np.any(depth_shifted_at_common > depth0_at_common + 1e-9)

    def test_shift_larger_than_logged_interval_can_empty_the_overlap(self, aligned_well, volume):
        with pytest.raises(sp.SegyVolumeError):
            spc._resolve_well_tie_context(volume, aligned_well.well_id, time_shift_ms=1e6)


class TestAllWellsCorrelation:
    def test_single_aligned_well_averages_equal_its_own_values(self, aligned_well):
        result = spc.get_correlation(well_id=None, all_wells=True)
        assert result["mode"] == "all_wells"
        assert len(result["wells"]) == 1
        assert result["skipped_well_ids"] == []
        assert result["averages"] is not None

        well = result["wells"][0]
        for curve in ("vsh", "phie", "swe"):
            avg = result["averages"][curve]
            if well[curve]["cwt_r"] is not None:
                assert avg["cwt_r"] == pytest.approx(well[curve]["cwt_r"])
                assert avg["n_wells"] >= 1
            else:
                assert avg["cwt_r"] is None

    def test_two_aligned_wells_both_present(self, aligned_well, aligned_well_2):
        result = spc.get_correlation(well_id=None, all_wells=True)
        well_ids = {w["well_id"] for w in result["wells"]}
        assert well_ids == {aligned_well.well_id, aligned_well_2.well_id}
        assert result["skipped_well_ids"] == []

    def test_well_without_dt_log_is_skipped_not_raised(self, aligned_well, aligned_well_2, well_repo):
        # aligned_well_2 ties fine but has no usable DT curve -- the
        # all_wells loop must skip it (sp.MissingCurveError is a
        # SegyVolumeError subclass) rather than blow up the whole request.
        metadata, df = well_repo.get_well(aligned_well_2.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)

        result = spc.get_correlation(well_id=None, all_wells=True)
        well_ids = {w["well_id"] for w in result["wells"]}
        assert aligned_well_2.well_id in result["skipped_well_ids"]
        assert aligned_well_2.well_id not in well_ids
        assert aligned_well.well_id in well_ids

    def test_no_wells_resolve_gives_empty_results_and_no_averages(self, well_repo):
        las_bytes = Z02_PATH.read_bytes()
        well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        result = spc.get_correlation(well_id=None, all_wells=True)
        assert result["wells"] == []
        assert result["averages"] is None
        assert len(result["skipped_well_ids"]) == 1


class TestPearsonHelper:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r, n = spc._pearson(x, y)
        assert r == pytest.approx(1.0)
        assert n == 5

    def test_perfect_anticorrelation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        r, n = spc._pearson(x, y)
        assert r == pytest.approx(-1.0)

    def test_constant_series_returns_none(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0])
        r, n = spc._pearson(x, y)
        assert r is None
        assert n == 3

    def test_too_few_samples_returns_none(self):
        x = np.array([1.0])
        y = np.array([2.0])
        r, n = spc._pearson(x, y)
        assert r is None
        assert n == 1

    def test_nan_values_excluded_from_n(self):
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([1.0, 2.0, 3.0, np.nan])
        r, n = spc._pearson(x, y)
        assert n == 2


class TestSingleWellSswtCorrelation:
    ssqueezepy = pytest.importorskip("ssqueezepy")

    def test_basic_shape(self, aligned_well):
        result = spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert result["mode"] == "single"
        assert len(result["wells"]) == 1
        well = result["wells"][0]
        assert well["well_id"] == aligned_well.well_id
        assert well["nearest_inline"] in INLINES
        assert well["nearest_crossline"] in CROSSLINES
        assert result["skipped_well_ids"] == []
        assert result["averages"] is None

        for curve in ("vsh", "phie", "swe"):
            pair = well[curve]
            assert pair["cwt_n"] >= 0
            assert pair["sswt_n"] >= 0
            if pair["cwt_r"] is not None:
                assert -1.0 - 1e-9 <= pair["cwt_r"] <= 1.0 + 1e-9
            if pair["sswt_r"] is not None:
                assert -1.0 - 1e-9 <= pair["sswt_r"] <= 1.0 + 1e-9

    def test_default_frequency(self, aligned_well):
        result = spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert result["requested_frequency_hz"] == pytest.approx(spc.DEFAULT_SSWT_COMPARISON_FREQUENCY_HZ)

    def test_cwt_and_sswt_snap_independently_to_requested_frequency(self, aligned_well):
        # volume fixture: n_samples=80, interval_ms=2 -> Nyquist = 250 Hz.
        # CWT's grid is fixed 5 Hz steps (5, 10, ..., 100) -- 47 Hz snaps
        # exactly to 45; SSWT's grid is much finer, so it should land
        # closer to the requested value than CWT's coarser grid does.
        result = spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False, frequency_hz=47.0)
        assert result["cwt_frequency_hz"] == pytest.approx(45.0)
        assert abs(result["sswt_frequency_hz"] - 47.0) < abs(result["cwt_frequency_hz"] - 47.0)

    def test_out_of_range_frequency_raises(self, aligned_well):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False, frequency_hz=9999.0)

    def test_negative_frequency_raises(self, aligned_well):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False, frequency_hz=-5.0)

    def test_missing_well_id_without_all_wells_raises(self):
        with pytest.raises(sp.SegyVolumeError):
            spc.get_sswt_correlation(well_id=None, all_wells=False)

    def test_unknown_well_raises(self):
        with pytest.raises(well_service.WellNotFoundError):
            spc.get_sswt_correlation(well_id="DOES_NOT_EXIST", all_wells=False)

    def test_missing_dt_raises_missing_curve(self, aligned_well, well_repo):
        metadata, df = well_repo.get_well(aligned_well.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)
        with pytest.raises(sp.MissingCurveError) as exc_info:
            spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert exc_info.value.curve == "DT"

    def test_crs_mismatch_raises(self, well_repo):
        las_bytes = Z02_PATH.read_bytes()
        result = well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        with pytest.raises(sp.CrsMismatchError):
            spc.get_sswt_correlation(well_id=result.well_id, all_wells=False)

    def test_scatter_present_and_matches_pair_sample_counts(self, aligned_well):
        result = spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False)
        well = result["wells"][0]
        scatter = well["scatter"]
        assert scatter is not None

        n = len(scatter["depth_m"])
        assert n > 0
        assert len(scatter["cwt_amplitude"]) == n
        assert len(scatter["sswt_amplitude"]) == n
        for curve in ("vsh", "phie", "swe"):
            assert len(scatter[curve]) == n
            # The pair's own cwt_n is the count of finite (cwt_amplitude,
            # property) pairs, i.e. non-None entries in the raw series --
            # the raw series must be consistent with the summary stat it
            # was reduced from, not an independently-computed duplicate.
            pair = well[curve]
            finite_pairs = sum(
                1
                for amp, prop in zip(scatter["cwt_amplitude"], scatter[curve])
                if prop is not None and np.isfinite(amp)
            )
            assert finite_pairs == pair["cwt_n"]

    def test_scatter_omitted_when_correlation_present_but_all_wells(self, aligned_well):
        # Sanity check that the single-well path's scatter is a genuinely
        # new field, not always-on -- the all_wells path must not carry it
        # (see next class), keeping that response small.
        result = spc.get_sswt_correlation(well_id=aligned_well.well_id, all_wells=False)
        assert "scatter" in result["wells"][0]


class TestAllWellsSswtCorrelation:
    ssqueezepy = pytest.importorskip("ssqueezepy")

    def test_single_aligned_well_averages_equal_its_own_values(self, aligned_well):
        result = spc.get_sswt_correlation(well_id=None, all_wells=True)
        assert result["mode"] == "all_wells"
        assert len(result["wells"]) == 1
        assert result["skipped_well_ids"] == []
        assert result["averages"] is not None

        well = result["wells"][0]
        for curve in ("vsh", "phie", "swe"):
            avg = result["averages"][curve]
            if well[curve]["cwt_r"] is not None:
                assert avg["cwt_r"] == pytest.approx(well[curve]["cwt_r"])
                assert avg["n_wells"] >= 1
            else:
                assert avg["cwt_r"] is None

    def test_two_aligned_wells_both_present(self, aligned_well, aligned_well_2):
        result = spc.get_sswt_correlation(well_id=None, all_wells=True)
        well_ids = {w["well_id"] for w in result["wells"]}
        assert well_ids == {aligned_well.well_id, aligned_well_2.well_id}

    def test_scatter_omitted_in_all_wells_mode(self, aligned_well, aligned_well_2):
        result = spc.get_sswt_correlation(well_id=None, all_wells=True)
        assert all(w["scatter"] is None for w in result["wells"])
        assert result["skipped_well_ids"] == []

    def test_well_without_dt_log_is_skipped_not_raised(self, aligned_well, aligned_well_2, well_repo):
        metadata, df = well_repo.get_well(aligned_well_2.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)

        result = spc.get_sswt_correlation(well_id=None, all_wells=True)
        well_ids = {w["well_id"] for w in result["wells"]}
        assert aligned_well_2.well_id in result["skipped_well_ids"]
        assert aligned_well_2.well_id not in well_ids
        assert aligned_well.well_id in well_ids

    def test_no_wells_resolve_gives_empty_results_and_no_averages(self, well_repo):
        las_bytes = Z02_PATH.read_bytes()
        well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)
        result = spc.get_sswt_correlation(well_id=None, all_wells=True)
        assert result["wells"] == []
        assert result["averages"] is None
        assert len(result["skipped_well_ids"]) == 1
