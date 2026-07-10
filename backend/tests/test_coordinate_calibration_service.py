"""
test_coordinate_calibration_service.py
-------------------------------------------
Tests for app/services/coordinate_calibration_service.py: the
orchestration layer that fits/persists the per-axis coordinate
calibration against a stable baseline of wells + the current SEG-Y
survey, validates other wells against that FROZEN baseline (not a
constantly-refit one -- see module docstring for why that distinction is
the whole point of fix #5), merges in manual tie-point overrides, and is
the single well->trace resolution path downstream tie/prediction
workflows should use (fixes #4/#5).
"""

from __future__ import annotations

import pytest

segyio = pytest.importorskip("segyio")

from app.coordinate_calibration_repository import FileCoordinateCalibrationRepository
from app.coordinate_tie_override_repository import FileCoordinateTieOverrideRepository, WellTraceOverride
from app.models.schemas import WellSummary
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import well_service
from tests.test_seismic_processor import _write_synthetic_segy

# Survey extent from _write_synthetic_segy's defaults: X 366820-366860 (5
# inlines, 10 apart), Y 2950260-2950290 (4 crosslines, 10 apart).


def _summary(well_id: str, well_x: float | None, well_y: float | None) -> WellSummary:
    return WellSummary(
        well_id=well_id,
        well_name=well_id,
        start_depth=0.0,
        stop_depth=100.0,
        step=0.5,
        n_samples=200,
        footage_logged=100.0,
        well_x=well_x,
        well_y=well_y,
    )


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path)
    return sp.SegyVolume(path)


@pytest.fixture
def override_repo(tmp_path):
    return FileCoordinateTieOverrideRepository(base_dir=tmp_path / "overrides")


@pytest.fixture
def calibration_repo(tmp_path):
    return FileCoordinateCalibrationRepository(path=tmp_path / "calibration.json")


# Raw well coordinates chosen so the fit maps them exactly onto real trace
# positions: well_x/y span [0, 40] / [0, 30], survey spans
# [366820, 366860] / [2950260, 2950290] -- both exactly 40 and 30 wide, so
# a=c=1 and every well below lands exactly on an existing trace after the
# transform (b=366820, d=2950260).
GOOD_WELLS = [
    _summary("W-LOW", 0.0, 0.0),  # -> (366820, 2950260) = inline 382, crossline 46
    _summary("W-MID", 20.0, 15.0),  # -> (366840, 2950275) -- between grid traces
    _summary("W-HIGH", 40.0, 30.0),  # -> (366860, 2950290) = inline 386, crossline 49
]
# Deliberately NOT included when the calibration baseline is first fit
# below -- mirrors the real scenario (fix #5): a well discovered/added
# AFTER a calibration baseline was established, checked against that
# already-frozen fit rather than silently expanding the fit's own "valid"
# range to include it (which would make it structurally unflaggable).
EXTRAPOLATED_WELL = _summary("W-FAR", 50000.0, 15.0)  # raw coord far outside [0,40]
NO_COORD_WELL = _summary("W-NONE", None, None)


def _patch_well_list(monkeypatch, wells: list[WellSummary]) -> None:
    """resolve_well_trace_index looks up a single well via get_well_summary
    (WellNotFoundError semantics for a truly nonexistent well_id), while
    the calibration fit needs the full list via list_well_summaries --
    patch both consistently from the same fake well set."""
    by_id = {w.well_id: w for w in wells}
    monkeypatch.setattr(well_service, "list_well_summaries", lambda repo=None: wells)

    def _get_well_summary(well_id: str, repo=None):
        if well_id not in by_id:
            raise well_service.WellNotFoundError(well_id)
        return by_id[well_id]

    monkeypatch.setattr(well_service, "get_well_summary", _get_well_summary)


def _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo):
    """Fit-and-store the baseline calibration from only GOOD_WELLS, as an
    explicit prior step -- simulates "the calibration was already
    established" before a new well is later checked against it."""
    _patch_well_list(monkeypatch, GOOD_WELLS)
    ccs.fit_and_store_calibration(volume, calibration_repo=calibration_repo)


class TestGetCalibrationReport:
    def test_reports_trustworthy_wells_within_extent(self, monkeypatch, volume, override_repo, calibration_repo):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        reports = ccs.get_calibration_report(volume, calibration_repo=calibration_repo, override_repo=override_repo)
        assert len(reports) == 3
        by_id = {r.well_id: r for r in reports}
        assert by_id["W-LOW"].trustworthy is True
        assert by_id["W-LOW"].nearest_inline == 382
        assert by_id["W-LOW"].nearest_crossline == 46
        assert by_id["W-HIGH"].nearest_inline == 386
        assert by_id["W-HIGH"].nearest_crossline == 49
        assert all(r.used_in_calibration for r in reports)

    def test_flags_well_added_after_calibration_was_established(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        # Now a new well shows up, discovered AFTER the baseline was fit.
        _patch_well_list(monkeypatch, GOOD_WELLS + [EXTRAPOLATED_WELL])
        reports = ccs.get_calibration_report(volume, calibration_repo=calibration_repo, override_repo=override_repo)
        by_id = {r.well_id: r for r in reports}
        assert by_id["W-FAR"].trustworthy is False
        assert by_id["W-FAR"].is_extrapolated is True
        assert by_id["W-FAR"].used_in_calibration is False

    def test_manual_override_surfaced_in_report(self, monkeypatch, volume, override_repo, calibration_repo):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        override_repo.save_override(WellTraceOverride(well_id="W-FAR", inline=384, crossline=47, note="confirmed"))
        _patch_well_list(monkeypatch, GOOD_WELLS + [EXTRAPOLATED_WELL])
        reports = ccs.get_calibration_report(volume, calibration_repo=calibration_repo, override_repo=override_repo)
        by_id = {r.well_id: r for r in reports}
        assert by_id["W-FAR"].has_manual_override is True
        assert by_id["W-FAR"].override_inline == 384
        assert by_id["W-FAR"].override_crossline == 47
        # The override is informational in the report -- the underlying
        # calibration flag is still shown, since it doesn't change on its
        # own just because an override exists.
        assert by_id["W-FAR"].is_extrapolated is True

    def test_bootstraps_automatically_when_nothing_stored(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        # No prior fit_and_store_calibration call -- should auto-bootstrap
        # from whatever wells currently exist rather than erroring.
        _patch_well_list(monkeypatch, GOOD_WELLS)
        reports = ccs.get_calibration_report(volume, calibration_repo=calibration_repo, override_repo=override_repo)
        assert len(reports) == 3

    def test_too_few_wells_raises(self, monkeypatch, volume, override_repo, calibration_repo):
        _patch_well_list(monkeypatch, GOOD_WELLS[:1])
        with pytest.raises(Exception):
            ccs.get_calibration_report(volume, calibration_repo=calibration_repo, override_repo=override_repo)


class TestResolveWellTraceIndex:
    def test_trustworthy_well_resolves_via_calibrated_fit(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        idx, distance_m, method = ccs.resolve_well_trace_index(
            volume, "W-LOW", calibration_repo=calibration_repo, override_repo=override_repo
        )
        assert method == "calibrated_fit"
        assert distance_m == pytest.approx(0.0, abs=1e-6)

    def test_well_added_after_calibration_raises_without_override(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        _patch_well_list(monkeypatch, GOOD_WELLS + [EXTRAPOLATED_WELL])
        with pytest.raises(ccs.UnresolvedCoordinateError):
            ccs.resolve_well_trace_index(
                volume, "W-FAR", calibration_repo=calibration_repo, override_repo=override_repo
            )

    def test_single_well_falls_back_to_direct_unvalidated(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        # Only 1 well with coordinates exists anywhere -- a per-axis
        # calibration can't be fit (needs >= 2), so this must fall back to
        # the legacy buffer-check + direct nearest-trace search rather
        # than erroring outright. The fallback compares RAW well
        # coordinates directly against the survey's RAW extent (no
        # calibration transform is available), so this well's raw
        # coordinate must itself already be in the survey's coordinate
        # system (unlike GOOD_WELLS, which need the calibration transform).
        same_crs_well = _summary("W-SAMECRS", 366825.0, 2950262.0)
        _patch_well_list(monkeypatch, [same_crs_well])
        idx, distance_m, method = ccs.resolve_well_trace_index(
            volume, "W-SAMECRS", calibration_repo=calibration_repo, override_repo=override_repo
        )
        assert method == "direct_unvalidated"
        assert int(volume.inline[idx]) == 382
        assert int(volume.crossline[idx]) == 46

    def test_single_well_far_outside_survey_raises_crs_mismatch(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        far_well = _summary("W-DISTANT", 9_000_000.0, 9_000_000.0)
        _patch_well_list(monkeypatch, [far_well])
        with pytest.raises(sp.CrsMismatchError):
            ccs.resolve_well_trace_index(
                volume, "W-DISTANT", calibration_repo=calibration_repo, override_repo=override_repo
            )

    def test_manual_override_bypasses_calibration_entirely(
        self, monkeypatch, volume, override_repo, calibration_repo
    ):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        override_repo.save_override(WellTraceOverride(well_id="W-FAR", inline=384, crossline=47))
        _patch_well_list(monkeypatch, GOOD_WELLS + [EXTRAPOLATED_WELL])
        idx, distance_m, method = ccs.resolve_well_trace_index(
            volume, "W-FAR", calibration_repo=calibration_repo, override_repo=override_repo
        )
        assert method == "manual_override"
        assert distance_m is None
        assert int(volume.inline[idx]) == 384
        assert int(volume.crossline[idx]) == 47

    def test_override_with_invalid_trace_raises(self, monkeypatch, volume, override_repo, calibration_repo):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        override_repo.save_override(WellTraceOverride(well_id="W-FAR", inline=99999, crossline=99999))
        with pytest.raises(ccs.UnresolvedCoordinateError):
            ccs.resolve_well_trace_index(
                volume, "W-FAR", calibration_repo=calibration_repo, override_repo=override_repo
            )

    def test_well_with_no_coordinates_raises(self, monkeypatch, volume, override_repo, calibration_repo):
        _bootstrap_calibration_from_good_wells(monkeypatch, volume, calibration_repo)
        _patch_well_list(monkeypatch, GOOD_WELLS + [NO_COORD_WELL])
        with pytest.raises(ccs.UnresolvedCoordinateError):
            ccs.resolve_well_trace_index(
                volume, "W-NONE", calibration_repo=calibration_repo, override_repo=override_repo
            )


class TestFitAndStoreCalibration:
    def test_explicit_recalibration_with_curated_subset(self, monkeypatch, volume, calibration_repo):
        _patch_well_list(monkeypatch, GOOD_WELLS + [EXTRAPOLATED_WELL])
        # Explicitly recalibrate using only the two trusted wells,
        # excluding the known-bad one -- the real fix path per fix #5.
        cal, used_well_ids, bin_spacing_m = ccs.fit_and_store_calibration(
            volume, well_ids=["W-LOW", "W-HIGH"], calibration_repo=calibration_repo
        )
        assert set(used_well_ids) == {"W-LOW", "W-HIGH"}

    def test_recalibration_with_unknown_well_raises(self, monkeypatch, volume, calibration_repo):
        _patch_well_list(monkeypatch, GOOD_WELLS)
        with pytest.raises(Exception):
            ccs.fit_and_store_calibration(volume, well_ids=["NOPE"], calibration_repo=calibration_repo)
