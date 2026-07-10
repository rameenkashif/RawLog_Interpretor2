"""
coordinate_calibration_service.py
--------------------------------------
Orchestrates coordinate_calibration.py's per-axis linear fit against the
currently loaded wells + SEG-Y survey, persists it (coordinate_calibration_
repository.py) so it has a stable reference set of calibration wells
across requests, merges in manual tie-point overrides
(coordinate_tie_override_repository.py), and is the single place a well
should be located on the seismic survey through --
resolve_well_trace_index() -- rather than a direct
find_nearest_trace_index(well_x, well_y, source_x, source_y) call, since
well and seismic coordinates are NOT directly comparable (fix #4: both
are labeled "meters" but sit on different, unknown coordinate reference
systems).

Every function here takes the SegyVolume as an explicit parameter rather
than calling seismic_processor.get_segy_volume() itself -- seismic_
processor.py is a CALLER of this module (SegyVolume.get_well_tie() routes
through resolve_well_trace_index()), so importing it here too would be
circular. Callers that don't already have a volume handy (e.g. router
endpoints) fetch one from seismic_processor.get_segy_volume() themselves
and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.coordinate_calibration import (
    AxisCalibration,
    CoordinateCalibrationError,
    estimate_bin_spacing_m,
    fit_per_axis_calibration,
    validate_and_flag_wells,
)
from app.coordinate_calibration_repository import (
    CoordinateCalibrationRepository,
    get_coordinate_calibration_repository,
)
from app.coordinate_tie_override_repository import (
    CoordinateTieOverrideRepository,
    get_coordinate_tie_override_repository,
)
from app.services import well_service
from app.well_seismic_tie import find_nearest_trace_index

if TYPE_CHECKING:
    from app.services.seismic_processor import SegyVolume

BIN_TOLERANCE_MULTIPLE = 2.0


class UnresolvedCoordinateError(Exception):
    """Raised when a well's position on the seismic survey can't be
    trusted -- neither a manual override nor a validated calibrated tie is
    available. The fix is a manual tie-point override, not a better
    algorithmic guess; see coordinate_tie_override_repository.py."""


@dataclass
class WellCalibrationReport:
    well_id: str
    well_name: str
    well_x: float
    well_y: float
    transformed_x: float
    transformed_y: float
    nearest_inline: int
    nearest_crossline: int
    nearest_trace_distance_m: float
    is_extrapolated: bool
    within_bin_tolerance: bool
    trustworthy: bool
    used_in_calibration: bool
    has_manual_override: bool
    override_inline: int | None = None
    override_crossline: int | None = None


def _fit_calibration(volume: "SegyVolume", well_ids: list[str] | None) -> tuple[AxisCalibration, list[str], float]:
    """Fit a fresh calibration. well_ids=None means "use every currently
    known well with coordinates" (the bootstrap/default case); passing an
    explicit list restricts the fit to just those wells (e.g. a curated
    subset a user trusts, when explicitly recalibrating)."""
    all_summaries = {
        s.well_id: s for s in well_service.list_well_summaries() if s.well_x is not None and s.well_y is not None
    }
    if well_ids is None:
        chosen = list(all_summaries.values())
    else:
        missing = [w for w in well_ids if w not in all_summaries]
        if missing:
            raise CoordinateCalibrationError(
                f"Well(s) {missing} have no surface coordinates -- can't use them to calibrate."
            )
        chosen = [all_summaries[w] for w in well_ids]

    if len(chosen) < 2:
        raise CoordinateCalibrationError(
            f"Only {len(chosen)} well(s) available to calibrate -- need at least 2 with known "
            "surface coordinates."
        )

    well_x = np.array([s.well_x for s in chosen])
    well_y = np.array([s.well_y for s in chosen])
    seismic_x_range = (float(volume.source_x.min()), float(volume.source_x.max()))
    seismic_y_range = (float(volume.source_y.min()), float(volume.source_y.max()))
    cal = fit_per_axis_calibration(well_x, well_y, seismic_x_range, seismic_y_range)
    bin_spacing_m = estimate_bin_spacing_m(volume.source_x, volume.source_y)
    return cal, [s.well_id for s in chosen], bin_spacing_m


def fit_and_store_calibration(
    volume: "SegyVolume",
    well_ids: list[str] | None = None,
    calibration_repo: CoordinateCalibrationRepository | None = None,
) -> tuple[AxisCalibration, list[str], float]:
    """Explicitly (re)fit the calibration and persist it as the new
    baseline. well_ids=None fits from every well currently known to have
    coordinates -- pass an explicit subset to recalibrate from only a
    curated set of wells a user trusts (e.g. excluding one flagged bad)."""
    calibration_repo = calibration_repo or get_coordinate_calibration_repository()
    cal, used_well_ids, bin_spacing_m = _fit_calibration(volume, well_ids)
    calibration_repo.save(cal, used_well_ids, bin_spacing_m, volume.path.name)
    return cal, used_well_ids, bin_spacing_m


def _get_or_bootstrap_calibration(
    volume: "SegyVolume", calibration_repo: CoordinateCalibrationRepository
) -> tuple[AxisCalibration, list[str], float]:
    """Load the persisted calibration if one exists for the current SEG-Y
    file; otherwise fit-and-store one from whatever wells are currently
    known (first-time bootstrap) so there's always a stable baseline to
    validate against, rather than silently refitting (and thus
    re-including any bad well) on every single call."""
    stored = calibration_repo.load()
    if stored is not None:
        cal, well_ids, bin_spacing_m, segy_filename = stored
        if segy_filename == volume.path.name:
            return cal, well_ids, bin_spacing_m
        # Stored calibration was fit against a different SEG-Y file (e.g.
        # the volume was replaced) -- stale, re-bootstrap rather than
        # silently applying an unrelated survey's fit.
    return fit_and_store_calibration(volume, well_ids=None, calibration_repo=calibration_repo)


def get_calibration_report(
    volume: "SegyVolume",
    calibration_repo: CoordinateCalibrationRepository | None = None,
    override_repo: CoordinateTieOverrideRepository | None = None,
) -> list[WellCalibrationReport]:
    """Full diagnostic report for every well with known coordinates,
    validated against the STORED (stable) calibration baseline -- not a
    freshly refit one, so a well added after the baseline was established
    can genuinely be flagged as extrapolated relative to it. Shows the
    calibrated transform's estimate, residual-vs-bin-spacing validation,
    extrapolation flag, whether the well was part of the calibration
    baseline itself, and whether a manual override exists."""
    calibration_repo = calibration_repo or get_coordinate_calibration_repository()
    override_repo = override_repo or get_coordinate_tie_override_repository()
    cal, calibration_well_ids, bin_spacing_m = _get_or_bootstrap_calibration(volume, calibration_repo)

    summaries = [s for s in well_service.list_well_summaries() if s.well_x is not None and s.well_y is not None]
    well_ids = [s.well_id for s in summaries]
    well_x = np.array([s.well_x for s in summaries])
    well_y = np.array([s.well_y for s in summaries])
    results = validate_and_flag_wells(
        cal, well_ids, well_x, well_y, volume.source_x, volume.source_y, bin_spacing_m, BIN_TOLERANCE_MULTIPLE
    )

    reports = []
    for summary, result in zip(summaries, results):
        override = override_repo.get_override(summary.well_id)
        reports.append(
            WellCalibrationReport(
                well_id=summary.well_id,
                well_name=summary.well_name,
                well_x=result.well_x,
                well_y=result.well_y,
                transformed_x=result.transformed_x,
                transformed_y=result.transformed_y,
                nearest_inline=int(volume.inline[result.nearest_trace_index]),
                nearest_crossline=int(volume.crossline[result.nearest_trace_index]),
                nearest_trace_distance_m=result.nearest_trace_distance_m,
                is_extrapolated=result.is_extrapolated,
                within_bin_tolerance=result.within_bin_tolerance,
                trustworthy=result.trustworthy,
                used_in_calibration=summary.well_id in calibration_well_ids,
                has_manual_override=override is not None,
                override_inline=override.inline if override else None,
                override_crossline=override.crossline if override else None,
            )
        )
    return reports


def _trace_index_for_inline_crossline(volume: "SegyVolume", inline: int, crossline: int) -> int:
    geometry = volume.get_grid_geometry()
    inlines_sorted = geometry["inlines_sorted"]
    crosslines_sorted = geometry["crosslines_sorted"]
    grid_trace_idx = geometry["grid_trace_idx"]
    il_pos = int(np.searchsorted(inlines_sorted, inline))
    xl_pos = int(np.searchsorted(crosslines_sorted, crossline))
    valid_il = il_pos < len(inlines_sorted) and inlines_sorted[il_pos] == inline
    valid_xl = xl_pos < len(crosslines_sorted) and crosslines_sorted[xl_pos] == crossline
    if not (valid_il and valid_xl):
        raise UnresolvedCoordinateError(
            f"Manual override inline={inline}/crossline={crossline} does not correspond to any "
            "trace in the current seismic survey."
        )
    idx = int(grid_trace_idx[il_pos, xl_pos])
    if idx < 0:
        raise UnresolvedCoordinateError(
            f"Manual override inline={inline}/crossline={crossline} falls in a gap in the survey "
            "grid (no trace there)."
        )
    return idx


def resolve_well_trace_index(
    volume: "SegyVolume",
    well_id: str,
    calibration_repo: CoordinateCalibrationRepository | None = None,
    override_repo: CoordinateTieOverrideRepository | None = None,
) -> tuple[int, float | None, str]:
    """Resolve a well to a trace index for downstream tie/prediction
    workflows -- the single place that should happen, instead of a direct
    find_nearest_trace_index(well_x, well_y, source_x, source_y) call.

    Returns (trace_index, distance_m_or_None, method) where method is
    'manual_override' or 'calibrated_fit' (distance_m is None for an
    override -- there's no calibration residual to report, the mapping is
    asserted directly). Raises UnresolvedCoordinateError if neither a
    manual override nor a trustworthy calibrated tie is available -- the
    caller must not silently proceed on a flagged well.
    """
    calibration_repo = calibration_repo or get_coordinate_calibration_repository()
    override_repo = override_repo or get_coordinate_tie_override_repository()

    override = override_repo.get_override(well_id)
    if override is not None:
        idx = _trace_index_for_inline_crossline(volume, override.inline, override.crossline)
        return idx, None, "manual_override"

    # get_well_summary (a single-well lookup) rather than list_well_summaries
    # here -- this well's own existence/coordinates shouldn't depend on
    # every OTHER well being enumerable too (raises WellNotFoundError
    # naturally if well_id doesn't exist at all).
    summary = well_service.get_well_summary(well_id)
    if summary.well_x is None or summary.well_y is None:
        raise UnresolvedCoordinateError(
            f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot locate it on "
            "the seismic survey. Add a manual tie-point override if the correct trace is known."
        )

    try:
        cal, _calibration_well_ids, bin_spacing_m = _get_or_bootstrap_calibration(volume, calibration_repo)
    except CoordinateCalibrationError:
        # Fewer than 2 wells with known coordinates exist anywhere -- a
        # per-axis calibration genuinely can't be fit from a single point
        # (fix #4 requires at least 2 to determine a line's slope AND
        # intercept per axis). Fall back to the legacy generous-buffer
        # "same ballpark" sanity check + a direct nearest-trace search --
        # unvalidated against real trace density, so surfaced distinctly
        # as tie_method='direct_unvalidated' rather than silently
        # pretending it went through the calibrated+validated path.
        volume.check_crs_alignment(well_id, summary.well_x, summary.well_y)
        idx, distance_m = find_nearest_trace_index(summary.well_x, summary.well_y, volume.source_x, volume.source_y)
        return idx, distance_m, "direct_unvalidated"

    results = validate_and_flag_wells(
        cal,
        [well_id],
        np.array([summary.well_x]),
        np.array([summary.well_y]),
        volume.source_x,
        volume.source_y,
        bin_spacing_m,
        BIN_TOLERANCE_MULTIPLE,
    )
    result = results[0]
    if not result.trustworthy:
        reasons = []
        if result.is_extrapolated:
            reasons.append(
                "its coordinates fall outside both the range the calibration was fit from and the "
                "survey's own coordinate footprint (extrapolated)"
            )
        if not result.within_bin_tolerance:
            reasons.append(
                f"its nearest real trace is {result.nearest_trace_distance_m:.0f} m away, more than "
                f"{BIN_TOLERANCE_MULTIPLE:g}x the survey's ~{bin_spacing_m:.0f} m trace spacing"
            )
        raise UnresolvedCoordinateError(
            f"Well '{well_id}' cannot be reliably tied to the seismic survey via the calibrated "
            f"well<->seismic coordinate transform: {' and '.join(reasons)}. This well's coordinates "
            "are unresolved, not just imprecise -- add a manual tie-point override (a confirmed "
            "inline/crossline) before running a tie or synthetic seismogram on it."
        )

    return result.nearest_trace_index, result.nearest_trace_distance_m, "calibrated_fit"
