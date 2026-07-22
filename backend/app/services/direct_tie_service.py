"""
services/direct_tie_service.py
-------------------------------
Direct nearest-trace well->seismic tie against THIS feature's single
active SEG-Y volume (app.services.seismic_processor.get_segy_volume) --
the same two-part algorithm tie_service.get_well_seismic_tie uses for the
Well-to-Seismic Tie page (a direct nearest-trace spatial search via
well_seismic_tie.find_nearest_trace_index, then a DPTM-based, full-
seismic-window frequency/polarity/bulk-shift correlation search via
well_seismic_tie.search_best_tie_full_window), just applied to the single
active volume instead of a named dataset.

Extracted as its own module so this (proven, on real field data, more
location-accurate than coordinate_calibration_service's calibrated fit --
see spectral_property_prediction_service.py's docstring for the numbers)
resolution lives in exactly one place, shared by every caller on the
single-active-volume side that wants it: currently
spectral_property_prediction_service.py and the Inline/Crossline Section
well-log overlay (section_well_log_service.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services.dashboard_upload_service import TIE_LOW_CONFIDENCE_THRESHOLD
from app.services.spectral_petro_correlation_service import _extract_curve
from app.services.tie_service import _load_config as _load_tie_config
from app.services import well_service

# Minimum valid DEPT/DPTM samples to trust a depth<->time mapping at all.
MIN_DEPTH_TIME_SAMPLES = 10


@dataclass
class DirectTieResult:
    """A well's direct nearest-trace tie, plus its DEPT<->DPTM mapping (the
    well's own time axis, before the bulk_shift_ms correction below is
    applied) -- everything a caller needs to place either a handful of
    seismic-time samples (spectral_property_prediction_service) or a whole
    log curve (section_well_log_service) onto real depth."""

    well_id: str
    trace_idx: int
    inline_number: int
    crossline_number: int
    distance_m: float
    correlation: float
    bulk_shift_ms: float
    best_freq_hz: float
    polarity: int
    boundary_pinned: bool
    low_confidence: bool
    depth_m: np.ndarray  # sorted, deduped, DEPT<->DPTM-valid depth samples
    dptm_ms: np.ndarray  # paired DPTM (well's own unshifted time axis), same length as depth_m


def resolve_direct_tie(volume, well_id: str) -> DirectTieResult:
    """Raises TieError/WellNotFoundError/SegyVolumeError on any failure --
    callers treat that as "excluded/unavailable", never silently proceed
    on a well that couldn't actually be tied."""
    config = _load_tie_config()
    max_radius_m = config.get("max_tie_search_radius_m")
    max_shift_ms = float(config.get("tie_search_max_shift_ms", wst.DEFAULT_TIE_SEARCH_MAX_SHIFT_MS))

    well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
    if well_summary.well_x is None or well_summary.well_y is None:
        raise wst.TieError(
            f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot locate it "
            "on the seismic survey."
        )
    trace_idx, distance_m = wst.find_nearest_trace_index(
        well_summary.well_x, well_summary.well_y, volume.source_x, volume.source_y, max_radius_m=max_radius_m
    )
    inline_number = int(volume.inline[trace_idx])
    crossline_number = int(volume.crossline[trace_idx])

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth = _extract_curve(rows, "DEPT")
    dt_log = _extract_curve(rows, "DT")
    rhob = _extract_curve(rows, "RHOB")
    dptm = _extract_curve(rows, "DPTM")

    t_rc, rc = wst.reflectivity_from_time_axis(dptm, dt_log, rhob, volume.sample_interval_ms)
    real_trace = volume.get_trace(trace_idx)
    tie = wst.search_best_tie_full_window(
        t_rc, rc, volume.twt_axis_ms, volume.sample_interval_ms, real_trace, max_shift_ms=max_shift_ms
    )
    boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - wst.BOUNDARY_PINNED_FRACTION) * max_shift_ms
    low_confidence = tie.correlation < TIE_LOW_CONFIDENCE_THRESHOLD or boundary_pinned

    # DEPT<->DPTM mapping -- independent of the DT/RHOB validity
    # reflectivity_from_time_axis required internally, per this codebase's
    # convention of each curve getting its own null mask (see
    # spectral_petro_correlation_service._property_series).
    valid = np.isfinite(depth) & np.isfinite(dptm)
    depth_v, dptm_v = depth[valid], dptm[valid]
    order = np.argsort(dptm_v)
    depth_v, dptm_v = depth_v[order], dptm_v[order]
    keep = np.concatenate([[True], np.diff(dptm_v) > 1e-6])
    depth_v, dptm_v = depth_v[keep], dptm_v[keep]
    if len(depth_v) < MIN_DEPTH_TIME_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}' has too few valid DEPT/DPTM samples ({len(depth_v)}) to build a "
            f"depth<->time mapping (need >= {MIN_DEPTH_TIME_SAMPLES})."
        )

    return DirectTieResult(
        well_id=well_id,
        trace_idx=trace_idx,
        inline_number=inline_number,
        crossline_number=crossline_number,
        distance_m=distance_m,
        correlation=tie.correlation,
        bulk_shift_ms=tie.bulk_shift_ms,
        best_freq_hz=tie.best_freq_hz,
        polarity=tie.polarity,
        boundary_pinned=boundary_pinned,
        low_confidence=low_confidence,
        depth_m=depth_v,
        dptm_ms=dptm_v,
    )
