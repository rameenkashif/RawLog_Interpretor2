"""
services/section_well_log_service.py
-------------------------------------
For the Inline/Crossline Section view: every well's VSH/PHIE/SWE log
curve, converted from depth to two-way time via direct_tie_service's
direct nearest-trace + DPTM full-window-search tie (the same
proven-accurate resolution spectral_property_prediction_service.py uses,
NOT coordinate_calibration_service's calibrated fit -- see that module's
docstring for why), ready to draw as a small "log-on-section" curve next
to the seismic section's own amplitude image.

Wells are placed on the section by their OWN tied inline/crossline,
regardless of whether it exactly equals the section's requested line
number -- a real seismic section is normally only ever exactly on a
handful of wells (if any), so this is a projection (every well shown at
its own position along the section's cross-axis), the same practical
compromise real interpretation tools make when "all wells" is requested
on a single 2D line. It is NOT claiming a well sits exactly on the
displayed line.
"""

from __future__ import annotations

import numpy as np

from app import well_seismic_tie as wst
from app.services import direct_tie_service as dts
from app.services import well_service
from app.services.spectral_petro_correlation_service import _extract_curve

PETRO_CURVE_LAS_NAMES = {"vsh": "VSH", "phie": "PHIE", "swe": "SWE"}


def _curve_at_depth(rows: list[dict], las_name: str, depth_all: np.ndarray, depth_m: np.ndarray) -> list[float | None]:
    """Interpolate one property curve onto depth_m (the well's DPTM-valid
    depth samples) against ITS OWN null mask -- same convention as
    spectral_petro_correlation_service._property_series, since a
    property's valid samples don't necessarily line up with DPTM's."""
    values = _extract_curve(rows, las_name)
    valid = np.isfinite(depth_all) & np.isfinite(values)
    if valid.sum() < 2:
        return [None] * len(depth_m)
    d, v = depth_all[valid], values[valid]
    interpolated = np.interp(depth_m, d, v, left=np.nan, right=np.nan)
    return [None if not np.isfinite(x) else float(x) for x in interpolated]


def get_section_well_logs(orientation: str, line_number: int) -> dict:
    """orientation: 'inline' or 'crossline' -- which section the frontend
    is currently showing (determines whether a well's position along the
    section's cross-axis is its own crossline or inline). line_number is
    only used to pick the seismic volume's own recorded time range each
    well's curve gets clipped to (every well is returned, positioned at
    its own tied location -- see module docstring)."""
    from app.services import seismic_processor as sp

    if orientation not in ("inline", "crossline"):
        raise ValueError(f"orientation must be 'inline' or 'crossline', got {orientation!r}.")

    volume = sp.get_segy_volume()
    # Touch the requested line so an out-of-range request fails the same
    # way the plain section endpoints do, even though its amplitude data
    # itself isn't used here.
    if orientation == "inline":
        volume.get_inline_section(line_number)
    else:
        volume.get_crossline_section(line_number)

    twt_min, twt_max = float(volume.twt_axis_ms[0]), float(volume.twt_axis_ms[-1])

    wells: list[dict] = []
    skipped: list[dict] = []
    for summary in well_service.list_well_summaries():
        well_id = summary.well_id
        try:
            result = dts.resolve_direct_tie(volume, well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            skipped.append({"well_id": well_id, "reason": str(exc)})
            continue
        if result.boundary_pinned or result.low_confidence:
            skipped.append({
                "well_id": well_id,
                "reason": f"Low-confidence tie (correlation={result.correlation:.3f}) -- not drawn.",
            })
            continue

        twt_full = result.dptm_ms + result.bulk_shift_ms
        in_range = (twt_full >= twt_min) & (twt_full <= twt_max)
        if not in_range.any():
            skipped.append({
                "well_id": well_id,
                "reason": "Logged interval falls outside the seismic survey's recorded time window.",
            })
            continue

        curves_response = well_service.get_well_curves(well_id)
        rows = curves_response["data"]
        depth_all = np.array(
            [row.get("DEPT") if row.get("DEPT") is not None else np.nan for row in rows], dtype=float
        )
        depth_in_range = result.depth_m[in_range]

        wells.append({
            "well_id": well_id,
            "position_on_axis": result.crossline_number if orientation == "inline" else result.inline_number,
            "correlation": result.correlation,
            "twt_ms": twt_full[in_range].tolist(),
            "vsh": _curve_at_depth(rows, PETRO_CURVE_LAS_NAMES["vsh"], depth_all, depth_in_range),
            "phie": _curve_at_depth(rows, PETRO_CURVE_LAS_NAMES["phie"], depth_all, depth_in_range),
            "swe": _curve_at_depth(rows, PETRO_CURVE_LAS_NAMES["swe"], depth_all, depth_in_range),
        })

    return {
        "orientation": orientation,
        "line_number": line_number,
        "wells": wells,
        "skipped_wells": skipped,
    }
