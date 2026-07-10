"""
well_zone_tie_service.py
----------------------------
"Well-Seismic Tie" map: ties each well's reservoir zone (mean VSH within
its Pay zone, per petrophysics.py's ZONES classification) to the seismic
survey via real-world coordinates (well X/Y -> nearest trace, same
find_nearest_trace_index used by the single-well synthetic seismogram
tie), then spatially interpolates those per-well values across the full
inline x crossline grid using inverse-distance weighting (IDW) so the
handful of wells can be viewed as a continuous map with well locations
overlaid -- e.g. a "Predicted VSH" style map.

IMPORTANT: with only a handful of wells, this is a purely geometric
interpolation between known control points (nearest-well-wins-nearby,
blended by distance) -- NOT a seismic inversion or ML prediction. It is
attribute-blind (never looks at seismic amplitude) and is only reliable
close to well control; treat structure between wells as a smooth guess,
not a resolved geological feature. This caveat is also returned in the
response's method_note so the frontend can surface it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import petrophysics as pp
from app.coordinate_calibration import CoordinateCalibrationError
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import well_service

DEFAULT_IDW_POWER = 2.0
MIN_WELLS_FOR_INTERPOLATION = 2
# Distance floor (m) so IDW weights don't blow up to infinity for a grid
# cell that lands (near) exactly on a well's own coordinate.
_MIN_DISTANCE_M = 1.0


class WellZoneTieError(Exception):
    pass


@dataclass
class _WellTiePoint:
    well_id: str
    well_name: str
    well_x: float
    well_y: float
    inline: int
    crossline: int
    distance_m: float
    mean_vsh_pay: float
    n_pay_samples: int


def _collect_tie_points(volume: sp.SegyVolume) -> tuple[list[_WellTiePoint], list[str]]:
    points: list[_WellTiePoint] = []
    warnings: list[str] = []

    for summary in well_service.list_well_summaries():
        well_id = summary.well_id
        if summary.well_x is None or summary.well_y is None:
            warnings.append(f"{well_id}: no surface coordinates in LAS header -- skipped.")
            continue
        try:
            trace_idx, distance_m, _tie_method = ccs.resolve_well_trace_index(volume, well_id)
        except (ccs.UnresolvedCoordinateError, CoordinateCalibrationError, sp.CrsMismatchError) as exc:
            warnings.append(f"{well_id}: {exc}")
            continue

        _, df = well_service.get_well_df(well_id)
        if "ZONES" not in df.columns or "VSH" not in df.columns:
            warnings.append(f"{well_id}: missing ZONES/VSH curves -- skipped.")
            continue
        pay_vsh = df.loc[df["ZONES"] == pp.ZONE_PAY, "VSH"].dropna()
        if pay_vsh.empty:
            warnings.append(f"{well_id}: no Pay-zone samples -- skipped.")
            continue

        distance_m = distance_m if distance_m is not None else 0.0
        points.append(
            _WellTiePoint(
                well_id=well_id,
                well_name=summary.well_name,
                well_x=summary.well_x,
                well_y=summary.well_y,
                inline=int(volume.inline[trace_idx]),
                crossline=int(volume.crossline[trace_idx]),
                distance_m=distance_m,
                mean_vsh_pay=float(pay_vsh.mean()),
                n_pay_samples=int(len(pay_vsh)),
            )
        )

    return points, warnings


def _idw_interpolate(volume: sp.SegyVolume, points: list[_WellTiePoint], power: float) -> np.ndarray:
    """Inverse-distance-weighted interpolation of each well's mean_vsh_pay
    across the full (n_inlines x n_crosslines) grid, using each grid
    cell's REAL trace coordinates (not grid index position) so the
    result respects actual survey geometry/bin spacing."""
    geometry = volume.get_grid_geometry()
    grid_trace_idx = geometry["grid_trace_idx"]
    predicted = np.full(grid_trace_idx.shape, np.nan, dtype=float)
    valid = grid_trace_idx >= 0
    trace_idx_valid = grid_trace_idx[valid]
    grid_x = volume.source_x[trace_idx_valid]
    grid_y = volume.source_y[trace_idx_valid]

    well_x = np.array([p.well_x for p in points])
    well_y = np.array([p.well_y for p in points])
    well_val = np.array([p.mean_vsh_pay for p in points])

    dx = grid_x[:, None] - well_x[None, :]
    dy = grid_y[:, None] - well_y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, _MIN_DISTANCE_M)
    weights = 1.0 / (dist**power)
    values = (weights @ well_val) / weights.sum(axis=1)

    predicted[valid] = values
    return predicted


def compute_well_zone_tie_map(power: float = DEFAULT_IDW_POWER) -> dict:
    volume = sp.get_segy_volume()
    points, warnings = _collect_tie_points(volume)

    if len(points) < MIN_WELLS_FOR_INTERPOLATION:
        detail = "; ".join(warnings) if warnings else "no wells have usable coordinates/zones."
        raise WellZoneTieError(
            f"Only {len(points)} well(s) could be tied to the seismic survey -- need at least "
            f"{MIN_WELLS_FOR_INTERPOLATION} for spatial interpolation. {detail}"
        )

    predicted = _idw_interpolate(volume, points, power)
    geometry = volume.get_grid_geometry()

    return {
        "inline_axis": geometry["inlines_sorted"].tolist(),
        "crossline_axis": geometry["crosslines_sorted"].tolist(),
        "predicted_vsh": predicted.tolist(),
        "wells": [
            {
                "well_id": p.well_id,
                "well_name": p.well_name,
                "inline": p.inline,
                "crossline": p.crossline,
                "distance_m": p.distance_m,
                "mean_vsh_pay": p.mean_vsh_pay,
                "n_pay_samples": p.n_pay_samples,
            }
            for p in points
        ],
        "warnings": warnings,
        "method_note": (
            f"Predicted VSH is inverse-distance-weighted (power={power:g}) spatial "
            "interpolation of each well's mean VSH within its Pay zone (ZONES==Pay), tied to "
            "the survey via real well/trace coordinates -- NOT a seismic inversion or ML "
            "prediction. It never looks at seismic amplitude, so it will not reproduce "
            "fault/channel-scale texture between wells; treat structure away from well "
            "control as a smooth geometric guess, not a resolved feature."
        ),
    }
