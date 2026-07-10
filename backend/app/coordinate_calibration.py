"""
coordinate_calibration.py
-----------------------------
Well header X/Y and SEG-Y SourceX/SourceY can both be labeled "meters"
while NOT sharing a coordinate reference system -- confirmed for this
field's real data (Zamzama Z-02-Z-08) by a mismatch ratio that differs on
X (~1.22x) vs Y (~0.44x), which rules out a simple feet/meters conversion
(a uniform scale error would show the same ratio on both axes). There is
no known CRS/EPSG for either dataset, so a real reprojection isn't
possible.

This module fits an independent PER-AXIS linear transform as a working
default:
    X_seismic = a * X_well + b
    Y_seismic = c * Y_well + d
calibrated from the wells' own coordinate extent mapped onto the seismic
survey's coordinate extent (i.e. a 2-point fit per axis, pinned by
whichever wells sit at each axis's min/max). This is deliberately a weak,
2-free-parameter-per-axis model -- see validate_and_flag_wells() below,
which exists specifically because this fit can look perfect for wells
inside its own calibration range while being silently wrong outside it.
That happened for a real 8th well in this field: its transformed position
landed 1.9 km outside the survey footprint despite passing every other QC
check, because the fit was extrapolating far beyond the 2 wells that
calibrated it.

Nothing here should be trusted blindly: always call validate_and_flag_wells,
treat a flagged well as unresolved (don't run downstream tie/prediction on
it without explicit confirmation), and prefer a manual tie-point override
(see coordinate_tie_override_repository.py) over this fit whenever one is
available for a well -- no algorithm can recover a true correspondence
from genuinely ambiguous coordinate data alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CoordinateCalibrationError(Exception):
    pass


@dataclass
class AxisCalibration:
    a: float
    b: float
    c: float
    d: float
    well_x_range: tuple[float, float]
    well_y_range: tuple[float, float]
    seismic_x_range: tuple[float, float]
    seismic_y_range: tuple[float, float]
    n_wells_used: int


def fit_per_axis_calibration(
    well_x: np.ndarray,
    well_y: np.ndarray,
    seismic_x_range: tuple[float, float],
    seismic_y_range: tuple[float, float],
) -> AxisCalibration:
    """Fit X_seismic = a*X_well + b and Y_seismic = c*Y_well + d from the
    WELLS' own coordinate extent (min/max of well_x, well_y across all
    wells with known coordinates) mapped onto the SURVEY's coordinate
    extent. This assumes the wells' relative spread along each axis
    correlates with the survey's own extent along that axis -- a coarse,
    unverified working assumption (there is no known CRS for either
    dataset), which is exactly why every caller must also run
    validate_and_flag_wells() before trusting this."""
    well_x = np.asarray(well_x, dtype=float)
    well_y = np.asarray(well_y, dtype=float)
    valid = np.isfinite(well_x) & np.isfinite(well_y)
    well_x, well_y = well_x[valid], well_y[valid]

    if len(well_x) < 2:
        raise CoordinateCalibrationError(
            f"Need at least 2 wells with known coordinates to fit a calibration, got {len(well_x)}."
        )

    well_x_min, well_x_max = float(well_x.min()), float(well_x.max())
    well_y_min, well_y_max = float(well_y.min()), float(well_y.max())
    if well_x_max == well_x_min or well_y_max == well_y_min:
        raise CoordinateCalibrationError(
            "All calibration wells share the same X or Y coordinate -- can't fit a linear scale "
            "on that axis (need spatial spread across at least 2 distinct values)."
        )

    sx_min, sx_max = seismic_x_range
    sy_min, sy_max = seismic_y_range

    a = (sx_max - sx_min) / (well_x_max - well_x_min)
    b = sx_min - a * well_x_min
    c = (sy_max - sy_min) / (well_y_max - well_y_min)
    d = sy_min - c * well_y_min

    return AxisCalibration(
        a=a,
        b=b,
        c=c,
        d=d,
        well_x_range=(well_x_min, well_x_max),
        well_y_range=(well_y_min, well_y_max),
        seismic_x_range=(sx_min, sx_max),
        seismic_y_range=(sy_min, sy_max),
        n_wells_used=len(well_x),
    )


def apply_calibration(cal: AxisCalibration, well_x: float, well_y: float) -> tuple[float, float]:
    return cal.a * well_x + cal.b, cal.c * well_y + cal.d


def estimate_bin_spacing_m(trace_x: np.ndarray, trace_y: np.ndarray, sample_size: int = 200) -> float:
    """Rough estimate of the survey's trace spacing (bin size) in real
    coordinate units, from the median nearest-neighbor distance among a
    random sample of traces. Used as the tolerance UNIT for
    validate_and_flag_wells: a good tie should land within ~1-2 bin
    spacings of a real trace, which means something concrete for every
    survey, unlike an arbitrary fixed-meters threshold."""
    n = len(trace_x)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    nearest_dists = []
    for i in sample_idx:
        d = np.sqrt((trace_x - trace_x[i]) ** 2 + (trace_y - trace_y[i]) ** 2)
        d[i] = np.inf
        nearest_dists.append(float(d.min()))
    return float(np.median(nearest_dists)) if nearest_dists else 1.0


@dataclass
class WellCalibrationResult:
    well_id: str
    well_x: float
    well_y: float
    transformed_x: float
    transformed_y: float
    nearest_trace_index: int
    nearest_trace_distance_m: float
    is_extrapolated: bool
    within_bin_tolerance: bool

    @property
    def trustworthy(self) -> bool:
        """Both independent checks must pass before this well's calibrated
        tie is used automatically -- either one failing means "unresolved",
        requiring a manual tie-point override before downstream tie/
        prediction workflows run on this well."""
        return self.within_bin_tolerance and not self.is_extrapolated


def _nearest_trace(x: float, y: float, trace_x: np.ndarray, trace_y: np.ndarray) -> tuple[int, float]:
    d = np.sqrt((trace_x - x) ** 2 + (trace_y - y) ** 2)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def validate_and_flag_wells(
    cal: AxisCalibration,
    well_ids: list[str],
    well_x: np.ndarray,
    well_y: np.ndarray,
    trace_x: np.ndarray,
    trace_y: np.ndarray,
    bin_spacing_m: float,
    bin_tolerance_multiple: float = 2.0,
) -> list[WellCalibrationResult]:
    """For every well: apply the calibration, find its nearest real trace,
    and flag it two independent ways:

    - within_bin_tolerance: residual distance to the nearest trace is
      within bin_tolerance_multiple * bin_spacing_m. A calibration that's
      only "perfect" for its own 2 calibration wells but wildly off
      elsewhere shows up here as False for the other wells.
    - is_extrapolated: the well's RAW (untransformed) coordinate falls
      outside the coordinate range that was actually used to fit the
      calibration, AND ALSO its transformed position falls outside the
      survey's own real coordinate extent. Both conditions together mean
      the fit is being asked to extrapolate beyond its evidence AND the
      result doesn't even land somewhere plausible -- don't run
      downstream tie/prediction workflows on a well flagged either way
      without an explicit manual override.
    """
    results: list[WellCalibrationResult] = []
    for well_id, wx, wy in zip(well_ids, well_x, well_y):
        tx, ty = apply_calibration(cal, wx, wy)
        idx, dist = _nearest_trace(tx, ty, trace_x, trace_y)

        within_calibration_range = (
            cal.well_x_range[0] <= wx <= cal.well_x_range[1]
            and cal.well_y_range[0] <= wy <= cal.well_y_range[1]
        )
        within_survey_extent = (
            cal.seismic_x_range[0] <= tx <= cal.seismic_x_range[1]
            and cal.seismic_y_range[0] <= ty <= cal.seismic_y_range[1]
        )
        is_extrapolated = not within_calibration_range and not within_survey_extent
        within_bin_tolerance = dist <= bin_tolerance_multiple * bin_spacing_m

        results.append(
            WellCalibrationResult(
                well_id=well_id,
                well_x=float(wx),
                well_y=float(wy),
                transformed_x=float(tx),
                transformed_y=float(ty),
                nearest_trace_index=idx,
                nearest_trace_distance_m=dist,
                is_extrapolated=is_extrapolated,
                within_bin_tolerance=within_bin_tolerance,
            )
        )
    return results
