"""
services/synthetic_seismogram_service.py
-------------------------------------------
Orchestrates the synthetic seismogram / well-tie module: pulls a well's
unit-standardized header + curves, validates its coordinates against the
SEG-Y survey, estimates density (real RHOB, calibrated Gardner, or
rock-physics), builds acoustic impedance + reflectivity, generates/extracts
a wavelet, convolves to a synthetic trace, applies any saved manual
stretch/squeeze, finds the nearest real trace, and cross-correlates --
assembling everything the /api/synthetic/* router needs.

This is orchestration, not new computational logic: it reuses well_service
(LAS/curve loading), well_seismic_tie (impedance/reflectivity/wavelets/
convolution/correlation/QC/density), seismic_processor (SegyVolume:
geometry + real traces + CRS check), and synthetic_tie_repository
(persisted manual tie points) rather than duplicating any of them.
"""

from __future__ import annotations

import numpy as np

from app import well_seismic_tie as wst
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import well_service
from app.synthetic_tie_repository import (
    TiePoint,
    TiePointSet,
    get_synthetic_tie_repository,
)

VALID_DENSITY_METHODS = ("rhob", "gardner", "rock_physics")
VALID_WAVELET_METHODS = ("statistical", "ricker")

# No deviation survey exists in any of Z-02..Z-08's LAS files -- every well
# uses the vertical assumption (MD = TVD). Surfaced as a static badge in the
# API response and UI; flip this (and wire in a real MD->TVD correction) if
# a deviation survey is ever added for a well.
VERTICAL_ASSUMPTION_NOTE = (
    "Vertical assumption — no deviation survey available. MD = TVD for this well."
)
NO_CHECKSHOT_NOTE = (
    "Synthetic/derived time-depth curve (no checkshot/VSP available) -- built by "
    "integrating the sonic (DT) log, which only measures travel time within the "
    "logged interval, not absolute two-way time from the surface. Anchored to the "
    "seismic volume's own first sample time as a non-degenerate starting point "
    "(arbitrary, not physically derived) -- expect to need real stretch/squeeze "
    "correction via the manual tie points below before trusting this tie."
)


class SyntheticSeismogramError(Exception):
    """Base class for synthetic-seismogram-module errors."""


class MissingCurveError(SyntheticSeismogramError):
    def __init__(self, well_id: str, curve: str):
        self.well_id = well_id
        self.curve = curve
        super().__init__(
            f"Well '{well_id}' has no usable '{curve}' curve (all null/missing)."
        )


def _extract_curve(rows: list[dict], name: str) -> np.ndarray:
    arr = np.array(
        [row.get(name) if row.get(name) is not None else np.nan for row in rows], dtype=float
    )
    arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
    return arr


def _resolve_density(
    well_id: str,
    density_method: str,
    velocity_m_s: np.ndarray,
    rhob_real: np.ndarray,
    vsh: np.ndarray,
    phie: np.ndarray,
) -> tuple[np.ndarray, str, dict | None]:
    """Returns (density_g_cc, note, gardner_coefficients_or_None)."""
    if density_method == "rhob":
        if not np.isfinite(rhob_real).any():
            raise MissingCurveError(well_id, "RHOB")
        return rhob_real, "Using the well's real RHOB curve.", None

    if density_method == "gardner":
        valid = np.isfinite(velocity_m_s) & np.isfinite(rhob_real)
        if valid.sum() >= 20:
            a, b = wst.calibrate_gardner_coefficients(velocity_m_s[valid], rhob_real[valid])
            coeffs = {"a": a, "b": b, "calibrated": True}
            note = f"Gardner's equation with field-calibrated coefficients (a={a:.4f}, b={b:.4f})."
        else:
            a, b = 0.31, 0.25
            coeffs = {"a": a, "b": b, "calibrated": False}
            note = (
                "Gardner's equation with generic textbook coefficients (a=0.31, b=0.25) -- "
                "not enough real RHOB samples in this well to calibrate locally."
            )
        return wst.gardner_density(velocity_m_s, a, b), note, coeffs

    if density_method == "rock_physics":
        if not (np.isfinite(vsh).any() and np.isfinite(phie).any()):
            raise MissingCurveError(well_id, "VSH/PHIE (required for rock-physics density)")
        return (
            wst.rock_physics_density(vsh, phie),
            "Rock-physics density from VSH/PHIE (matrix/shale/fluid mixing model), not Gardner's equation.",
            None,
        )

    raise SyntheticSeismogramError(
        f"Unknown density_method '{density_method}' -- expected one of {VALID_DENSITY_METHODS}."
    )


def generate(
    well_id: str,
    wavelet_method: str = "statistical",
    wavelet_freq_hz: float = 25.0,
    density_method: str = "rhob",
    apply_saved_tie: bool = True,
) -> dict:
    if wavelet_method not in VALID_WAVELET_METHODS:
        raise SyntheticSeismogramError(
            f"Unknown wavelet_method '{wavelet_method}' -- expected one of {VALID_WAVELET_METHODS}."
        )

    volume = sp.get_segy_volume()
    well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent

    # Well location resolved via coordinate_calibration_service, NOT a
    # direct find_nearest_trace_index(well_x, well_y, source_x, source_y)
    # call -- well and seismic coordinates are on different, unknown
    # coordinate reference systems, so a raw distance comparison between
    # them is meaningless without the calibrated transform (or an
    # explicit manual override) -- see that module's docstring.
    trace_idx, distance_m, tie_method = ccs.resolve_well_trace_index(volume, well_id)

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth = _extract_curve(rows, "DEPT")
    dt_log = _extract_curve(rows, "DT")
    rhob_real = _extract_curve(rows, "RHOB")
    nphi = _extract_curve(rows, "NPHI")
    vsh = _extract_curve(rows, "VSH")
    phie = _extract_curve(rows, "PHIE")

    if not np.isfinite(dt_log).any():
        raise MissingCurveError(well_id, "DT")

    # DT is assumed us/ft (standard imperial sonic units), matching the rest
    # of this pipeline (well_seismic_tie.build_synthetic's default).
    velocity_m_s = wst.FT_PER_M * 1e6 / dt_log
    density, density_note, gardner_coeffs = _resolve_density(
        well_id, density_method, velocity_m_s, rhob_real, vsh, phie
    )

    washout_flags = wst.washout_qc_flag(
        nphi, rhob_real if np.isfinite(rhob_real).any() else density, dt_log
    )

    real_trace = volume.get_trace(trace_idx)
    dt_ms = volume.sample_interval_ms

    if wavelet_method == "statistical":
        wavelet_t_ms, wavelet = wst.extract_statistical_wavelet(real_trace, dt_ms=dt_ms)
    else:
        wavelet_t_s, wavelet = wst.ricker_wavelet(wavelet_freq_hz, dt_ms / 1000.0)
        wavelet_t_ms = wavelet_t_s * 1000.0  # ricker_wavelet's t axis is in seconds
    spectra = wst.wavelet_spectra(wavelet, dt_ms)

    valid = np.isfinite(depth) & np.isfinite(dt_log) & np.isfinite(density) & (dt_log > 0) & (density > 0)
    depth_v, dt_v, density_v = depth[valid], dt_log[valid], density[valid]
    if len(depth_v) < 10:
        raise SyntheticSeismogramError(
            "Too few valid DT/density samples after removing nulls/invalid values."
        )

    # Anchor the well's own (0-based) sonic-integrated TWT to the seismic
    # volume's own first sample time -- without this, the well's curve and
    # the seismic's real (non-zero-delay) time axis have no overlap at all,
    # and resampling later would silently produce an all-zero synthetic.
    # This is an arbitrary but non-degenerate default; manual stretch/
    # squeeze (applied below) refines it once a real tie is available.
    twt_ms = wst.depth_to_twt(depth_v, dt_v, dt_unit="us_per_ft", t0_ms=float(volume.twt_axis_ms[0]))
    ai = wst.acoustic_impedance(dt_v, density_v, dt_unit="us_per_ft")
    refl = wst.reflectivity_series(ai)
    refl_twt_ms = (twt_ms[1:] + twt_ms[:-1]) / 2.0
    refl_depth_m = (depth_v[1:] + depth_v[:-1]) / 2.0

    tie_points: list[TiePoint] = []
    if apply_saved_tie:
        saved = get_synthetic_tie_repository().get_tie_points(well_id)
        if saved:
            tie_points = saved.points
    if tie_points:
        refl_twt_ms = wst.apply_stretch_squeeze(
            refl_depth_m, refl_twt_ms, [(p.md_m, p.time_shift_ms) for p in tie_points]
        )

    reg_twt_ms = np.arange(refl_twt_ms[0], refl_twt_ms[-1], dt_ms)
    refl_reg = np.interp(reg_twt_ms, refl_twt_ms, refl)
    full_conv = np.convolve(refl_reg, wavelet, mode="full")
    start = (len(full_conv) - len(refl_reg)) // 2
    synthetic_reg = full_conv[start : start + len(refl_reg)]
    synthetic_on_seismic_axis = np.interp(
        volume.twt_axis_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0
    )

    tie = wst.cross_correlate_and_shift(synthetic_on_seismic_axis, real_trace, dt_ms)

    return {
        "well_id": well_id,
        "well_header": {
            "well_x": well_summary.well_x,
            "well_y": well_summary.well_y,
            "kb_m": well_summary.kb_m,
            "td_m": well_summary.td_m,
            "coordinate_unit_detected": well_summary.coordinate_unit_detected,
            "unit_conversion_applied": well_summary.unit_conversion_applied,
            "td_stop_ratio": well_summary.td_stop_ratio,
        },
        "vertical_assumption_note": VERTICAL_ASSUMPTION_NOTE,
        "time_depth_note": NO_CHECKSHOT_NOTE,
        "density_method": density_method,
        "density_note": density_note,
        "gardner_coefficients": gardner_coeffs,
        "nearest_inline": int(volume.inline[trace_idx]),
        "nearest_crossline": int(volume.crossline[trace_idx]),
        "distance_m": distance_m,
        "tie_method": tie_method,
        "depth_m": depth_v.tolist(),
        "twt_ms": twt_ms.tolist(),
        "acoustic_impedance": ai.tolist(),
        "reflectivity_depth_m": refl_depth_m.tolist(),
        "reflectivity": refl.tolist(),
        "reflectivity_twt_ms": refl_twt_ms.tolist(),
        "washout_depth_m": depth.tolist(),
        "washout_flag": washout_flags.tolist(),
        "wavelet_method": wavelet_method,
        "wavelet_freq_hz": wavelet_freq_hz,
        "wavelet_t_ms": np.asarray(wavelet_t_ms).tolist(),
        "wavelet_amplitude": np.asarray(wavelet).tolist(),
        "wavelet_spectrum_freq_hz": spectra["freq_hz"].tolist(),
        "wavelet_spectrum_amplitude": spectra["amplitude"].tolist(),
        "wavelet_spectrum_phase_deg": spectra["phase_deg"].tolist(),
        "seismic_twt_ms": volume.twt_axis_ms.tolist(),
        "synthetic": synthetic_on_seismic_axis.tolist(),
        "shifted_synthetic": tie["shifted_synthetic"].tolist(),
        "real_trace": real_trace.tolist(),
        "best_shift_ms": tie["best_shift_ms"],
        "correlation": tie["correlation"],
        "applied_tie_points": [{"md_m": p.md_m, "time_shift_ms": p.time_shift_ms} for p in tie_points],
    }


def save_tie_points(well_id: str, points: list[dict], wavelet_method: str, wavelet_freq_hz: float) -> TiePointSet:
    """Persist manual stretch/squeeze control points for a well so they
    survive across sessions instead of being recomputed from scratch."""
    volume = sp.get_segy_volume()
    tie = TiePointSet(
        well_id=well_id,
        points=[TiePoint(md_m=p["md_m"], time_shift_ms=p["time_shift_ms"]) for p in points],
        wavelet_method=wavelet_method,
        wavelet_freq_hz=wavelet_freq_hz,
        segy_filename=volume.path.name,
    )
    get_synthetic_tie_repository().save_tie_points(tie)
    return tie


def get_tie_points(well_id: str) -> TiePointSet | None:
    return get_synthetic_tie_repository().get_tie_points(well_id)


def delete_tie_points(well_id: str) -> bool:
    return get_synthetic_tie_repository().delete_tie_points(well_id)


def nearest_trace(well_id: str) -> dict:
    """Standalone nearest-trace lookup (without generating the full
    synthetic), for a lightweight "where does this well tie to" check."""
    volume = sp.get_segy_volume()
    well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
    trace_idx, distance_m, tie_method = ccs.resolve_well_trace_index(volume, well_id)
    return {
        "well_id": well_id,
        "trace_index": trace_idx,
        "inline": int(volume.inline[trace_idx]),
        "crossline": int(volume.crossline[trace_idx]),
        "distance_m": distance_m,
        "tie_method": tie_method,
    }
