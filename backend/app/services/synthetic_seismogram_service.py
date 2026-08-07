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
from app.services import seismic_processor as sp
from app.services import well_service
from app.services.tie_service import _load_config as _load_tie_config
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
    max_shift_ms: float = wst.DEFAULT_MAX_SHIFT_MS,
    auto_optimize_tie: bool = False,
) -> dict:
    if wavelet_method not in VALID_WAVELET_METHODS:
        raise SyntheticSeismogramError(
            f"Unknown wavelet_method '{wavelet_method}' -- expected one of {VALID_WAVELET_METHODS}."
        )

    volume = sp.get_segy_volume()
    well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent

    # Raw nearest-trace distance (well_seismic_tie.find_nearest_trace_index),
    # the same trace resolution the main Seismic page's Well-to-Seismic Tie
    # uses (tie_service.py / direct_tie_service.py) -- not coordinate_
    # calibration_service's calibrated fit this page used to rely on.
    # direct_tie_service.py's own docstring: raw nearest-trace is proven
    # more location-accurate on this survey's real field data than the
    # calibrated fit is.
    tie_config = _load_tie_config()
    max_radius_m = tie_config.get("max_tie_search_radius_m")
    if well_summary.well_x is None or well_summary.well_y is None:
        raise wst.TieError(
            f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot locate it "
            "on the seismic survey."
        )
    trace_idx, distance_m = wst.find_nearest_trace_index(
        well_summary.well_x, well_summary.well_y, volume.source_x, volume.source_y, max_radius_m=max_radius_m
    )
    tie_method = "nearest_trace"

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth = _extract_curve(rows, "DEPT")
    dt_log = _extract_curve(rows, "DT")
    rhob_real = _extract_curve(rows, "RHOB")
    nphi = _extract_curve(rows, "NPHI")
    vsh = _extract_curve(rows, "VSH")
    phie = _extract_curve(rows, "PHIE")
    dptm = _extract_curve(rows, "DPTM")

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

    # Datum check is an independent sanity-check of the SURVEY's own
    # recording delay (does it plausibly correspond to "surface to the top
    # of the logged interval"?), not of which depth-time curve we end up
    # tying with below -- always computed, regardless of which of the two
    # paths that is (see dashboard_upload_service.py, which gates well
    # eligibility on datum_check.plausible unconditionally).
    t0_ms = float(volume.twt_axis_ms[0])
    datum_check = wst.cross_check_delay_datum(delay_ms=t0_ms, logged_top_depth_m=float(depth_v[0]))

    # Depth-time: prefer the well's own DPTM curve (vendor-precomputed when
    # the LAS carries one, else petrophysics.compute_dptm's sonic-
    # integration fallback -- the SAME curve and preference order the main
    # Seismic page's Well-to-Seismic Tie uses, see tie_service.py), an
    # absolute two-way-time reference, rather than this page's own
    # from-scratch sonic integration arbitrarily anchored to the seismic
    # volume's first sample time (kept below as a fallback for the rare
    # case DPTM itself has too few valid samples for this well -- shouldn't
    # normally happen, since compute_dptm produces something whenever DT
    # does, but keeps this page working rather than hard-failing if it ever
    # doesn't).
    dptm_valid = valid & np.isfinite(dptm)
    depth_dptm_v, dt_dptm_v, density_dptm_v, dptm_v = depth[dptm_valid], dt_log[dptm_valid], density[dptm_valid], dptm[dptm_valid]
    if len(depth_dptm_v) >= 10:
        order = np.argsort(dptm_v)
        depth_v, dt_v, density_v, dptm_v = (
            depth_dptm_v[order], dt_dptm_v[order], density_dptm_v[order], dptm_v[order]
        )
        keep = np.concatenate([[True], np.diff(dptm_v) > 1e-6])
        depth_v, dt_v, density_v, dptm_v = depth_v[keep], dt_v[keep], density_v[keep], dptm_v[keep]
        twt_ms = dptm_v
        time_depth_note = (
            "Depth-time relationship uses the well's own DPTM curve (vendor-precomputed when the "
            "LAS carries one, else a sonic-integration approximation -- see petrophysics.compute_dptm), "
            "the same source and preference order the main Seismic page's Well-to-Seismic Tie uses -- "
            "not re-derived from scratch on this page."
        )
    else:
        twt_ms = wst.depth_to_twt(depth_v, dt_v, dt_unit="us_per_ft", t0_ms=t0_ms)
        time_depth_note = NO_CHECKSHOT_NOTE

    ai = wst.acoustic_impedance(dt_v, density_v, dt_unit="us_per_ft")
    refl = wst.reflectivity_series(ai)
    # Reflectivity is sparse (near-zero between reflectors) -- despike
    # with a RMS-floored MAD threshold, not a naive one, so genuine
    # outliers are removed without collapsing the real signal (fix #10).
    refl = wst.despike_mad(refl)
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

    polarity = 1
    tie_search_note: str | None = None
    if auto_optimize_tie and wavelet_method == "ricker":
        # Same search as the main Seismic page's Well-to-Seismic Tie
        # (well_seismic_tie.search_best_tie_full_window): Ricker frequency
        # is jointly searched along with polarity and shift, scanning the
        # ENTIRE seismic window rather than a local cross-correlation lag
        # search around a rough starting position -- see that function's
        # own docstring for why (without a checkshot, the well's time axis
        # can plausibly sit anywhere in a wide window, even using a real
        # DPTM curve).
        search = wst.search_best_tie_full_window(
            reg_twt_ms, refl_reg, volume.twt_axis_ms, dt_ms, real_trace, max_shift_ms=max_shift_ms
        )
        polarity = search.polarity
        wavelet_freq_hz = search.best_freq_hz  # report the winning frequency, not the requested one
        _, wavelet = wst.ricker_wavelet(wavelet_freq_hz, dt_ms / 1000.0)
        wavelet = polarity * wavelet  # keep wavelet_amplitude/spectra consistent with the winning polarity
        spectra = wst.wavelet_spectra(wavelet, dt_ms)

        # search_best_tie_full_window only returns arrays covering the
        # well's own reflectivity interval (a QC plot's window), not the
        # whole seismic trace this chart displays -- rebuild the
        # full-seismic-axis synthetic (unshifted and shifted) the same way
        # the non-auto-optimize path below does, using the winning wavelet.
        full_conv = np.convolve(refl_reg, wavelet, mode="full")
        start = (len(full_conv) - len(refl_reg)) // 2
        synthetic_reg = full_conv[start : start + len(refl_reg)]
        synthetic_on_seismic_axis = np.interp(volume.twt_axis_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0)
        shifted_synthetic = np.interp(
            volume.twt_axis_ms, reg_twt_ms + search.bulk_shift_ms, synthetic_reg, left=0.0, right=0.0
        )
        boundary_pinned = abs(search.bulk_shift_ms) >= (1.0 - wst.BOUNDARY_PINNED_FRACTION) * max_shift_ms
        tie = {
            "shifted_synthetic": shifted_synthetic,
            "best_shift_ms": search.bulk_shift_ms,
            "correlation": search.correlation,
            "max_shift_ms": max_shift_ms,
            "boundary_pinned": boundary_pinned,
        }
        tie_search_note = (
            f"Auto-optimized (same search as the main Seismic page's Well-to-Seismic Tie): best fit is "
            f"{wavelet_freq_hz:g} Hz Ricker at {'reversed' if polarity < 0 else 'normal'} polarity "
            f"(r={search.correlation:.3f}, within +/-{max_shift_ms:g}ms)."
        )
    elif auto_optimize_tie:  # wavelet_method == "statistical" -- the full-window search is Ricker-only
        search = wst.search_best_tie(
            refl_reg,
            reg_twt_ms,
            volume.twt_axis_ms,
            dt_ms,
            real_trace,
            candidate_freqs_hz=None,
            fixed_wavelet=wavelet,
            search_polarity=True,
            max_shift_ms=max_shift_ms,
        )
        polarity = search.polarity
        wavelet = polarity * wavelet  # keep wavelet_amplitude/spectra consistent with the winning polarity
        spectra = wst.wavelet_spectra(wavelet, dt_ms)
        synthetic_on_seismic_axis = search.synthetic
        tie = {
            "shifted_synthetic": search.shifted_synthetic,
            "best_shift_ms": search.best_shift_ms,
            "correlation": search.correlation,
            "max_shift_ms": search.max_shift_ms,
            "boundary_pinned": search.boundary_pinned,
        }
        tie_search_note = (
            f"Auto-optimized: searched {search.n_candidates_tried} (position, polarity) combinations "
            f"within +/-{max_shift_ms:g}ms -- best fit is the extracted statistical wavelet at "
            f"{'reversed' if polarity < 0 else 'normal'} polarity (r={search.correlation:.3f})."
        )
    else:
        full_conv = np.convolve(refl_reg, wavelet, mode="full")
        start = (len(full_conv) - len(refl_reg)) // 2
        synthetic_reg = full_conv[start : start + len(refl_reg)]
        synthetic_on_seismic_axis = np.interp(
            volume.twt_axis_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0
        )
        tie = wst.cross_correlate_and_shift(synthetic_on_seismic_axis, real_trace, dt_ms, max_shift_ms=max_shift_ms)

    # Frequency-domain view of the same real-vs-synthetic comparison shown
    # in the time-domain overlay -- reuses wavelet_spectra (a generic FFT
    # amplitude/phase helper, not wavelet-specific) rather than duplicating
    # the FFT logic. Both traces share seismic_twt_ms/dt_ms, so one freq
    # axis covers both.
    real_trace_spectrum = wst.wavelet_spectra(real_trace, dt_ms)
    synthetic_spectrum = wst.wavelet_spectra(tie["shifted_synthetic"], dt_ms)

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
        "time_depth_note": time_depth_note,
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
        "trace_spectrum_freq_hz": real_trace_spectrum["freq_hz"].tolist(),
        "real_trace_spectrum_amplitude": real_trace_spectrum["amplitude"].tolist(),
        "synthetic_spectrum_amplitude": synthetic_spectrum["amplitude"].tolist(),
        "best_shift_ms": tie["best_shift_ms"],
        "correlation": tie["correlation"],
        "max_shift_ms": tie["max_shift_ms"],
        "boundary_pinned": tie["boundary_pinned"],
        "polarity": polarity,
        "auto_optimize_tie": auto_optimize_tie,
        "tie_search_note": tie_search_note,
        "datum_check": {
            "delay_ms": datum_check.delay_ms,
            "implied_depth_m": datum_check.implied_depth_m,
            "logged_top_depth_m": datum_check.logged_top_depth_m,
            "relative_error": datum_check.relative_error,
            "avg_velocity_m_s": datum_check.avg_velocity_m_s,
            "plausible": datum_check.plausible,
        },
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
    synthetic), for a lightweight "where does this well tie to" check.
    Same raw nearest-trace resolution as generate() -- see its comment."""
    volume = sp.get_segy_volume()
    well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
    if well_summary.well_x is None or well_summary.well_y is None:
        raise wst.TieError(
            f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot locate it "
            "on the seismic survey."
        )
    max_radius_m = _load_tie_config().get("max_tie_search_radius_m")
    trace_idx, distance_m = wst.find_nearest_trace_index(
        well_summary.well_x, well_summary.well_y, volume.source_x, volume.source_y, max_radius_m=max_radius_m
    )
    return {
        "well_id": well_id,
        "trace_index": trace_idx,
        "inline": int(volume.inline[trace_idx]),
        "crossline": int(volume.crossline[trace_idx]),
        "distance_m": distance_m,
        "tie_method": "nearest_trace",
    }
