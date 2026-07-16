"""
services/spectral_petro_correlation_service.py
--------------------------------------------------
"CWT vs SWT" and "CWT vs SSWT" Petrophysical Correlation: at a well tied to
the seismic volume, quantifies how strongly two spectral methods'
amplitude correlates with VSH/PHIE/SWE over the well's logged interval, at
a MATCHED frequency -- so the two methods are compared like-for-like
instead of one method's adaptive frequency axis against the other's fixed
parameterization:

- CWT vs SWT: CWT is sampled at the SWT level's own dyadic band-center
  frequency (SWT has no continuous frequency axis, only a handful of
  octave bands -- see get_correlation).
- CWT vs SSWT: both have a continuous frequency axis (SSWT's is just much
  finer-grained), so both are independently snapped to the nearest bin to
  a single user-requested frequency -- see get_sswt_correlation.

Pure orchestration, no new spectral or petrophysical computation lives
here: reuses coordinate_calibration_service (well->trace tie, the same
resolution used by the Well Tie / Synthetic Seismogram modules),
well_seismic_tie.depth_to_twt (the same sonic-integration depth-time
relationship SegyVolume.get_well_tie uses), well_service (VSH/PHIE/SWE
are already computed and stored per well by app/petrophysics.py at LAS
load time -- not recomputed here), and
seismic_processor.get_spectral_decomposition_trace (the exact CWT/SWT/SSWT
computation the Spectral Decomposition tab uses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import well_service

# Same "not enough real samples to trust a fit" convention already used
# elsewhere in this codebase (Gardner coefficient calibration, MAD
# despiking) -- reused here as the threshold for the low_sample_warning
# flag, not a hard failure (a short tie interval is still worth showing,
# just flagged rather than hidden per the feature spec).
MIN_RELIABLE_SAMPLES = 20
# Minimum valid DT samples to trust a sonic-integrated depth-time curve at
# all -- matches synthetic_seismogram_service's build_synthetic guard.
MIN_DEPTH_TIME_SAMPLES = 10
# Default frequency for the CWT-vs-SSWT comparison, matching the default
# already used by the plain CWT/STFT frequency slider in SpectralDecompView.
DEFAULT_SSWT_COMPARISON_FREQUENCY_HZ = 30.0

PETRO_CURVES = ("vsh", "phie", "swe")
_CURVE_LAS_NAMES = {"vsh": "VSH", "phie": "PHIE", "swe": "SWE"}


def _extract_curve(rows: list[dict], name: str) -> np.ndarray:
    arr = np.array(
        [row.get(name) if row.get(name) is not None else np.nan for row in rows], dtype=float
    )
    arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
    return arr


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """Pearson r over the finite overlap of x and y, or (None, n) if there
    aren't enough points or either series is constant (undefined r) --
    same np.corrcoef + isfinite-guard convention as
    well_seismic_tie.cross_correlate_and_shift's own correlation."""
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 2:
        return None, n
    xv, yv = x[valid], y[valid]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return None, n
    r = float(np.corrcoef(xv, yv)[0, 1])
    if not np.isfinite(r):
        return None, n
    return r, n


def _nearest_freq_match(freqs: np.ndarray, target_hz: float) -> tuple[float, int]:
    """Nearest available frequency bin (+ its index) to target_hz -- same
    snapping convention get_spectral_decomposition_inline's frequency_hz
    fast path already uses."""
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(freqs[idx]), idx


@dataclass
class _WellTieContext:
    """Everything both correlation paths (CWT-vs-SWT, CWT-vs-SSWT) need
    about a well's tie that doesn't depend on which spectral methods are
    being compared -- extracted so neither path duplicates tie resolution,
    the sonic-integration depth-time relationship, or the seismic/logged
    time-window overlap."""

    well_id: str
    inline_number: int
    crossline_number: int
    distance_m: float | None
    tie_method: str
    rows: list[dict]
    depth: np.ndarray
    depth_at_time: np.ndarray
    overlap: np.ndarray  # boolean mask into volume.twt_axis_ms


def _resolve_well_tie_context(volume: sp.SegyVolume, well_id: str) -> _WellTieContext:
    well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
    # Well location resolved the same way as every other tie in this app
    # (Well Tie, Synthetic Seismogram) -- NOT a raw distance comparison,
    # see coordinate_calibration_service's docstring for why.
    trace_idx, distance_m, tie_method = ccs.resolve_well_trace_index(volume, well_id)
    inline_number = int(volume.inline[trace_idx])
    crossline_number = int(volume.crossline[trace_idx])

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth = _extract_curve(rows, "DEPT")
    dt_log = _extract_curve(rows, "DT")

    if not np.isfinite(dt_log).any():
        raise sp.MissingCurveError(well_id, "DT")

    valid_dt = np.isfinite(depth) & np.isfinite(dt_log) & (dt_log > 0)
    depth_v, dt_v = depth[valid_dt], dt_log[valid_dt]
    if len(depth_v) < MIN_DEPTH_TIME_SAMPLES:
        raise sp.SegyVolumeError(
            f"Well '{well_id}' has too few valid DT samples ({len(depth_v)}) to build a "
            f"depth-time relationship (need >= {MIN_DEPTH_TIME_SAMPLES})."
        )

    # Same sonic-integration depth-time relationship as the Well Tie module
    # (SegyVolume.get_well_tie) and Synthetic Seismogram module, anchored
    # at the seismic survey's own first sample time -- see
    # well_seismic_tie.depth_to_twt.
    t0_ms = float(volume.twt_axis_ms[0])
    twt_ms_v = wst.depth_to_twt(depth_v, dt_v, dt_unit="us_per_ft", t0_ms=t0_ms)

    # Overlap between the seismic's own recorded time window and the
    # logged interval's sonic-integrated time coverage -- only this
    # window has both a real trace and a depth-time relationship to place
    # the logs on, per the feature spec.
    seismic_twt = volume.twt_axis_ms
    overlap = (seismic_twt >= twt_ms_v[0]) & (seismic_twt <= twt_ms_v[-1])
    if not overlap.any():
        raise sp.SegyVolumeError(
            f"Well '{well_id}'s logged interval does not overlap the seismic survey's "
            "recorded time window -- no samples to correlate."
        )
    seismic_twt_sub = seismic_twt[overlap]

    # Seismic time -> depth, via the depth-time curve above -- the first
    # step of the two-step interpolation _property_series finishes (depth
    # -> property value, against that property's OWN null mask).
    depth_at_time = np.interp(seismic_twt_sub, twt_ms_v, depth_v)

    return _WellTieContext(
        well_id=well_id,
        inline_number=inline_number,
        crossline_number=crossline_number,
        distance_m=distance_m,
        tie_method=tie_method,
        rows=rows,
        depth=depth,
        depth_at_time=depth_at_time,
        overlap=overlap,
    )


def _property_series(ctx: _WellTieContext, curve_name: str) -> np.ndarray:
    """Two-step interpolation (seismic time -> depth, done once in
    _resolve_well_tie_context -> depth -> property value here), so each
    property is looked up against its OWN null mask rather than assuming
    VSH/PHIE/SWE share DT's valid samples."""
    values = _extract_curve(ctx.rows, _CURVE_LAS_NAMES[curve_name])
    valid_p = np.isfinite(ctx.depth) & np.isfinite(values)
    if valid_p.sum() < 2:
        return np.full_like(ctx.depth_at_time, np.nan)
    d, v = ctx.depth[valid_p], values[valid_p]
    return np.interp(ctx.depth_at_time, d, v, left=np.nan, right=np.nan)


# ---- CWT vs SWT ---------------------------------------------------------


def _band_and_cwt_match(volume: sp.SegyVolume, swt_level: int, wavelet: str) -> tuple[list[float], float, int]:
    """SWT level's dyadic band and its matched CWT frequency (+ index into
    the CWT freq axis) -- survey-level constants (depend only on the
    sample interval / Nyquist and the fixed CWT frequency list, not on any
    particular well), computed once from an arbitrary valid trace via the
    same per-trace decomposition endpoint the Spectral Decomposition tab
    uses, so callers never need to reach into SegyVolume's internals."""
    inline0 = int(volume.inline[0])
    crossline0 = int(volume.crossline[0])
    swt_probe = volume.get_spectral_decomposition_trace(inline0, crossline0, method="swt", wavelet=wavelet)
    band_lo, band_hi = swt_probe["bands_hz"][swt_level - 1]
    cwt_probe = volume.get_spectral_decomposition_trace(inline0, crossline0, method="cwt")
    cwt_freqs = np.array(cwt_probe["freq_hz"])
    center_hz = (band_lo + band_hi) / 2.0
    cwt_freq_hz, freq_idx = _nearest_freq_match(cwt_freqs, center_hz)
    return [float(band_lo), float(band_hi)], cwt_freq_hz, freq_idx


def _correlate_well(
    volume: sp.SegyVolume, well_id: str, swt_level: int, wavelet: str, cwt_freq_idx: int
) -> dict:
    ctx = _resolve_well_tie_context(volume, well_id)

    # SWT/CWT: the exact per-trace decomposition the Spectral Decomposition
    # tab uses, sliced at the requested level / matched frequency index.
    swt_result = volume.get_spectral_decomposition_trace(
        ctx.inline_number, ctx.crossline_number, method="swt", wavelet=wavelet
    )
    swt_amplitude = np.array(swt_result["energy"])[:, swt_level - 1][ctx.overlap]

    cwt_result = volume.get_spectral_decomposition_trace(ctx.inline_number, ctx.crossline_number, method="cwt")
    cwt_amplitude = np.array(cwt_result["energy"])[:, cwt_freq_idx][ctx.overlap]

    correlations: dict[str, dict] = {}
    for curve_name in PETRO_CURVES:
        prop_t = _property_series(ctx, curve_name)
        cwt_r, cwt_n = _pearson(cwt_amplitude, prop_t)
        swt_r, swt_n = _pearson(swt_amplitude, prop_t)
        correlations[curve_name] = {"cwt_r": cwt_r, "cwt_n": cwt_n, "swt_r": swt_r, "swt_n": swt_n}

    low_sample_warning = any(
        c["cwt_n"] < MIN_RELIABLE_SAMPLES or c["swt_n"] < MIN_RELIABLE_SAMPLES for c in correlations.values()
    )

    return {
        "well_id": well_id,
        "nearest_inline": ctx.inline_number,
        "nearest_crossline": ctx.crossline_number,
        "distance_m": ctx.distance_m,
        "tie_method": ctx.tie_method,
        "low_sample_warning": low_sample_warning,
        **correlations,
    }


def get_correlation(
    well_id: str | None,
    all_wells: bool = False,
    swt_level: int = sp.SWT_DEFAULT_LEVEL,
    wavelet: str = sp.SWT_DEFAULT_WAVELET,
) -> dict:
    swt_level = sp._validate_swt_level(swt_level)  # noqa: SLF001 -- reusing the existing validator, not duplicating it
    wavelet = sp._validate_swt_wavelet(wavelet)  # noqa: SLF001

    volume = sp.get_segy_volume()
    band_hz, cwt_freq_hz, cwt_freq_idx = _band_and_cwt_match(volume, swt_level, wavelet)

    if not all_wells:
        if not well_id:
            raise sp.SegyVolumeError("well_id is required unless all_wells=true.")
        result = _correlate_well(volume, well_id, swt_level, wavelet, cwt_freq_idx)
        return {
            "mode": "single",
            "swt_level": swt_level,
            "swt_band_hz": band_hz,
            "cwt_frequency_hz": cwt_freq_hz,
            "wavelet": wavelet,
            "wells": [result],
            "skipped_well_ids": [],
            "averages": None,
        }

    well_results: list[dict] = []
    skipped: list[str] = []
    for summary in well_service.list_well_summaries():
        try:
            well_results.append(_correlate_well(volume, summary.well_id, swt_level, wavelet, cwt_freq_idx))
        except (ccs.UnresolvedCoordinateError, sp.SegyVolumeError):
            skipped.append(summary.well_id)

    averages = None
    if well_results:
        averages = {}
        for curve_name in PETRO_CURVES:
            cwt_rs = [r[curve_name]["cwt_r"] for r in well_results if r[curve_name]["cwt_r"] is not None]
            swt_rs = [r[curve_name]["swt_r"] for r in well_results if r[curve_name]["swt_r"] is not None]
            averages[curve_name] = {
                "cwt_r": float(np.mean(cwt_rs)) if cwt_rs else None,
                "swt_r": float(np.mean(swt_rs)) if swt_rs else None,
                "n_wells": max(len(cwt_rs), len(swt_rs)),
            }

    return {
        "mode": "all_wells",
        "swt_level": swt_level,
        "swt_band_hz": band_hz,
        "cwt_frequency_hz": cwt_freq_hz,
        "wavelet": wavelet,
        "wells": well_results,
        "skipped_well_ids": skipped,
        "averages": averages,
    }


# ---- CWT vs SSWT ---------------------------------------------------------


def _cwt_sswt_freq_match(volume: sp.SegyVolume, frequency_hz: float) -> tuple[float, int, float, int, float]:
    """Both CWT and SSWT are independently snapped to their own nearest
    available bin to the SAME requested frequency -- unlike CWT-vs-SWT,
    SSWT has its own continuous (just much finer-grained) frequency axis
    rather than a handful of octave bands, so there's no "band center" to
    derive; the user names a frequency directly. Computed once from an
    arbitrary valid trace (include_sswt=True), same pattern as
    _band_and_cwt_match, so callers never repeat this per well. Also
    validates frequency_hz is within the survey's Nyquist."""
    inline0 = int(volume.inline[0])
    crossline0 = int(volume.crossline[0])
    probe = volume.get_spectral_decomposition_trace(inline0, crossline0, method="cwt", include_sswt=True)
    nyquist_hz = float(probe["nyquist_hz"])
    if not (0.0 <= frequency_hz <= nyquist_hz):
        raise sp.SegyVolumeError(
            f"frequency_hz={frequency_hz:g} out of range -- expected 0-{nyquist_hz:g} (this survey's Nyquist)."
        )

    cwt_freq_hz, cwt_idx = _nearest_freq_match(np.array(probe["freq_hz"]), frequency_hz)
    sswt_freq_hz, sswt_idx = _nearest_freq_match(np.array(probe["sswt_freq_hz"]), frequency_hz)
    return cwt_freq_hz, cwt_idx, sswt_freq_hz, sswt_idx, nyquist_hz


def _correlate_well_sswt(
    volume: sp.SegyVolume, well_id: str, cwt_freq_idx: int, sswt_freq_idx: int
) -> dict:
    ctx = _resolve_well_tie_context(volume, well_id)

    # Single call returns BOTH the existing CWT and the SSWT amplitude for
    # this trace (include_sswt=True is additive, see
    # seismic_processor.get_spectral_decomposition_trace) -- no separate
    # SSWT-only request needed.
    result = volume.get_spectral_decomposition_trace(
        ctx.inline_number, ctx.crossline_number, method="cwt", include_sswt=True
    )
    cwt_amplitude = np.array(result["energy"])[:, cwt_freq_idx][ctx.overlap]
    sswt_amplitude = np.array(result["sswt_amplitude"])[:, sswt_freq_idx][ctx.overlap]

    correlations: dict[str, dict] = {}
    for curve_name in PETRO_CURVES:
        prop_t = _property_series(ctx, curve_name)
        cwt_r, cwt_n = _pearson(cwt_amplitude, prop_t)
        sswt_r, sswt_n = _pearson(sswt_amplitude, prop_t)
        correlations[curve_name] = {"cwt_r": cwt_r, "cwt_n": cwt_n, "sswt_r": sswt_r, "sswt_n": sswt_n}

    low_sample_warning = any(
        c["cwt_n"] < MIN_RELIABLE_SAMPLES or c["sswt_n"] < MIN_RELIABLE_SAMPLES for c in correlations.values()
    )

    return {
        "well_id": well_id,
        "nearest_inline": ctx.inline_number,
        "nearest_crossline": ctx.crossline_number,
        "distance_m": ctx.distance_m,
        "tie_method": ctx.tie_method,
        "low_sample_warning": low_sample_warning,
        **correlations,
    }


def get_sswt_correlation(
    well_id: str | None,
    all_wells: bool = False,
    frequency_hz: float = DEFAULT_SSWT_COMPARISON_FREQUENCY_HZ,
) -> dict:
    volume = sp.get_segy_volume()
    cwt_freq_hz, cwt_freq_idx, sswt_freq_hz, sswt_freq_idx, nyquist_hz = _cwt_sswt_freq_match(
        volume, frequency_hz
    )

    if not all_wells:
        if not well_id:
            raise sp.SegyVolumeError("well_id is required unless all_wells=true.")
        result = _correlate_well_sswt(volume, well_id, cwt_freq_idx, sswt_freq_idx)
        return {
            "mode": "single",
            "requested_frequency_hz": frequency_hz,
            "cwt_frequency_hz": cwt_freq_hz,
            "sswt_frequency_hz": sswt_freq_hz,
            "nyquist_hz": nyquist_hz,
            "wells": [result],
            "skipped_well_ids": [],
            "averages": None,
        }

    well_results: list[dict] = []
    skipped: list[str] = []
    for summary in well_service.list_well_summaries():
        try:
            well_results.append(_correlate_well_sswt(volume, summary.well_id, cwt_freq_idx, sswt_freq_idx))
        except (ccs.UnresolvedCoordinateError, sp.SegyVolumeError):
            skipped.append(summary.well_id)

    averages = None
    if well_results:
        averages = {}
        for curve_name in PETRO_CURVES:
            cwt_rs = [r[curve_name]["cwt_r"] for r in well_results if r[curve_name]["cwt_r"] is not None]
            sswt_rs = [r[curve_name]["sswt_r"] for r in well_results if r[curve_name]["sswt_r"] is not None]
            averages[curve_name] = {
                "cwt_r": float(np.mean(cwt_rs)) if cwt_rs else None,
                "sswt_r": float(np.mean(sswt_rs)) if sswt_rs else None,
                "n_wells": max(len(cwt_rs), len(sswt_rs)),
            }

    return {
        "mode": "all_wells",
        "requested_frequency_hz": frequency_hz,
        "cwt_frequency_hz": cwt_freq_hz,
        "sswt_frequency_hz": sswt_freq_hz,
        "nyquist_hz": nyquist_hz,
        "wells": well_results,
        "skipped_well_ids": skipped,
        "averages": averages,
    }
