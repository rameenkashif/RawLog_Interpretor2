"""
services/spectral_petro_correlation_service.py
--------------------------------------------------
"CWT vs SWT -- Petrophysical Correlation": at a well tied to the seismic
volume, quantifies how strongly CWT (at a single fixed frequency) and SWT
(at a chosen dyadic level) spectral amplitude correlate with VSH/PHIE/SWE
over the well's logged interval, at a MATCHED frequency band -- CWT is
sampled at the SWT level's own band-center frequency, so the two methods
are compared like-for-like instead of CWT's adaptive frequency axis
against a fixed SWT level.

Pure orchestration, no new spectral or petrophysical computation lives
here: reuses coordinate_calibration_service (well->trace tie, the same
resolution used by the Well Tie / Synthetic Seismogram modules),
well_seismic_tie.depth_to_twt (the same sonic-integration depth-time
relationship SegyVolume.get_well_tie uses), well_service (VSH/PHIE/SWE
are already computed and stored per well by app/petrophysics.py at LAS
load time -- not recomputed here), and
seismic_processor.get_spectral_decomposition_trace (the exact CWT/SWT
computation the Spectral Decomposition tab uses).
"""

from __future__ import annotations

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
    freq_idx = int(np.argmin(np.abs(cwt_freqs - center_hz)))
    return [float(band_lo), float(band_hi)], float(cwt_freqs[freq_idx]), freq_idx


def _correlate_well(
    volume: sp.SegyVolume, well_id: str, swt_level: int, wavelet: str, cwt_freq_idx: int
) -> dict:
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

    # Two-step interpolation (seismic time -> depth -> property value), so
    # each property is looked up against its OWN null mask rather than
    # assuming VSH/PHIE/SWE share DT's valid samples.
    depth_at_time = np.interp(seismic_twt_sub, twt_ms_v, depth_v)

    def _property_at_time(curve_name: str) -> np.ndarray:
        values = _extract_curve(rows, _CURVE_LAS_NAMES[curve_name])
        valid_p = np.isfinite(depth) & np.isfinite(values)
        if valid_p.sum() < 2:
            return np.full_like(depth_at_time, np.nan)
        d, v = depth[valid_p], values[valid_p]
        return np.interp(depth_at_time, d, v, left=np.nan, right=np.nan)

    # SWT/CWT: the exact per-trace decomposition the Spectral Decomposition
    # tab uses, sliced at the requested level / matched frequency index.
    swt_result = volume.get_spectral_decomposition_trace(
        inline_number, crossline_number, method="swt", wavelet=wavelet
    )
    swt_amplitude = np.array(swt_result["energy"])[:, swt_level - 1][overlap]

    cwt_result = volume.get_spectral_decomposition_trace(inline_number, crossline_number, method="cwt")
    cwt_amplitude = np.array(cwt_result["energy"])[:, cwt_freq_idx][overlap]

    correlations: dict[str, dict] = {}
    for curve_name in PETRO_CURVES:
        prop_t = _property_at_time(curve_name)
        cwt_r, cwt_n = _pearson(cwt_amplitude, prop_t)
        swt_r, swt_n = _pearson(swt_amplitude, prop_t)
        correlations[curve_name] = {"cwt_r": cwt_r, "cwt_n": cwt_n, "swt_r": swt_r, "swt_n": swt_n}

    low_sample_warning = any(
        c["cwt_n"] < MIN_RELIABLE_SAMPLES or c["swt_n"] < MIN_RELIABLE_SAMPLES for c in correlations.values()
    )

    return {
        "well_id": well_id,
        "nearest_inline": inline_number,
        "nearest_crossline": crossline_number,
        "distance_m": distance_m,
        "tie_method": tie_method,
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
