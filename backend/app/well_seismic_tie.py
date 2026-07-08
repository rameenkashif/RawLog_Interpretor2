"""
well_seismic_tie.py
--------------------
Real well-to-seismic tie: converts sonic (DT) + density (RHOB) logs into a
synthetic seismogram via reflectivity * wavelet convolution, then correlates
it against the nearest real seismic trace.

This is NOT the amplitude-heuristic proxy in seismic_attributes.py — this is
the standard geophysical technique (Ricker wavelet synthetic + sonic-based
depth-time conversion), used because no checkshot survey is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate


class TieError(Exception):
    """Raised when a tie cannot be computed (missing curves, bad geometry, etc.)."""


FT_PER_M = 0.3048


@dataclass
class SyntheticResult:
    twt_ms: np.ndarray          # time axis of the synthetic, ms
    synthetic: np.ndarray       # synthetic amplitude values
    reflectivity_twt_ms: np.ndarray
    reflectivity: np.ndarray


def depth_to_twt(
    depth_m: np.ndarray,
    dt_log: np.ndarray,
    dt_unit: str = "us_per_ft",
) -> np.ndarray:
    """
    Integrate the sonic (DT) log to build a depth -> two-way-time relationship,
    since no checkshot survey is available for these wells.

    dt_unit: "us_per_ft" (standard imperial sonic units, most common even when
    depth curves are stored in meters) or "us_per_m".
    """
    if len(depth_m) < 2:
        raise TieError("Not enough depth samples to integrate sonic log.")

    depth_step_m = np.diff(depth_m)
    dt_mid = (dt_log[:-1] + dt_log[1:]) / 2.0  # µs per unit, midpoint of each interval

    if dt_unit == "us_per_ft":
        step_ft = depth_step_m / FT_PER_M
        one_way_us = dt_mid * step_ft
    elif dt_unit == "us_per_m":
        one_way_us = dt_mid * depth_step_m
    else:
        raise TieError(f"Unknown dt_unit: {dt_unit}")

    two_way_us = 2.0 * one_way_us
    cum_us = np.concatenate([[0.0], np.cumsum(two_way_us)])
    return cum_us / 1000.0  # -> ms


def acoustic_impedance(dt_log: np.ndarray, rhob: np.ndarray, dt_unit: str = "us_per_ft") -> np.ndarray:
    """AI = velocity * density. Velocity derived from DT (slowness)."""
    if dt_unit == "us_per_ft":
        velocity_m_s = FT_PER_M * 1e6 / dt_log  # µs/ft -> m/s
    elif dt_unit == "us_per_m":
        velocity_m_s = 1e6 / dt_log
    else:
        raise TieError(f"Unknown dt_unit: {dt_unit}")
    return velocity_m_s * rhob


def reflectivity_series(ai: np.ndarray) -> np.ndarray:
    """R[i] = (AI[i+1] - AI[i]) / (AI[i+1] + AI[i]); one shorter than input."""
    num = ai[1:] - ai[:-1]
    den = ai[1:] + ai[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den != 0, num / den, 0.0)
    return r


def ricker_wavelet(freq_hz: float, dt_s: float, length_s: float = 0.128) -> tuple[np.ndarray, np.ndarray]:
    """Standard zero-phase Ricker wavelet, used since no statistical wavelet
    extraction is implemented (would require a longer real seismic window
    plus spectral analysis — a documented future enhancement)."""
    t = np.arange(-length_s / 2, length_s / 2, dt_s)
    a = (np.pi * freq_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return t, w


def build_synthetic(
    depth_m: np.ndarray,
    dt_log: np.ndarray,
    rhob: np.ndarray,
    seismic_dt_ms: float,
    seismic_twt_axis_ms: np.ndarray,
    wavelet_freq_hz: float = 30.0,
    dt_unit: str = "us_per_ft",
) -> SyntheticResult:
    """Full pipeline: sonic integration -> impedance -> reflectivity ->
    convolve with Ricker wavelet -> resample onto the seismic's own time axis."""
    valid = np.isfinite(depth_m) & np.isfinite(dt_log) & np.isfinite(rhob) & (dt_log > 0) & (rhob > 0)
    depth_m, dt_log, rhob = depth_m[valid], dt_log[valid], rhob[valid]

    if len(depth_m) < 10:
        raise TieError("Too few valid DT/RHOB samples after removing nulls/invalid values.")

    twt_ms = depth_to_twt(depth_m, dt_log, dt_unit=dt_unit)
    ai = acoustic_impedance(dt_log, rhob, dt_unit=dt_unit)
    refl = reflectivity_series(ai)
    refl_twt_ms = (twt_ms[1:] + twt_ms[:-1]) / 2.0  # midpoints, matches refl length

    # Resample reflectivity onto a regular grid at the seismic's sample rate
    # so convolution with the wavelet is meaningful.
    reg_twt_ms = np.arange(refl_twt_ms[0], refl_twt_ms[-1], seismic_dt_ms)
    refl_reg = np.interp(reg_twt_ms, refl_twt_ms, refl)

    _, wavelet = ricker_wavelet(wavelet_freq_hz, seismic_dt_ms / 1000.0)
    full_conv = np.convolve(refl_reg, wavelet, mode="full")
    # np.convolve's "same" mode returns max(len(a), len(v)) rather than
    # always len(a) -- for short well intervals the wavelet can be longer
    # than the reflectivity series, breaking that assumption. Crop the
    # full convolution back to refl_reg's length explicitly instead.
    start = (len(full_conv) - len(refl_reg)) // 2
    synthetic_reg = full_conv[start : start + len(refl_reg)]
    # Resample onto the actual seismic trace's time axis for direct comparison.
    synthetic_on_seismic_axis = np.interp(
        seismic_twt_axis_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0
    )

    return SyntheticResult(
        twt_ms=seismic_twt_axis_ms,
        synthetic=synthetic_on_seismic_axis,
        reflectivity_twt_ms=refl_twt_ms,
        reflectivity=refl,
    )


def find_nearest_trace_index(
    well_x: float,
    well_y: float,
    trace_x: np.ndarray,
    trace_y: np.ndarray,
    max_radius_m: float | None = None,
) -> tuple[int, float]:
    """Nearest-neighbor search by Euclidean distance. Returns (index, distance_m).
    Assumes well and trace coordinates share the same CRS/units -- this is NOT
    verified here; a mismatch will silently return a wrong-but-plausible answer,
    so callers should sanity check the returned distance against survey extent."""
    d = np.sqrt((trace_x - well_x) ** 2 + (trace_y - well_y) ** 2)
    idx = int(np.argmin(d))
    dist = float(d[idx])
    if max_radius_m is not None and dist > max_radius_m:
        raise TieError(
            f"Nearest trace is {dist:.0f} m away, outside max_tie_search_radius_m={max_radius_m}."
        )
    return idx, dist


def cross_correlate_and_shift(synthetic: np.ndarray, real: np.ndarray, dt_ms: float) -> dict:
    """Cross-correlates synthetic vs. real trace, finds the best-fit lag,
    and reports Pearson correlation at that alignment."""
    synthetic = np.nan_to_num(synthetic)
    real = np.nan_to_num(real)

    syn_n = (synthetic - synthetic.mean()) / (synthetic.std() + 1e-12)
    real_n = (real - real.mean()) / (real.std() + 1e-12)

    xcorr = correlate(real_n, syn_n, mode="full")
    lags = np.arange(-len(syn_n) + 1, len(real_n))
    best_lag = int(lags[np.argmax(xcorr)])
    best_shift_ms = best_lag * dt_ms

    shifted_syn = np.roll(synthetic, best_lag)
    if best_lag > 0:
        shifted_syn[:best_lag] = 0
    elif best_lag < 0:
        shifted_syn[best_lag:] = 0

    valid = (shifted_syn != 0) & np.isfinite(shifted_syn) & np.isfinite(real)
    if valid.sum() < 5:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(shifted_syn[valid], real[valid])[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    return {
        "best_shift_ms": best_shift_ms,
        "correlation": correlation,
        "shifted_synthetic": shifted_syn,
    }