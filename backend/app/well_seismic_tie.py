"""
well_seismic_tie.py
--------------------
Real well-to-seismic tie: converts sonic (DT) + density (RHOB) logs into a
synthetic seismogram via reflectivity * wavelet convolution, then correlates
it against the nearest real seismic trace.

This is NOT the amplitude-heuristic proxy in seismic_attributes.py — this is
the standard geophysical technique (Ricker wavelet synthetic + sonic-based
depth-time conversion), used because no checkshot survey is available.

Also provides the supporting calculations for the synthetic-seismogram
module: density estimation when RHOB is unavailable (Gardner's equation,
locally calibrated, plus a rock-physics alternative), a soft washout/hole-
quality QC proxy (no CALI curve is available), and wavelet extraction
(statistical, from a real trace, alongside the Ricker generator above).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
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
    datum_check: "DatumCheckResult | None" = None


def depth_to_twt(
    depth_m: np.ndarray,
    dt_log: np.ndarray,
    dt_unit: str = "us_per_ft",
    t0_ms: float = 0.0,
) -> np.ndarray:
    """
    Integrate the sonic (DT) log to build a depth -> two-way-time relationship,
    since no checkshot survey is available for these wells.

    dt_unit: "us_per_ft" (standard imperial sonic units, most common even when
    depth curves are stored in meters) or "us_per_m".

    t0_ms: starting two-way time, added to the whole cumulative curve.
    IMPORTANT: sonic integration only measures travel time *within the
    logged interval* -- it has no way to know the two-way time from the
    surface down to the top of the log (that's exactly what a checkshot
    provides, and none exists here), so the curve returned with t0_ms=0
    always starts at 0 ms. A real seismic survey's recorded time axis
    almost never starts at 0 ms (it starts at some recording delay, e.g.
    2000+ ms for a deep target) -- resampling a 0-anchored synthetic onto
    that axis has NO overlap and silently produces an all-zero synthetic.
    Callers tying against a real seismic volume should pass the volume's
    own first sample time as a sane (arbitrary, but non-degenerate) default
    anchor, refinable via manual stretch/squeeze (see apply_stretch_squeeze).
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
    return t0_ms + cum_us / 1000.0  # -> ms


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
    """Standard zero-phase Ricker wavelet -- a fallback/comparison option
    alongside extract_statistical_wavelet() below when a real seismic trace
    at the well location is available to extract from."""
    t = np.arange(-length_s / 2, length_s / 2, dt_s)
    a = (np.pi * freq_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return t, w


def extract_statistical_wavelet(
    trace: np.ndarray, dt_ms: float, length_ms: float = 128.0
) -> tuple[np.ndarray, np.ndarray]:
    """Statistical (zero-phase) wavelet extraction from a real seismic trace
    segment: take the trace's own average amplitude spectrum, assume zero
    phase, and inverse-transform -- the standard "statistical" approach (as
    opposed to deterministic extraction, which needs a trusted reflectivity
    series to solve for phase too). A reasonable default alongside the
    Ricker generator above when a real trace near the well is available.

    Returns (t_ms, wavelet) -- t_ms centered at zero, same convention as
    ricker_wavelet(), wavelet normalized to unit peak amplitude.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    if len(trace) < 8:
        raise TieError("Trace too short for statistical wavelet extraction (need >= 8 samples).")

    spectrum = np.fft.rfft(trace * np.hanning(len(trace)))
    amplitude = np.abs(spectrum)
    # Zero-phase: inverse-transform the amplitude spectrum alone (phase
    # discarded), then center it at zero lag like ricker_wavelet() does.
    wavelet_full = np.fft.fftshift(np.fft.irfft(amplitude, n=len(trace)))

    n_samples = max(3, int(round(length_ms / dt_ms)))
    center = len(wavelet_full) // 2
    half = n_samples // 2
    wavelet = wavelet_full[max(0, center - half) : center + half + 1]

    peak = np.max(np.abs(wavelet))
    if peak > 0:
        wavelet = wavelet / peak

    t_ms = (np.arange(len(wavelet)) - len(wavelet) // 2) * dt_ms
    return t_ms, wavelet


def wavelet_spectra(wavelet: np.ndarray, dt_ms: float) -> dict:
    """Amplitude and (unwrapped, degrees) phase spectrum of a wavelet, so a
    geophysicist can QC an extracted or Ricker wavelet's phase behavior
    before trusting the synthetic it produces."""
    n = len(wavelet)
    spectrum = np.fft.rfft(wavelet)
    freq_hz = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    amplitude = np.abs(spectrum)
    phase_deg = np.degrees(np.unwrap(np.angle(spectrum)))
    return {"freq_hz": freq_hz, "amplitude": amplitude, "phase_deg": phase_deg}


# ---- Density estimation (when RHOB is unavailable) --------------------------
def gardner_density(velocity_m_s: np.ndarray, a: float = 0.31, b: float = 0.25) -> np.ndarray:
    """Gardner's equation: rho = a * V^b (metric form -- V in m/s, rho in
    g/cc). Fallback density estimate for a well with no RHOB curve. Defaults
    (a=0.31, b=0.25) are generic textbook constants -- prefer
    calibrate_gardner_coefficients() against a well with real RHOB in the
    same field instead of these where possible."""
    return a * np.power(velocity_m_s, b)


def calibrate_gardner_coefficients(
    velocity_m_s: np.ndarray, rhob: np.ndarray, a0: float = 0.31, b0: float = 0.25
) -> tuple[float, float]:
    """Fit Gardner's a, b coefficients to a well's own real velocity (from
    DT) vs. RHOB via scipy.optimize.curve_fit, so a future density-less well
    in the same field uses field-calibrated coefficients rather than generic
    textbook constants. Requires at least 20 valid samples; raises TieError
    if there isn't enough data or the fit doesn't converge."""
    velocity_m_s = np.asarray(velocity_m_s, dtype=float)
    rhob = np.asarray(rhob, dtype=float)
    valid = np.isfinite(velocity_m_s) & np.isfinite(rhob) & (velocity_m_s > 0) & (rhob > 0)
    velocity_m_s, rhob = velocity_m_s[valid], rhob[valid]
    if valid.sum() < 20:
        raise TieError(
            f"Only {int(valid.sum())} valid velocity/RHOB sample pairs -- need at least 20 "
            "to calibrate Gardner's coefficients."
        )
    try:
        (a, b), _ = curve_fit(gardner_density, velocity_m_s, rhob, p0=[a0, b0], maxfev=5000)
    except RuntimeError as exc:
        raise TieError(f"Gardner coefficient calibration did not converge: {exc}") from exc
    return float(a), float(b)


def rock_physics_density(
    vsh: np.ndarray,
    phie: np.ndarray,
    rho_matrix: float = 2.65,
    rho_shale: float = 2.75,
    rho_fluid: float = 1.0,
) -> np.ndarray:
    """Alternative density estimate from existing VSH/PHIE outputs, as a
    comparison against Gardner's equation or real RHOB: blends a sand-matrix
    and shale-matrix density by VSH (Vsh-derived mineralogy), then dilutes
    by pore fluid via PHIE --
        rho = (rho_matrix*(1-VSH) + rho_shale*VSH) * (1-PHIE) + rho_fluid*PHIE
    """
    vsh = np.asarray(vsh, dtype=float)
    phie = np.asarray(phie, dtype=float)
    rho_matrix_eff = rho_matrix * (1.0 - vsh) + rho_shale * vsh
    return rho_matrix_eff * (1.0 - phie) + rho_fluid * phie


# ---- Washout / hole-quality QC proxy (no CALI available) --------------------
def washout_qc_flag(
    nphi: np.ndarray,
    rhob: np.ndarray,
    dt_log: np.ndarray,
    rhob_matrix: float = 2.65,
    rhob_fluid: float = 1.0,
    crossover_threshold: float = 0.15,
    dt_zscore_threshold: float = 3.0,
    dt_rolling_window: int = 21,
) -> np.ndarray:
    """Soft QC proxy flagging depth intervals as "possible washout /
    unreliable interval" -- NOT a real caliper substitute (no CALI curve is
    available in these wells), just a heuristic. Flags a depth if EITHER:

    - NPHI-RHOB crossover: density porosity ((rhob_matrix-RHOB)/(rhob_matrix
      -rhob_fluid)) disagrees with NPHI by more than crossover_threshold --
      an enlarged/washed-out hole typically makes the neutron tool read
      spuriously high porosity while the density tool (sensitive to
      standoff) reads erratically.
    - DT spikes: DT deviates from a local rolling median by more than
      dt_zscore_threshold rolling standard deviations -- washouts often show
      up as erratic sonic cycle-skipping.

    Returns a boolean array, same length as the inputs.
    """
    nphi = np.asarray(nphi, dtype=float)
    rhob = np.asarray(rhob, dtype=float)
    dt_log = np.asarray(dt_log, dtype=float)

    phid = (rhob_matrix - rhob) / (rhob_matrix - rhob_fluid)
    with np.errstate(invalid="ignore"):
        crossover_flag = np.abs(phid - nphi) > crossover_threshold

    window = dt_rolling_window if dt_rolling_window % 2 == 1 else dt_rolling_window + 1
    if len(dt_log) < window:
        dt_spike_flag = np.zeros(len(dt_log), dtype=bool)
    else:
        half = window // 2
        padded = np.pad(dt_log, (half, half), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        with np.errstate(invalid="ignore"):
            rolling_median = np.nanmedian(windows, axis=1)
            rolling_std = np.nanstd(windows, axis=1)
        # Floor the rolling std at a fraction of the whole trace's std --
        # edge-replication padding (and any genuinely flat sub-interval)
        # can otherwise make a locally near-zero std amplify a tiny, benign
        # deviation into a huge z-score. Also never flag the first/last
        # half-window of samples: their rolling window is built partly from
        # padded (not real) values, so a "spike" there is at least as
        # likely to be a padding artifact as a real washout.
        global_std = np.nanstd(dt_log)
        std_floor = 0.1 * global_std if global_std > 0 else 1e-6
        rolling_std = np.maximum(rolling_std, std_floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            dt_zscore = np.abs(dt_log - rolling_median) / rolling_std
        dt_spike_flag = np.nan_to_num(dt_zscore, nan=0.0) > dt_zscore_threshold
        dt_spike_flag[:half] = False
        dt_spike_flag[len(dt_spike_flag) - half :] = False

    flag = crossover_flag | dt_spike_flag
    valid = np.isfinite(nphi) & np.isfinite(rhob) & np.isfinite(dt_log)
    return flag & valid


DEFAULT_OVERBURDEN_VELOCITY_M_S = 3000.0
DATUM_CHECK_MAX_RELATIVE_ERROR = 0.5


@dataclass
class DatumCheckResult:
    delay_ms: float
    implied_depth_m: float
    logged_top_depth_m: float
    relative_error: float
    avg_velocity_m_s: float
    plausible: bool


def cross_check_delay_datum(
    delay_ms: float,
    logged_top_depth_m: float,
    avg_velocity_m_s: float = DEFAULT_OVERBURDEN_VELOCITY_M_S,
    max_relative_error: float = DATUM_CHECK_MAX_RELATIVE_ERROR,
) -> DatumCheckResult:
    """Sanity-check the "seed the sonic integration at DelayRecordingTime"
    datum assumption (see depth_to_twt's t0_ms): convert the seismic
    survey's recording delay to an implied depth using a plausible average
    overburden velocity, and compare against the LOGGED interval's own top
    depth. These LAS files only log a partial reservoir interval (not from
    surface), so a 0-based sonic-integration datum is physically wrong on
    its own -- anchoring at the delay instead is a reasonable approximation
    ONLY if the delay actually corresponds to roughly "surface to the top
    of the logged interval". This is NOT a real calibration (a single
    average velocity is a rough stand-in for a geologically complex,
    unknown overburden) -- it exists to catch the case where that
    assumption is wildly wrong (e.g. the delay reflects something else
    entirely, like a processing static) rather than silently proceeding
    with a datum that doesn't mean what it's assumed to mean.
    """
    one_way_time_s = (delay_ms / 1000.0) / 2.0
    implied_depth_m = one_way_time_s * avg_velocity_m_s
    if logged_top_depth_m > 0:
        relative_error = abs(implied_depth_m - logged_top_depth_m) / logged_top_depth_m
    else:
        relative_error = float("inf")
    return DatumCheckResult(
        delay_ms=delay_ms,
        implied_depth_m=implied_depth_m,
        logged_top_depth_m=logged_top_depth_m,
        relative_error=relative_error,
        avg_velocity_m_s=avg_velocity_m_s,
        plausible=relative_error <= max_relative_error,
    )


DEFAULT_MAD_THRESHOLD = 5.0
DEFAULT_MAD_FLOOR_FRACTION = 0.15


def despike_mad(
    signal: np.ndarray,
    threshold_n_mad: float = DEFAULT_MAD_THRESHOLD,
    mad_floor_fraction: float = DEFAULT_MAD_FLOOR_FRACTION,
) -> np.ndarray:
    """Remove spikes from a signal via a MAD (median absolute deviation)
    threshold, floored at mad_floor_fraction of the signal's own RMS
    amplitude. A naive MAD threshold (no floor) degenerates to ~0 on
    sparse signals -- like a reflectivity series, which is mostly
    near-zero between reflectors -- which would flag almost every
    nonzero sample as a "spike" and zero it out, silently destroying the
    signal before correlation/QC even runs. Flooring the MAD keeps the
    threshold from collapsing on sparse-but-real signals, so only genuine
    outliers get removed.

    Returns a copy of signal with flagged samples set to 0. Raises
    TieError if despiking collapses the signal to zero variance (while
    the input had real variance) -- a sanity check that catches this
    class of bug at the source instead of surfacing downstream as an
    unexplained zero-correlation result.
    """
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if not finite.any():
        return signal.copy()

    median = np.median(signal[finite])
    mad = np.median(np.abs(signal[finite] - median))
    rms = np.sqrt(np.mean(signal[finite] ** 2))
    effective_mad = max(mad, mad_floor_fraction * rms)

    # No special-case for effective_mad == 0 -- that's exactly the naive
    # bug this function fixes (an unfloored MAD degenerates to 0 on a
    # sparse signal, so a 0 threshold flags every nonzero sample as a
    # "spike"). Let it compute naturally; the post-despike variance check
    # below is what actually catches an over-aggressive result, so a
    # deliberately-disabled floor is still observable instead of silently
    # masked.
    despiked = signal.copy()
    with np.errstate(invalid="ignore"):
        is_spike = finite & (np.abs(signal - median) > threshold_n_mad * effective_mad)
    despiked[is_spike] = 0.0

    input_std = float(np.std(signal[finite]))
    despiked_std = float(np.std(despiked[finite]))
    if input_std > 0 and despiked_std == 0:
        raise TieError(
            "Despiking collapsed the signal to zero variance -- the spike threshold was too "
            "aggressive for this (likely sparse) signal. Check mad_floor_fraction/threshold_n_mad "
            "rather than trusting downstream correlation/QC on this result."
        )
    return despiked


def build_synthetic(
    depth_m: np.ndarray,
    dt_log: np.ndarray,
    rhob: np.ndarray,
    seismic_dt_ms: float,
    seismic_twt_axis_ms: np.ndarray,
    wavelet_freq_hz: float = 30.0,
    dt_unit: str = "us_per_ft",
    t0_ms: float | None = None,
    despike: bool = True,
    avg_overburden_velocity_m_s: float = DEFAULT_OVERBURDEN_VELOCITY_M_S,
) -> SyntheticResult:
    """Full pipeline: sonic integration -> impedance -> reflectivity ->
    (optional) despike -> convolve with Ricker wavelet -> resample onto the
    seismic's own time axis.

    t0_ms: starting two-way time for the sonic-integrated curve (see
    depth_to_twt's docstring for why this matters). Defaults to
    seismic_twt_axis_ms[0] (the seismic survey's own DelayRecordingTime) --
    without an anchor, the well's own integrated curve starts at 0 ms and,
    resampled onto a real seismic survey's non-zero-delay time axis, has NO
    overlap and silently produces an all-zero synthetic (correlation 0).
    This is only a reasonable datum if the delay actually corresponds to
    roughly "surface to the top of the logged interval" -- see
    cross_check_delay_datum, run automatically here and returned as
    SyntheticResult.datum_check so callers can flag rather than silently
    trust it. Refine further with manual stretch/squeeze
    (apply_stretch_squeeze) once a real tie is available.

    despike: apply despike_mad to the reflectivity series before
    convolution (default True) -- a reflectivity series is sparse
    (near-zero between reflectors), which a naive spike threshold can
    mistake for "all outliers" and zero out entirely; see despike_mad.
    """
    valid = np.isfinite(depth_m) & np.isfinite(dt_log) & np.isfinite(rhob) & (dt_log > 0) & (rhob > 0)
    depth_m, dt_log, rhob = depth_m[valid], dt_log[valid], rhob[valid]

    if len(depth_m) < 10:
        raise TieError("Too few valid DT/RHOB samples after removing nulls/invalid values.")

    if t0_ms is None:
        t0_ms = float(seismic_twt_axis_ms[0]) if len(seismic_twt_axis_ms) else 0.0

    datum_check = cross_check_delay_datum(
        delay_ms=t0_ms, logged_top_depth_m=float(depth_m[0]), avg_velocity_m_s=avg_overburden_velocity_m_s
    )

    twt_ms = depth_to_twt(depth_m, dt_log, dt_unit=dt_unit, t0_ms=t0_ms)
    ai = acoustic_impedance(dt_log, rhob, dt_unit=dt_unit)
    refl = reflectivity_series(ai)
    if despike:
        refl = despike_mad(refl)
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
        datum_check=datum_check,
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


def apply_stretch_squeeze(
    depth_m: np.ndarray, twt_ms: np.ndarray, tie_points: list[tuple[float, float]]
) -> np.ndarray:
    """Apply manual stretch/squeeze correction to a depth-derived TWT curve:
    tie_points is a list of (md_m, time_shift_ms) control points a user has
    picked to nudge the sonic-integration time-depth relationship (no real
    checkshot exists to calibrate it otherwise). The shift is interpolated
    piecewise-linearly by depth and added to twt_ms; outside the control
    points' MD range, the shift holds constant at the nearest endpoint
    (np.interp's default behavior) rather than extrapolating.
    """
    if not tie_points:
        return twt_ms
    pts = sorted(tie_points, key=lambda p: p[0])
    mds = np.array([p[0] for p in pts], dtype=float)
    shifts = np.array([p[1] for p in pts], dtype=float)
    shift_at_depth = np.interp(depth_m, mds, shifts)
    return twt_ms + shift_at_depth


DEFAULT_MAX_SHIFT_MS = 300.0
BOUNDARY_PINNED_FRACTION = 0.05


def cross_correlate_and_shift(
    synthetic: np.ndarray, real: np.ndarray, dt_ms: float, max_shift_ms: float = DEFAULT_MAX_SHIFT_MS
) -> dict:
    """Cross-correlates synthetic vs. real trace, finds the best-fit lag
    WITHIN +/-max_shift_ms, and reports Pearson correlation at that
    alignment.

    max_shift_ms bounds the search range -- default +/-300ms. With no
    checkshot, the sonic-derived time-depth curve can be off from the
    seismic by 100-300ms even after anchoring at the delay datum (see
    build_synthetic's t0_ms/cross_check_delay_datum); a narrow window
    (e.g. +/-40ms) can silently return ~0 correlation for every well if
    the true answer sits just outside it, looking exactly like a broken
    tie rather than a search-range problem. Widen or narrow this per
    well/field as appropriate (e.g. narrower once a real checkshot
    constrains the answer).

    Also flags boundary_pinned: True if the best-fit shift lands within
    ~5% of the search range's edge. That's diagnostic of a spurious
    correlation match against noise -- the "best" shift keeps drifting
    toward the edge of the search window as you widen it, rather than
    converging to a stable interior value -- not a genuine tie.
    Boundary-pinned results should be excluded from aggregate statistics
    (mean correlation, ML training sets) by default.
    """
    synthetic = np.nan_to_num(synthetic)
    real = np.nan_to_num(real)

    syn_n = (synthetic - synthetic.mean()) / (synthetic.std() + 1e-12)
    real_n = (real - real.mean()) / (real.std() + 1e-12)

    xcorr = correlate(real_n, syn_n, mode="full")
    lags = np.arange(-len(syn_n) + 1, len(real_n))

    max_shift_samples = max_shift_ms / dt_ms if dt_ms > 0 else np.inf
    within_range = np.abs(lags) <= max_shift_samples
    if not within_range.any():
        raise TieError(
            f"max_shift_ms={max_shift_ms:g} (+/-{max_shift_samples:.1f} samples at dt_ms={dt_ms:g}) "
            "excludes every possible lag -- widen max_shift_ms or check dt_ms."
        )
    xcorr_bounded = np.where(within_range, xcorr, -np.inf)

    best_lag = int(lags[np.argmax(xcorr_bounded)])
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

    boundary_pinned = bool(abs(best_shift_ms) >= (1.0 - BOUNDARY_PINNED_FRACTION) * max_shift_ms)

    return {
        "best_shift_ms": best_shift_ms,
        "correlation": correlation,
        "shifted_synthetic": shifted_syn,
        "max_shift_ms": max_shift_ms,
        "boundary_pinned": boundary_pinned,
    }


# Matches the field range/count a real checkshot-free tie search needs to
# cover typical seismic bandwidth without being so dense it's mostly wasted
# compute -- 7 candidates, 5 Hz apart, spanning the usual "reasonable
# dominant frequency" range for this kind of survey.
DEFAULT_CANDIDATE_FREQS_HZ: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)


@dataclass
class TieSearchResult:
    """Winning (wavelet, polarity, position) combination from
    search_best_tie, plus the synthetic/correlation results at that
    combination. wavelet_freq_hz is None when a fixed (e.g. statistically
    extracted) wavelet was searched instead of a Ricker frequency grid."""

    wavelet_freq_hz: float | None
    polarity: int  # +1 or -1
    synthetic: np.ndarray  # unshifted, on the seismic's own time axis
    shifted_synthetic: np.ndarray
    best_shift_ms: float
    correlation: float
    max_shift_ms: float
    boundary_pinned: bool
    n_candidates_tried: int


def search_best_tie(
    refl_reg: np.ndarray,
    reg_twt_ms: np.ndarray,
    seismic_twt_axis_ms: np.ndarray,
    seismic_dt_ms: float,
    real_trace: np.ndarray,
    candidate_freqs_hz: tuple[float, ...] | None = DEFAULT_CANDIDATE_FREQS_HZ,
    search_polarity: bool = True,
    max_shift_ms: float = DEFAULT_MAX_SHIFT_MS,
    fixed_wavelet: np.ndarray | None = None,
) -> TieSearchResult:
    """Jointly searches wavelet frequency and polarity -- not just position
    (cross_correlate_and_shift's own +/-max_shift_ms search) -- keeping
    whichever (frequency, polarity, position) combination maximizes
    correlation against the real trace.

    Addresses a DIFFERENT failure mode than simply widening max_shift_ms:
    a synthetic built at the wrong dominant frequency or wrong polarity
    can look uncorrelated with the real trace at every possible shift, no
    matter how wide the position search is -- widening the shift range
    alone can't fix a wavelet assumption that's just wrong. Without a
    checkshot, neither the true dominant frequency nor the true polarity
    is known in advance, so both are search parameters here rather than
    fixed inputs.

    candidate_freqs_hz: Ricker frequencies to try (default
    DEFAULT_CANDIDATE_FREQS_HZ, 15-45 Hz). Pass None (with fixed_wavelet
    given instead) to search polarity only against ONE fixed wavelet --
    e.g. a statistically-extracted wavelet, which has no frequency
    parameter to sweep but is just as subject to a 180-degree polarity
    ambiguity as a Ricker wavelet is.

    search_polarity: try both the wavelet and its negation for every
    frequency candidate (2x the candidates) -- polarity is a genuine
    unknown without a known-polarity checkshot/VSP tie.

    NOT wired into any default path -- opt-in only (see
    synthetic_seismogram_service.generate's auto_optimize_tie), since
    this changes what "the" tie for a well is (which frequency, which
    polarity), not just how wide a position search runs.
    """
    if candidate_freqs_hz is None:
        if fixed_wavelet is None:
            raise TieError("search_best_tie needs either candidate_freqs_hz or fixed_wavelet.")
        candidates: list[tuple[float | None, np.ndarray]] = [(None, fixed_wavelet)]
    else:
        candidates = [(f, ricker_wavelet(f, seismic_dt_ms / 1000.0)[1]) for f in candidate_freqs_hz]

    polarities = (1, -1) if search_polarity else (1,)

    best: TieSearchResult | None = None
    n_tried = 0
    for freq_hz, base_wavelet in candidates:
        for polarity in polarities:
            n_tried += 1
            wavelet = polarity * base_wavelet
            full_conv = np.convolve(refl_reg, wavelet, mode="full")
            # See build_synthetic's identical crop -- np.convolve's "same"
            # mode isn't used because it returns max(len(a), len(v)) rather
            # than always len(a), which breaks for short well intervals
            # where the wavelet is longer than the reflectivity series.
            start = (len(full_conv) - len(refl_reg)) // 2
            synthetic_reg = full_conv[start : start + len(refl_reg)]
            synthetic_on_seismic_axis = np.interp(
                seismic_twt_axis_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0
            )
            tie = cross_correlate_and_shift(
                synthetic_on_seismic_axis, real_trace, seismic_dt_ms, max_shift_ms=max_shift_ms
            )
            if best is None or tie["correlation"] > best.correlation:
                best = TieSearchResult(
                    wavelet_freq_hz=freq_hz,
                    polarity=polarity,
                    synthetic=synthetic_on_seismic_axis,
                    shifted_synthetic=tie["shifted_synthetic"],
                    best_shift_ms=tie["best_shift_ms"],
                    correlation=tie["correlation"],
                    max_shift_ms=tie["max_shift_ms"],
                    boundary_pinned=tie["boundary_pinned"],
                    n_candidates_tried=0,  # filled in once, below, after the loop finishes
                )

    assert best is not None  # candidates always has >= 1 entry
    best.n_candidates_tried = n_tried
    return best