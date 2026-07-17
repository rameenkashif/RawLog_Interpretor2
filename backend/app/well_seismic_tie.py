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


# ---- Full-window tie search (vendor DPTM time axis) --------------------------
# This is a second, distinct tie pipeline alongside build_synthetic() /
# search_best_tie() above -- it doesn't re-derive a depth-time relationship
# via depth_to_twt at all. Instead it trusts a caller-supplied time axis
# directly (e.g. a well's own vendor-precomputed DPTM curve, see
# petrophysics.compute_dptm's preference for it over sonic integration), and
# searches wavelet frequency, polarity, AND bulk shift jointly across the
# entire seismic recording window rather than a narrow +/-max_shift_ms
# cross-correlation around a fixed position. Wired in by tie_service.py as
# the default for the dataset-based well tie.


def reflectivity_from_time_axis(
    time_ms: np.ndarray,
    dt_log: np.ndarray,
    rhob: np.ndarray,
    seismic_dt_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean/sort/dedupe a well's logs on a caller-supplied time axis, build
    an RHOB/DT acoustic-impedance proxy, resample onto the seismic's own
    sample rate, and difference to a reflectivity series.

    The impedance proxy is RHOB/DT rather than acoustic_impedance()'s
    velocity*density -- both give IDENTICAL reflectivity, since the missing
    velocity-conversion constant (dt_unit-dependent) is a pure scalar
    multiplier that cancels exactly in the reflectivity ratio
    (AI2-AI1)/(AI2+AI1); using the raw ratio directly here just avoids
    needing to know dt_unit for this path at all.

    time_ms is trusted as-is -- unlike build_synthetic's depth_to_twt path,
    nothing here re-derives a depth-time relationship. Raises TieError if
    there isn't enough valid, distinct, increasing time data to build a
    reflectivity series.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    dt_log = np.asarray(dt_log, dtype=float)
    rhob = np.asarray(rhob, dtype=float)

    valid = np.isfinite(time_ms) & np.isfinite(dt_log) & np.isfinite(rhob) & (dt_log > 0) & (rhob > 0)
    time_ms, dt_log, rhob = time_ms[valid], dt_log[valid], rhob[valid]
    if len(time_ms) < 10:
        raise TieError("Too few valid time/DT/RHOB samples to build a reflectivity series.")

    order = np.argsort(time_ms)
    time_ms, dt_log, rhob = time_ms[order], dt_log[order], rhob[order]
    keep = np.concatenate([[True], np.diff(time_ms) > 1e-6])
    time_ms, dt_log, rhob = time_ms[keep], dt_log[keep], rhob[keep]

    if len(time_ms) < 10:
        raise TieError("Too few distinct time samples after dedupe to build a reflectivity series.")

    ai = rhob / dt_log
    t0, t1 = float(time_ms[0]), float(time_ms[-1])
    if t1 <= t0:
        raise TieError("Time axis is not strictly increasing -- cannot build a uniform grid.")

    t_uniform = np.arange(t0, t1, seismic_dt_ms)
    ai_u = np.interp(t_uniform, time_ms, ai)

    with np.errstate(divide="ignore", invalid="ignore"):
        rc = (ai_u[1:] - ai_u[:-1]) / (ai_u[1:] + ai_u[:-1])
    rc = np.nan_to_num(rc)
    t_rc = t_uniform[:-1]
    return t_rc, rc


# Matches typical seismic bandwidth for a checkshot-free full-window search:
# 13 candidates, 2.5 Hz apart, spanning 15-45 Hz.
DEFAULT_TIE_SEARCH_FREQS_HZ: tuple[float, ...] = tuple(
    float(f) for f in np.arange(15.0, 45.0 + 1e-9, 2.5)
)
DEFAULT_TIE_SEARCH_MAX_SHIFT_MS = 100.0


@dataclass
class FullWindowTieResult:
    """Winning (frequency, polarity, shift) combination from
    search_best_tie_full_window, plus ready-to-plot arrays covering just the
    well's own reflectivity interval (NOT the full seismic trace -- a QC
    plot only needs the window the well actually has data for)."""

    best_freq_hz: float
    polarity: int  # +1 or -1
    bulk_shift_ms: float
    correlation: float
    n_used: int
    time_ms: np.ndarray  # t_rc + bulk_shift_ms, same length as reflectivity
    synthetic_amplitude: np.ndarray  # normalized, polarity-applied, full reflectivity-interval length
    seismic_amplitude: np.ndarray  # real trace interpolated onto time_ms, normalized, same length
    reflectivity: np.ndarray  # unshifted reflectivity series, same length


def search_best_tie_full_window(
    t_rc: np.ndarray,
    rc: np.ndarray,
    seismic_twt_axis_ms: np.ndarray,
    seismic_dt_ms: float,
    real_trace: np.ndarray,
    candidate_freqs_hz: tuple[float, ...] = DEFAULT_TIE_SEARCH_FREQS_HZ,
    max_shift_ms: float = DEFAULT_TIE_SEARCH_MAX_SHIFT_MS,
) -> FullWindowTieResult:
    """Jointly search Ricker wavelet frequency, polarity, and bulk time
    shift, sliding the synthetic across the ENTIRE seismic recording window
    (not a local search around a rough position) and keeping whichever
    combination maximizes normalized cross-correlation against the real
    trace.

    Distinct from search_best_tie() above in two ways: it operates on an
    already-built reflectivity series/time axis (see
    reflectivity_from_time_axis) rather than re-convolving inside
    build_synthetic's pipeline, and its position search is a full window
    scan by absolute time (checking real overlap against
    seismic_twt_axis_ms at each candidate shift) rather than a discrete
    sample-lag cross-correlation (cross_correlate_and_shift) -- appropriate
    here because, without a checkshot, the well's time axis (even a vendor
    DPTM curve) can plausibly sit anywhere in a wide window relative to the
    seismic, not just near a rough starting alignment.

    Raises TieError if no (frequency, polarity, shift) combination produces
    enough overlap with the seismic window to compute a correlation.
    """
    rc = np.asarray(rc, dtype=float)
    t_rc = np.asarray(t_rc, dtype=float)
    if len(rc) < 5:
        raise TieError("Reflectivity series too short for a full-window tie search.")

    wavelet_len_s = min(0.100, max(0.030, 0.6 * len(rc) * seismic_dt_ms / 1000.0))
    min_needed = min(30, max(10, int(0.5 * len(t_rc))))
    shift_candidates = np.arange(-max_shift_ms, max_shift_ms + 0.5 * seismic_dt_ms, seismic_dt_ms)

    best: dict | None = None
    for freq in candidate_freqs_hz:
        _, wav = ricker_wavelet(freq, seismic_dt_ms / 1000.0, wavelet_len_s)
        synth = np.convolve(rc, wav, mode="same")
        if len(synth) != len(rc):
            synth = synth[: len(rc)] if len(synth) > len(rc) else np.pad(synth, (0, len(rc) - len(synth)))
        synth = synth - synth.mean()
        if synth.std() > 0:
            synth = synth / synth.std()

        for polarity in (1, -1):
            s = polarity * synth
            for shift_ms in shift_candidates:
                t_shifted = t_rc + shift_ms
                mask = (t_shifted >= seismic_twt_axis_ms[0]) & (t_shifted <= seismic_twt_axis_ms[-1])
                if mask.sum() < min_needed:
                    continue
                seis_interp = np.interp(t_shifted[mask], seismic_twt_axis_ms, real_trace)
                ss = s[mask]
                if seis_interp.std() == 0 or ss.std() == 0:
                    continue
                seis_n = (seis_interp - seis_interp.mean()) / seis_interp.std()
                ss_n = (ss - ss.mean()) / ss.std()
                corr = float(np.mean(seis_n * ss_n))
                if best is None or corr > best["corr"]:
                    best = dict(
                        corr=corr,
                        freq=float(freq),
                        polarity=polarity,
                        shift_ms=float(shift_ms),
                        synth=s,
                        n_used=int(mask.sum()),
                    )

    if best is None:
        raise TieError(
            "No (frequency, polarity, shift) combination produced enough overlap with the "
            f"seismic window (need >= {min_needed} samples) -- check that the well's time axis "
            "actually falls near the seismic's recorded TWT range."
        )

    time_ms_full = t_rc + best["shift_ms"]
    seis_full = np.interp(time_ms_full, seismic_twt_axis_ms, real_trace)
    seis_std = seis_full.std()
    seismic_amplitude = (seis_full - seis_full.mean()) / seis_std if seis_std > 0 else seis_full - seis_full.mean()

    return FullWindowTieResult(
        best_freq_hz=best["freq"],
        polarity=best["polarity"],
        bulk_shift_ms=best["shift_ms"],
        correlation=best["corr"],
        n_used=best["n_used"],
        time_ms=time_ms_full,
        synthetic_amplitude=best["synth"],
        seismic_amplitude=seismic_amplitude,
        reflectivity=rc,
    )