"""
seismic_attributes.py
======================
Seismic attribute calculation engine, mirroring the structure of
petrophysics.py: one calculation per function, documented with its formula
and (where relevant) an explicit caveat about what it does and does not
tell you.

Input: a 2D amplitude matrix `traces` of shape (n_traces, n_samples), plus
the sample interval, as loaded by segy_loader.py.
Output: per-trace attribute arrays, assembled into a DataFrame by
`run_seismic_interpretation()`.

*** IMPORTANT CAVEAT ON VSH/PHIE/SWE "PROXIES" BELOW ***
Volume of shale, effective porosity, and water saturation are properly
derived from wireline logs (see petrophysics.py: Larionov VSH, density
porosity, Archie's equation). Raw post-stack seismic amplitude alone
cannot measure any of these directly -- doing so properly requires
seismic inversion to acoustic/elastic impedance, calibrated against a
real well tie, plus rock-physics relationships specific to the formation.

The "*_proxy" functions below are simple, UNCALIBRATED amplitude-based
heuristics included so lateral trends can be eyeballed on a dashboard
away from well control. They are explicitly flagged everywhere they
appear (API responses, docstrings, chat assistant system prompt) and
must not be used as a substitute for log-derived VSH/PHIE/SWE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import hilbert


# -----------------------------------------------------------------------------
# Core seismic attributes (standard, well-defined signal-processing quantities)
# -----------------------------------------------------------------------------
def compute_rms_amplitude(traces: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per trace.

        RMS_i = sqrt(mean(trace_i^2))

    A standard measure of overall reflectivity energy in a trace -- higher
    RMS amplitude generally indicates stronger/more numerous reflectors.
    """
    return np.sqrt(np.mean(np.square(traces), axis=1))


def compute_envelope(traces: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude envelope via the analytic signal (Hilbert transform).

        envelope(t) = |hilbert(trace)(t)|

    The envelope traces the "energy" of the seismic wavelet independent of
    its oscillating phase, and is the standard first step for many
    amplitude-based seismic attributes (bright spot detection, etc.).
    Returned with the same shape as `traces` (n_traces, n_samples).
    """
    return np.abs(hilbert(traces, axis=1))


def compute_average_envelope(traces: np.ndarray) -> np.ndarray:
    """Mean instantaneous-amplitude envelope per trace (see compute_envelope)."""
    return np.mean(compute_envelope(traces), axis=1)


def compute_dominant_frequency(
    traces: np.ndarray, sample_interval_ms: float
) -> np.ndarray:
    """Dominant (peak) frequency per trace via FFT magnitude spectrum.

        freq_i = argmax(|FFT(trace_i)|) mapped to Hz

    Sample interval is in milliseconds; converted to seconds for the FFT
    frequency-bin calculation. Useful as a rough indicator of tuning
    effects / thin-bed interference (lower dominant frequency can indicate
    a thin-bed tuning response) but is not itself a rock property.
    """
    n_samples = traces.shape[1]
    dt_s = sample_interval_ms / 1000.0
    freqs = np.fft.rfftfreq(n_samples, d=dt_s)
    spectrum = np.abs(np.fft.rfft(traces, axis=1))
    dominant_idx = np.argmax(spectrum, axis=1)
    return freqs[dominant_idx]


# -----------------------------------------------------------------------------
# Normalization helper
# -----------------------------------------------------------------------------
def _normalize_0_1(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Percentile-clipped min-max normalization to [0, 1].

    Clips to [low_percentile, high_percentile] before scaling so a single
    outlier trace (dead trace, noise burst, edge effect) doesn't compress
    the rest of the dataset's dynamic range to nearly nothing.
    """
    attr_cfg = config.get("attributes", {})
    low_p = attr_cfg.get("normalize_low_percentile", 2)
    high_p = attr_cfg.get("normalize_high_percentile", 98)

    low = np.percentile(values, low_p)
    high = np.percentile(values, high_p)
    if high <= low:
        return np.zeros_like(values)

    clipped = np.clip(values, low, high)
    return (clipped - low) / (high - low)


# -----------------------------------------------------------------------------
# Heuristic VSH / PHIE / SWE "seismic proxies" -- see module-level caveat above.
# -----------------------------------------------------------------------------
def compute_vsh_seismic_proxy(
    avg_envelope: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Amplitude-based lithology-contrast proxy (NOT a measured shale volume).

        VSH_SEISMIC_PROXY = normalize_0_1(avg_envelope)

    Rationale (heuristic, uncalibrated): stronger reflectivity/envelope
    amplitude often marks lithology boundaries or bedding contrasts. This
    does NOT distinguish shale from other lithology contrasts and must be
    calibrated against a real well tie before use.
    """
    if not config.get("vsh_proxy", {}).get("enabled", True):
        return np.full_like(avg_envelope, np.nan)
    return _normalize_0_1(avg_envelope, config)


def compute_phie_seismic_proxy(
    rms_amplitude: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Porosity-trend proxy from inverted relative amplitude (NOT a measured porosity).

        PHIE_SEISMIC_PROXY = 1 - normalize_0_1(rms_amplitude)

    Rationale (heuristic, uncalibrated): in clean elastic sands, acoustic
    impedance tends to anti-correlate with porosity (higher porosity ->
    lower impedance). True impedance requires seismic inversion; RMS
    amplitude is used here only as a rough stand-in for relative impedance
    trends, inverted so that "brighter/lower-amplitude" regions read as
    higher relative porosity. This is a trend indicator only.
    """
    if not config.get("phie_proxy", {}).get("enabled", True):
        return np.full_like(rms_amplitude, np.nan)
    return 1.0 - _normalize_0_1(rms_amplitude, config)


def compute_swe_seismic_proxy(
    rms_amplitude: np.ndarray, avg_envelope: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """ "Bright spot" hydrocarbon-indicator proxy (NOT a measured water saturation).

        bright = avg_envelope > percentile(avg_envelope, bright_spot_percentile)
        SWE_SEISMIC_PROXY = 1 - normalize_0_1(rms_amplitude), reduced further on bright traces

    Rationale (heuristic, uncalibrated): anomalously bright amplitude
    events can, in some AVO settings (e.g. Class 3 sands), indicate
    hydrocarbon presence. This flags candidate traces only -- tuning
    effects, lithology contrasts, and multiples all also produce bright
    amplitudes, so this heuristic produces many false positives and is a
    first-pass screening aid, not a substitute for Archie-derived Sw from
    resistivity logs.
    """
    swe_cfg = config.get("swe_proxy", {})
    percentile = swe_cfg.get("bright_spot_percentile", 90)

    base = 1.0 - _normalize_0_1(rms_amplitude, config)
    threshold = np.percentile(avg_envelope, percentile)
    is_bright = avg_envelope > threshold

    # Bright-spot traces get their proxy pulled toward 0 (lower "water
    # saturation" / more hydrocarbon-like) proportional to how far above
    # the threshold they are, capped so it never goes below 0.
    pulled = np.where(is_bright, base * 0.4, base)
    return np.clip(pulled, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Orchestration -- run the full seismic attribute pipeline for one dataset
# -----------------------------------------------------------------------------
def run_seismic_interpretation(
    traces: np.ndarray, sample_interval_ms: float, config: dict[str, Any]
) -> pd.DataFrame:
    """Run the complete seismic attribute pipeline on one dataset's amplitude
    matrix and return a per-trace DataFrame. Single entry point that
    services/routers should call, mirroring petrophysics.run_full_interpretation.
    """
    rms_amplitude = compute_rms_amplitude(traces)
    avg_envelope = compute_average_envelope(traces)
    dominant_freq = compute_dominant_frequency(traces, sample_interval_ms)

    vsh_proxy = compute_vsh_seismic_proxy(avg_envelope, config)
    phie_proxy = compute_phie_seismic_proxy(rms_amplitude, config)
    swe_proxy = compute_swe_seismic_proxy(rms_amplitude, avg_envelope, config)

    return pd.DataFrame(
        {
            "TRACE_INDEX": np.arange(traces.shape[0]),
            "RMS_AMPLITUDE": rms_amplitude,
            "AVG_ENVELOPE": avg_envelope,
            "DOMINANT_FREQ_HZ": dominant_freq,
            "VSH_SEISMIC_PROXY": vsh_proxy,
            "PHIE_SEISMIC_PROXY": phie_proxy,
            "SWE_SEISMIC_PROXY": swe_proxy,
        }
    )
