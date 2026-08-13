"""
services/sweet_spot_feature_engine.py
----------------------------------------
The 55-attribute (+3 scale-invariant) seismic feature pool for the
Sweet-Spot prediction module, following the reference V11 architecture
doc's "Step 3/4/5" feature engine as closely as its own description
specifies. Every family is computed VECTORIZED across every trace in a
region/neighborhood at once (2D/3D numpy, no per-trace Python loop),
matching seismic_processor._decompose_cwt's existing batched convention.

Shape convention: compute_feature_pool takes a 3D region array
(n_il, n_xl, n_time) -- the exact shape seismic_processor.get_region_traces
and get_trace_neighbors_3x3 already return -- and returns every feature as
a 2D (n_time, n_pos) array, where n_pos = n_il*n_xl, flattened in C-order
(position index = il_idx*n_xl + xl_idx). A caller that knows a trace's
(il_idx, xl_idx) position within the region (e.g. get_trace_neighbors_3x3's
center_il_pos/center_xl_pos) can always recover its column via
`il_idx * n_xl + xl_idx`.

Individual family functions (amplitude_family, envelope_family, ...) take
the already-flattened 2D (n_time, n_pos) traces directly, so they're
independently testable/reusable against a single trace too (n_pos=1).
Only structural_family needs the 3D grid shape, since inline/crossline
gradients are inherently spatial.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert

# ---- Family 1-4: per-sample shift taps ------------------------------------
# 11 taps per family: center (shift=0) + 5 deeper (+1..+5) + 5 shallower
# (-1..-5), matching the doc's "center + shift -5..+5" families exactly.
SHIFT_TAPS: tuple[int, ...] = tuple(range(-5, 6))


def _shift_key(prefix: str, k: int, center_name: str | None = None) -> str:
    if k == 0:
        return center_name if center_name is not None else f"{prefix}_center"
    return f"{prefix}_shift_+{k}" if k > 0 else f"{prefix}_shift_{k}"


def _shift(signal: np.ndarray, k: int) -> np.ndarray:
    """signal (n_time, n_pos) shifted by k samples along time (axis 0):
    positive k looks DEEPER (later time), negative k looks SHALLOWER
    (earlier time). Out-of-bounds edge rows become NaN rather than
    wrapping (np.roll's default), since a wrapped deep sample bleeding
    into a shallow one would silently corrupt the feature."""
    if k == 0:
        return signal.copy()
    shifted = np.roll(signal, -k, axis=0)
    if k > 0:
        shifted[-k:] = np.nan
    else:
        shifted[:-k] = np.nan
    return shifted


def _family_with_shifts(prefix: str, base: np.ndarray, center_name: str | None = None) -> dict[str, np.ndarray]:
    return {_shift_key(prefix, k, center_name): _shift(base, k) for k in SHIFT_TAPS}


# ---- Family 1: Amplitude (11) ----------------------------------------------
def amplitude_family(traces: np.ndarray) -> dict[str, np.ndarray]:
    """Raw amplitude at the center sample + shifted samples above/below --
    adjacent amplitudes capture local wavelet shape and interference
    (e.g. a thin-bed doublet's characteristic amplitude reversal)."""
    return _family_with_shifts("amp", traces)


# ---- Family 2: Envelope (11) -----------------------------------------------
def envelope_family(traces: np.ndarray) -> dict[str, np.ndarray]:
    """Hilbert envelope (instantaneous amplitude, phase-independent total
    acoustic energy) + its first derivative (rate of energy change, sharp
    at bed boundaries) + shifted taps."""
    envelope = np.abs(hilbert(traces, axis=0))
    deriv = np.gradient(envelope, axis=0)
    out = _family_with_shifts("env", envelope)
    out["env_deriv"] = deriv
    # Note: the doc's own family header says "(11 attributes)" but its
    # bullet list (center + deriv + 10 shifts = 12) doesn't match that
    # count -- a minor inconsistency in the source doc. Nothing it
    # describes by name is dropped here; CURATED_FEATURES only ever uses
    # env_center/env_deriv anyway, so this doesn't affect training.
    return out


# ---- Family 3: Instantaneous frequency (11) --------------------------------
def inst_freq_family(traces: np.ndarray, dt_ms: float) -> dict[str, np.ndarray]:
    """Time-derivative of the analytic signal's unwrapped phase, in Hz --
    drops in gas-saturated sands (velocity dispersion), rises in tight
    carbonates; physically tied to rock stiffness/pore-fluid compressibility."""
    phase = np.unwrap(np.angle(hilbert(traces, axis=0)), axis=0)
    dt_s = dt_ms / 1000.0
    ifreq = np.gradient(phase, axis=0) / (2.0 * np.pi * dt_s)
    return _family_with_shifts("ifreq", ifreq)


# ---- Family 4: Sweetness (11) ----------------------------------------------
def sweetness_family(envelope: np.ndarray, inst_freq: np.ndarray) -> dict[str, np.ndarray]:
    """Sweetness = envelope / sqrt(|instantaneous frequency|) -- highlights
    high-amplitude, low-frequency anomalies, a classic gas/light-hydrocarbon
    sand-channel indicator. abs() on inst_freq since phase noise can drive
    it transiently negative, which sqrt can't take; eps guards a
    near-zero-frequency division."""
    eps = 1e-6
    sweetness = envelope / np.sqrt(np.abs(inst_freq) + eps)
    # Doc's curated list names the center tap "sweetness", not
    # "sweetness_center" (unlike amp/env/ifreq) -- matched verbatim here.
    return _family_with_shifts("sweetness", sweetness, center_name="sweetness")


# ---- Family 5: CWT spectral decomposition (5) ------------------------------
# (label, low_hz, high_hz) -- matches the doc's table exactly (Section 6).
CWT_BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("10hz", 8.0, 12.0),
    ("15hz", 13.0, 17.0),
    ("20hz", 18.0, 22.0),
    ("30hz", 27.0, 33.0),
    ("40hz", 37.0, 43.0),
)


def cwt_band_energy_family(traces: np.ndarray, dt_ms: float) -> dict[str, np.ndarray]:
    """Bandpass Hilbert envelope at 5 discrete frequency bands, via a hard
    FFT-domain cutoff + inverse FFT + Hilbert envelope of the real part --
    the doc's own exact method (Section 6), not a Butterworth filter.
    Low frequencies see thick regional trends; the 40 Hz band is the most
    diagnostic for sub-tuning thin beds (Section 6's "key insight")."""
    n_time = traces.shape[0]
    freqs = np.fft.fftfreq(n_time, d=dt_ms / 1000.0)
    f_trace = np.fft.fft(traces, axis=0)
    out: dict[str, np.ndarray] = {}
    for label, low, high in CWT_BANDS_HZ:
        band_mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
        f_filtered = f_trace.copy()
        f_filtered[~band_mask, :] = 0
        filtered = np.fft.ifft(f_filtered, axis=0)
        out[f"spec_amp_{label}"] = np.abs(hilbert(np.real(filtered), axis=0))
    return out


# ---- Family 6: Rolling window statistics (4) -------------------------------
def rolling_window_family(signal: np.ndarray, window: int = 5) -> dict[str, np.ndarray]:
    """Rolling window max/mean/min/std over the RAW AMPLITUDE trace (local
    amplitude variability is a direct heterogeneity/reflector-density
    indicator, per the doc's own rationale) -- centered window, edge
    samples (where a full window doesn't fit) are NaN rather than a
    shrunken partial window."""
    half = window // 2
    n_time = signal.shape[0]
    windows = sliding_window_view(signal, window_shape=window, axis=0)  # (n_time-window+1, n_pos, window)

    out_max = np.full(signal.shape, np.nan)
    out_mean = np.full(signal.shape, np.nan)
    out_min = np.full(signal.shape, np.nan)
    out_std = np.full(signal.shape, np.nan)
    out_max[half : n_time - half] = windows.max(axis=-1)
    out_mean[half : n_time - half] = windows.mean(axis=-1)
    out_min[half : n_time - half] = windows.min(axis=-1)
    out_std[half : n_time - half] = windows.std(axis=-1)
    return {"win_max": out_max, "win_mean": out_mean, "win_min": out_min, "win_std": out_std}


# ---- Family 7: Structural & positional (7) ---------------------------------
def _neighbor_grid_diff(grid_3d: np.ndarray, axis: int, offset: int) -> np.ndarray:
    """grid_3d - grid_3d shifted by `offset` positions along spatial `axis`
    (0=inline, 1=crossline) -- i.e. neighbor_value - this_position's_value.
    Edge positions with no such neighbor become NaN."""
    shifted = np.roll(grid_3d, -offset, axis=axis)
    n = grid_3d.shape[axis]
    if offset > 0:
        idx = [slice(None)] * grid_3d.ndim
        idx[axis] = slice(n - offset, n)
        shifted[tuple(idx)] = np.nan
    elif offset < 0:
        idx = [slice(None)] * grid_3d.ndim
        idx[axis] = slice(0, -offset)
        shifted[tuple(idx)] = np.nan
    return shifted - grid_3d


def structural_family(traces_3d: np.ndarray, dt_ms: float) -> dict[str, np.ndarray]:
    """Structural/positional attributes computed on the natural
    (n_il, n_xl, n_time) grid (needs real spatial neighbors, unlike
    families 1-6), then flattened to (n_time, n_pos) C-order like every
    other family:

    - polarity_index: amplitude/envelope, bounded [-1, 1] -- peak vs. trough
    - rel_pos: normalized position within the trace (0-1), compaction-trend proxy
    - acoustic_impedance: cumulative-amplitude-sum relative impedance proxy
      (RAW cumulative sum along time -- a genuine impedance log grows
      roughly monotonically with cumulative reflectivity, this is the
      seismic-only analogue with no absolute scale)
    - calibrated_ai: the impedance proxy above, locally normalized against
      its own rolling mean/std (removes the proxy's arbitrary absolute
      scale/drift without needing an external well-tie fit inside this
      pure feature function -- the doc's own "calibrated to well tie
      scale/shift" is otherwise underspecified beyond this)
    - neighborhood gradients: n_il_p1/n_xl_p1 acoustic-impedance diff,
      n_il_m1 instantaneous-frequency diff (this position's inline+1/
      crossline+1/inline-1 neighbor's value minus this position's own)
    """
    n_il, n_xl, n_time = traces_3d.shape
    envelope_3d = np.abs(hilbert(traces_3d, axis=-1))
    eps = 1e-9
    polarity_index_3d = traces_3d / (envelope_3d + eps)
    polarity_index_3d = np.clip(polarity_index_3d, -1.0, 1.0)

    rel_pos_1d = np.linspace(0.0, 1.0, n_time)
    rel_pos_3d = np.broadcast_to(rel_pos_1d, traces_3d.shape).copy()

    ai_proxy_3d = np.cumsum(traces_3d, axis=-1)

    window = min(21, n_time if n_time % 2 == 1 else n_time - 1)
    window = max(window, 3)
    ai_2d_for_roll = ai_proxy_3d.transpose(2, 0, 1).reshape(n_time, n_il * n_xl)
    roll_stats = rolling_window_family(ai_2d_for_roll, window=window)
    roll_mean = roll_stats["win_mean"].reshape(n_time, n_il, n_xl).transpose(1, 2, 0)
    roll_std = roll_stats["win_std"].reshape(n_time, n_il, n_xl).transpose(1, 2, 0)
    calibrated_ai_3d = (ai_proxy_3d - roll_mean) / (roll_std + eps)

    phase_3d = np.unwrap(np.angle(hilbert(traces_3d, axis=-1)), axis=-1)
    ifreq_3d = np.gradient(phase_3d, axis=-1) / (2.0 * np.pi * (dt_ms / 1000.0))

    n_il_p1_ai_diff_3d = _neighbor_grid_diff(ai_proxy_3d, axis=0, offset=1)
    n_xl_p1_ai_diff_3d = _neighbor_grid_diff(ai_proxy_3d, axis=1, offset=1)
    n_il_m1_ifreq_diff_3d = _neighbor_grid_diff(ifreq_3d, axis=0, offset=-1)

    def _flatten(grid_3d: np.ndarray) -> np.ndarray:
        return grid_3d.transpose(2, 0, 1).reshape(n_time, n_il * n_xl)

    return {
        "polarity_index": _flatten(polarity_index_3d),
        "rel_pos": _flatten(rel_pos_3d),
        "acoustic_impedance": _flatten(ai_proxy_3d),
        "calibrated_ai": _flatten(calibrated_ai_3d),
        "n_il_p1_acoustic_impedance_diff": _flatten(n_il_p1_ai_diff_3d),
        "n_xl_p1_acoustic_impedance_diff": _flatten(n_xl_p1_ai_diff_3d),
        "n_il_m1_ifreq_center_diff": _flatten(n_il_m1_ifreq_diff_3d),
    }


# ---- Scale-invariant (SI) features (3) -------------------------------------
def si_features(
    traces_3d: np.ndarray, band_energy: dict[str, np.ndarray], amp_center: np.ndarray, env_center: np.ndarray
) -> dict[str, np.ndarray]:
    """Spectral-shape (not magnitude) features, robust across amplitude
    variation between strong and weak reflectors -- see doc Section 7."""
    eps = 1e-9
    total_spec = sum(band_energy.values()) + eps
    si_spec_frac_10 = band_energy["spec_amp_10hz"] / total_spec
    si_spec_frac_40 = band_energy["spec_amp_40hz"] / total_spec

    n_il, n_xl, n_time = traces_3d.shape
    n_il_p1_amp_diff_3d = _neighbor_grid_diff(
        amp_center.reshape(n_time, n_il, n_xl).transpose(1, 2, 0), axis=0, offset=1
    )
    n_il_p1_amp_diff = n_il_p1_amp_diff_3d.transpose(2, 0, 1).reshape(n_time, n_il * n_xl)
    si_norm_grad_il = n_il_p1_amp_diff / (env_center + eps)

    return {
        "si_spec_frac_10": si_spec_frac_10,
        "si_spec_frac_40": si_spec_frac_40,
        "si_norm_grad_il": si_norm_grad_il,
    }


def compute_feature_pool(traces_3d: np.ndarray, dt_ms: float) -> dict[str, np.ndarray]:
    """All 55 + 3 SI = 58 attributes, keyed by attr_*/si_* name (prefix
    added here to match the doc's exact column-naming convention), each a
    (n_time, n_pos) array with n_pos = n_il*n_xl flattened C-order.

    traces_3d: (n_il, n_xl, n_time), e.g. straight from
    seismic_processor.get_region_traces or get_trace_neighbors_3x3's
    'traces' field.
    """
    n_il, n_xl, n_time = traces_3d.shape
    traces_2d = traces_3d.transpose(2, 0, 1).reshape(n_time, n_il * n_xl)

    amp = amplitude_family(traces_2d)
    env = envelope_family(traces_2d)
    ifreq = inst_freq_family(traces_2d, dt_ms)
    sweetness = sweetness_family(env["env_center"], ifreq["ifreq_center"])
    bands = cwt_band_energy_family(traces_2d, dt_ms)
    window_stats = rolling_window_family(traces_2d, window=5)
    structural = structural_family(traces_3d, dt_ms)

    pool: dict[str, np.ndarray] = {}
    for name, arr in {**amp, **env, **ifreq, **sweetness, **bands, **window_stats}.items():
        pool[f"attr_{name}"] = arr
    for name, arr in structural.items():
        pool[f"attr_{name}"] = arr

    si = si_features(traces_3d, bands, pool["attr_amp_center"], pool["attr_env_center"])
    pool.update(si)  # si_* names already match the doc's convention as-is

    return pool


# ---- Curated 22-feature set -------------------------------------------------
# Taken VERBATIM from the source doc's REDUCED_FEATURES list (Section 8) --
# not re-derived -- selected there to balance information content against
# curse-of-dimensionality risk on a small (7-well) dataset.
CURATED_FEATURES: tuple[str, ...] = (
    "attr_amp_center",
    "attr_amp_shift_+2",
    "attr_env_center",
    "attr_env_deriv",
    "attr_ifreq_center",
    "attr_ifreq_shift_+1",
    "attr_sweetness",
    "attr_sweetness_shift_+2",
    "attr_spec_amp_10hz",
    "attr_spec_amp_20hz",
    "attr_spec_amp_30hz",
    "attr_spec_amp_40hz",
    "attr_win_std",
    "attr_win_max",
    "attr_n_il_p1_acoustic_impedance_diff",
    "attr_n_xl_p1_acoustic_impedance_diff",
    "attr_n_il_m1_ifreq_center_diff",
    "attr_polarity_index",
    "attr_rel_pos",
    "si_spec_frac_10",
    "si_spec_frac_40",
    "si_norm_grad_il",
)
