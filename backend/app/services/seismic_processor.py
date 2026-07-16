"""
seismic_processor.py
----------------------
Direct-from-SEG-Y interpretation for the "Seismic Visualization" feature:
inline/crossline sections, time slices, well ties, and amplitude spectra.

This reads the raw SEG-Y volume in backend/data/seismic_raw/ directly via
segyio, rather than going through the upload/attribute pipeline
(segy_loader.py / seismic_repository.py) -- that pipeline never stores
inline/crossline geometry, which this feature needs to reshape traces into
2D sections and 3D-style time slices.

NON-STANDARD TRACE HEADER LAYOUT: this vendor's (LMKR) SEG-Y export
declares its own trace-header byte locations for inline/crossline (bytes
9-12/13-16, NOT the rev1-standard 189/193) directly in the textual
header ("Trace Inline At 9 And Size 4" etc.) -- see
app/segy_header_parser.py, which regex-parses those declarations (falling
back to rev1 standard locations for anything not declared) instead of
hardcoding either convention. The textual header itself is also plain
ASCII rather than the rev1-mandated EBCDIC, which segy_header_parser
detects by trying both encodings and keeping whichever decodes to more
printable text -- segyio's own f.text[] property always assumes EBCDIC
and would garble an ASCII header, so the raw bytes are read directly
instead. _verify_header_layout() double-checks the resolved byte
locations with a raw struct.unpack against trace 0 at file-open time, so
a segyio behavior change (or a byte-location parse gone wrong) fails
loudly rather than silently mis-reading the geometry.

The recording time axis also has a non-zero DelayRecordingTime (this
survey: 2030 ms) -- i.e. the first sample is NOT at TWT=0. That's read
explicitly from trace 0's header and used to build twt_axis_ms
(delay_ms + arange(n_samples)*sample_interval_ms) rather than trusting
segyio's f.samples to have picked it up correctly.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
import segyio
from scipy.signal import fftconvolve, hilbert, stft

logger = logging.getLogger("uvicorn.error")

from app import segy_header_parser as shp
from app import well_seismic_tie as wst
from app.services import coordinate_calibration_service as ccs
from app.services import well_service

RAW_SEISMIC_DIR = Path(__file__).resolve().parents[2] / "data" / "seismic_raw"

# Default number of traces sampled for a volume-wide amplitude spectrum when
# no inline is specified -- running an FFT over all 61k+ traces isn't
# necessary for a representative average spectrum, and would be far slower.
DEFAULT_SPECTRUM_SAMPLE_TRACES = 300

# ---- Spectral decomposition (STFT / CWT) parameters ------------------------
# STFT window: short (32 samples =~ 64 ms at 2 ms sampling) with heavy overlap
# (75%). This is a deliberate tradeoff for this survey's short (~624 ms)
# traces: a short window gives usable *time* resolution to localize tuning
# effects along the trace, at the cost of coarse *frequency* resolution
# (bin spacing = fs/nperseg =~ 500/32 =~ 15.6 Hz at 2 ms sampling). A longer
# window would sharpen frequency resolution but blur exactly when along the
# trace a given frequency's energy occurs, which defeats the purpose of
# spectral decomposition (vs. the single flat FFT in get_amplitude_spectrum).
STFT_WINDOW_SAMPLES = 32
STFT_OVERLAP_FRACTION = 0.75

# CWT: scipy.signal.cwt/morlet2 were removed in recent scipy releases (not
# available in this environment's scipy 1.17), so the Morlet wavelet is
# hand-rolled below -- the same approach well_seismic_tie.ricker_wavelet()
# already takes for the same reason. w0 = 6 cycles under the Gaussian
# envelope is the standard default balancing time vs. frequency resolution.
CWT_MORLET_W0 = 6.0
CWT_DEFAULT_FREQS_HZ: tuple[float, ...] = tuple(float(f) for f in range(5, 101, 5))

# SWT (Stationary/"undecimated" Wavelet Transform, via PyWavelets): unlike
# STFT/CWT's continuous frequency axis, SWT decomposes into a fixed set of
# discrete detail levels, each covering a dyadic (octave) frequency band --
# level N covers approximately [Nyquist/2^N, Nyquist/2^(N-1)] (level 1 =
# highest-frequency/finest-scale band, closest to Nyquist; higher levels
# step down an octave at a time). sym8 (Symlet-8) is the default: smoother
# and closer to linear phase than shorter wavelets, which matters for
# picking bed-boundary edges without introducing much phase distortion.
# coif3 (Coiflet-3) is offered as an alternative with a different
# smoothness/support tradeoff. Both are real, orthogonal wavelets (unlike
# the complex Morlet used for CWT), so amplitude here is the Hilbert
# envelope of the detail coefficients rather than |complex coefficient| --
# the seismic "instantaneous amplitude" convention, and visually comparable
# to STFT/CWT's smooth energy maps rather than a raw jagged coefficient.
SWT_DEFAULT_WAVELET = "sym8"
VALID_SWT_WAVELETS = ("sym8", "coif3")
SWT_MIN_LEVEL = 1
SWT_MAX_LEVEL = 6
SWT_DEFAULT_LEVEL = 3

# SSWT (Synchrosqueezed Wavelet Transform, via ssqueezepy's ssq_cwt): a
# post-processing reassignment of the CWT's coefficients to their true
# instantaneous frequency, sharpening the same time-frequency smearing the
# hand-rolled Morlet CWT above shows -- useful for resolving closely-spaced
# thin-bed frequency signatures that blur together in a plain CWT. Kept
# strictly opt-in (include_sswt query flag, CWT only) and trace-level only
# -- ssq_cwt operates on a single 1D signal and costs roughly an order of
# magnitude more than the existing per-trace CWT even after numba's one-time
# JIT warm-up (much worse on a cold process), so it is NOT wired into the
# inline/full-volume decomposition path or the XGBoost feature pipeline;
# see _decompose_sswt's benchmark log. ssqueezepy is imported lazily inside
# _decompose_sswt (not at module load time) so a missing/broken install only
# disables this one opt-in feature instead of the whole Spectral
# Decomposition module -- easy to rip out entirely if the compute cost
# proves impractical.
SSWT_MIN_SAMPLES = 8

# Typical usable seismic bandwidth, returned alongside the full frequency
# axis so the frontend can highlight/default-zoom to this band rather than
# the full 0-Nyquist range.
TYPICAL_USEFUL_BAND_HZ = (5.0, 80.0)

VALID_SPECTRAL_METHODS = ("stft", "cwt", "swt")


def _validate_spectral_method(method: str) -> str:
    method = method.lower()
    if method not in VALID_SPECTRAL_METHODS:
        raise SegyVolumeError(
            f"Unknown spectral decomposition method '{method}' -- expected one of "
            f"{VALID_SPECTRAL_METHODS}."
        )
    return method


def _validate_swt_wavelet(wavelet: str) -> str:
    wavelet = wavelet.lower()
    if wavelet not in VALID_SWT_WAVELETS:
        raise SegyVolumeError(
            f"Unknown SWT wavelet '{wavelet}' -- expected one of {VALID_SWT_WAVELETS}."
        )
    return wavelet


def _validate_swt_level(level: int) -> int:
    level = int(level)
    if not (SWT_MIN_LEVEL <= level <= SWT_MAX_LEVEL):
        raise SegyVolumeError(
            f"SWT level {level} out of range -- expected {SWT_MIN_LEVEL}-{SWT_MAX_LEVEL}."
        )
    return level


def _morlet_wavelet(freq_hz: float, dt_s: float, w0: float = CWT_MORLET_W0) -> np.ndarray:
    """Complex Morlet wavelet centered at freq_hz, sampled at dt_s, spanning
    +/-3 standard deviations of its Gaussian envelope."""
    sigma_t = w0 / (2 * np.pi * freq_hz)
    half_span = max(1, int(np.ceil(3 * sigma_t / dt_s)))
    t = np.arange(-half_span, half_span + 1) * dt_s
    norm = (np.pi**-0.25) / np.sqrt(sigma_t)
    return norm * np.exp(2j * np.pi * freq_hz * t) * np.exp(-(t**2) / (2 * sigma_t**2))


class SegyVolumeError(Exception):
    """Base class for seismic-visualization errors (bad inline/crossline/time,
    unreadable file, geometry mismatch, etc.)."""


class SegyFileNotFoundError(SegyVolumeError):
    """Raised when no SEG-Y file is available in backend/data/seismic_raw/."""


class MissingCurveError(SegyVolumeError):
    """Raised when a well-tie request needs a curve (DT/RHOB) the well doesn't have."""

    def __init__(self, well_id: str, curve: str):
        self.well_id = well_id
        self.curve = curve
        super().__init__(
            f"Well '{well_id}' has no usable '{curve}' curve (all null/missing) -- "
            f"'{curve}' is required to build the synthetic seismogram for a well tie."
        )


class CrsMismatchError(SegyVolumeError):
    """Raised when a well's LAS coordinates fall far outside the seismic
    survey's coordinate extent -- almost always a coordinate-reference-system
    mismatch (e.g. a placeholder/local grid vs. the survey's real projected
    CRS) rather than the well genuinely being outside the survey."""


@dataclass
class SurveyInfo:
    source_filename: str
    n_traces: int
    n_samples: int
    sample_interval_ms: float
    twt_start_ms: float
    twt_end_ms: float
    inline_min: int
    inline_max: int
    crossline_min: int
    crossline_max: int
    n_inlines: int
    n_crosslines: int
    best_time_ms: float
    textual_header_encoding: str
    byte_locations: dict[str, int]
    byte_locations_declared: dict[str, bool]
    delay_recording_time_ms: float
    delay_recording_time_uniform: bool


class SegyVolume:
    """Opens a SEG-Y file once, reads its full trace matrix and geometry
    into memory, and serves inline/crossline sections, time slices,
    amplitude spectra, and well ties. Read-only; safe to share across
    requests once constructed (see get_segy_volume() for the process-wide
    singleton).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise SegyFileNotFoundError(f"SEG-Y file not found: {self.path}")

        # Detect textual-header encoding (ASCII vs EBCDIC) and any
        # vendor-declared trace-header byte locations BEFORE opening with
        # segyio's own header parsing, since segyio's f.text[] always
        # assumes EBCDIC and would garble this vendor's plain-ASCII header
        # -- see segy_header_parser module docstring.
        header_result, byte_result = shp.detect_geometry(str(self.path))
        self.textual_header_encoding = header_result.encoding
        self.textual_header_printable_fraction = header_result.printable_fraction
        self.byte_locations = byte_result.byte_locations
        self.byte_locations_declared = byte_result.declared
        resolved_fields = shp.resolve_trace_fields(self.byte_locations)

        f = segyio.open(str(self.path), "r", ignore_geometry=True)
        try:
            f.mmap()
            self.n_traces = f.tracecount

            self.inline = np.asarray(f.attributes(resolved_fields["inline"])[:], dtype=int)
            self.crossline = np.asarray(f.attributes(resolved_fields["crossline"])[:], dtype=int)
            self.source_x = np.asarray(f.attributes(resolved_fields["source_x"])[:], dtype=float)
            self.source_y = np.asarray(f.attributes(resolved_fields["source_y"])[:], dtype=float)

            self._resolved_fields = resolved_fields
            self._verify_header_layout(f)

            # Explicit DelayRecordingTime read (trace header bytes 109-110)
            # rather than trusting f.samples to have picked it up --
            # DelayRecordingTime is the actual start of the recorded time
            # axis (this survey: 2030 ms, i.e. the first sample is NOT at
            # TWT=0). Sanity-checked for consistency across traces: a
            # varying delay would mean traces aren't directly comparable
            # sample-for-sample, which every 2D section/time-slice/well-tie
            # method in this class assumes.
            delay_all = np.asarray(f.attributes(segyio.TraceField.DelayRecordingTime)[:], dtype=float)
            self.delay_recording_time_ms = float(delay_all[0]) if len(delay_all) else 0.0
            self.delay_recording_time_uniform = bool(np.all(delay_all == delay_all[0])) if len(delay_all) else True

            interval_us = float(segyio.tools.dt(f))
            self.sample_interval_ms = interval_us / 1000.0
            self.n_samples = len(f.samples)
            self.twt_axis_ms = self.delay_recording_time_ms + np.arange(self.n_samples) * self.sample_interval_ms

            # Load the full amplitude matrix once (a ~90 MB SEG-Y file is a
            # ~75-80 MB float32 array in memory) so every read below is a
            # plain numpy index instead of a per-request disk read.
            self._traces = segyio.tools.collect(f.trace[:]).astype(np.float32)
        finally:
            f.close()

        if self.n_traces == 0:
            raise SegyVolumeError(f"SEG-Y file '{self.path.name}' contains no traces.")

        self.inline_min = int(self.inline.min())
        self.inline_max = int(self.inline.max())
        self.crossline_min = int(self.crossline.min())
        self.crossline_max = int(self.crossline.max())

        self._inlines_sorted = np.unique(self.inline)
        self._crosslines_sorted = np.unique(self.crossline)

        self._inline_index: dict[int, np.ndarray] = {}
        for il in self._inlines_sorted:
            idx = np.where(self.inline == il)[0]
            self._inline_index[int(il)] = idx[np.argsort(self.crossline[idx])]

        self._crossline_index: dict[int, np.ndarray] = {}
        for xl in self._crosslines_sorted:
            idx = np.where(self.crossline == xl)[0]
            self._crossline_index[int(xl)] = idx[np.argsort(self.inline[idx])]

        # Dense (n_inlines x n_crosslines) lookup table of trace index, used
        # by get_time_slice(). -1 marks a gap (this survey is documented as
        # a perfect regular grid, but the lookup degrades gracefully -- a
        # gap becomes NaN in the returned map instead of an index error).
        il_pos = np.searchsorted(self._inlines_sorted, self.inline)
        xl_pos = np.searchsorted(self._crosslines_sorted, self.crossline)
        self._grid_trace_idx = np.full(
            (len(self._inlines_sorted), len(self._crosslines_sorted)), -1, dtype=int
        )
        self._grid_trace_idx[il_pos, xl_pos] = np.arange(self.n_traces)

        # Some surveys are exported as a windowed/extracted subvolume rather
        # than a raw full cube -- each trace only carries real amplitude in
        # a window around its own horizon pick, and everything outside that
        # window is zero-padded to keep a uniform time axis across traces.
        # No single absolute time_ms then has every trace "on" at once,
        # so pick the sample with the fewest exact-zero traces as the
        # best default for the Time Slice view (see get_time_slice, which
        # also masks exact zero as NaN so padding renders as transparent
        # "no data" instead of a flat mid-colorscale blob).
        zero_frac_per_sample = np.mean(self._traces == 0.0, axis=0)
        self._best_time_sample_idx = int(np.argmin(zero_frac_per_sample))

        # Spectral decomposition is compute-heavier than the flat FFT above,
        # so full (inline, method) results are cached in memory -- repeated
        # frontend requests for the same inline (e.g. a user scrubbing a
        # frequency slider) index into the cached array instead of
        # recomputing the whole STFT/CWT. Cleared naturally whenever a new
        # SegyVolume is constructed (e.g. get_segy_volume(refresh=True)).
        self._spectral_cache: dict[tuple[int, str], dict] = {}

    def _verify_header_layout(self, f) -> None:
        """Cross-check the bulk header read against a manual struct.unpack of
        trace 0's raw header bytes at the DYNAMICALLY RESOLVED byte offsets
        (self.byte_locations, from segy_header_parser -- never hardcoded),
        so a segyio version/behavior difference -- or a byte-location parse
        gone wrong -- fails loudly instead of silently mis-reading this
        file's geometry."""
        raw = bytes(f.header[0].buf)

        def _unpack_at(byte_1indexed: int) -> int:
            offset = byte_1indexed - 1  # SEG-Y byte locations are 1-indexed
            return struct.unpack(">i", raw[offset : offset + 4])[0]

        expected = (
            _unpack_at(self.byte_locations["inline"]),
            _unpack_at(self.byte_locations["crossline"]),
            _unpack_at(self.byte_locations["source_x"]),
            _unpack_at(self.byte_locations["source_y"]),
        )
        actual = (int(self.inline[0]), int(self.crossline[0]), int(self.source_x[0]), int(self.source_y[0]))
        if expected != actual:
            raise SegyVolumeError(
                f"Trace header layout mismatch: manual byte-offset read of trace 0 at the "
                f"resolved locations {self.byte_locations} "
                f"(inline={expected[0]}, crossline={expected[1]}, srcX={expected[2]}, "
                f"srcY={expected[3]}) disagrees with the bulk field read (inline={actual[0]}, "
                f"crossline={actual[1]}, srcX={actual[2]}, srcY={actual[3]}). Refusing to serve "
                "possibly-wrong geometry."
            )

    # ---- geometry -----------------------------------------------------
    def survey_info(self) -> SurveyInfo:
        return SurveyInfo(
            source_filename=self.path.name,
            n_traces=self.n_traces,
            n_samples=self.n_samples,
            sample_interval_ms=self.sample_interval_ms,
            twt_start_ms=float(self.twt_axis_ms[0]),
            twt_end_ms=float(self.twt_axis_ms[-1]),
            inline_min=self.inline_min,
            inline_max=self.inline_max,
            crossline_min=self.crossline_min,
            crossline_max=self.crossline_max,
            n_inlines=len(self._inlines_sorted),
            n_crosslines=len(self._crosslines_sorted),
            best_time_ms=float(self.twt_axis_ms[self._best_time_sample_idx]),
            textual_header_encoding=self.textual_header_encoding,
            byte_locations=self.byte_locations,
            byte_locations_declared=self.byte_locations_declared,
            delay_recording_time_ms=self.delay_recording_time_ms,
            delay_recording_time_uniform=self.delay_recording_time_uniform,
        )

    def get_trace(self, index: int) -> np.ndarray:
        """Raw amplitude of a single trace by its flat index -- a public
        accessor so callers outside this module (e.g. the synthetic
        seismogram service) don't need to reach into the private
        self._traces matrix directly."""
        if not (0 <= index < self.n_traces):
            raise SegyVolumeError(f"Trace index {index} out of range [0, {self.n_traces}).")
        return self._traces[index].astype(float)

    def get_inline_section(self, inline_number: int) -> dict:
        idx = self._inline_index.get(int(inline_number))
        if idx is None:
            raise SegyVolumeError(
                f"Inline {inline_number} not found. Valid range: {self.inline_min}-{self.inline_max}."
            )
        amplitude = self._traces[idx].T  # (n_samples, n_traces_in_line)
        return {
            "inline_number": int(inline_number),
            "crossline_axis": self.crossline[idx].tolist(),
            "twt_axis_ms": self.twt_axis_ms.tolist(),
            "amplitude": amplitude.tolist(),
        }

    def get_crossline_section(self, crossline_number: int) -> dict:
        idx = self._crossline_index.get(int(crossline_number))
        if idx is None:
            raise SegyVolumeError(
                f"Crossline {crossline_number} not found. Valid range: "
                f"{self.crossline_min}-{self.crossline_max}."
            )
        amplitude = self._traces[idx].T  # (n_samples, n_traces_in_line)
        return {
            "crossline_number": int(crossline_number),
            "inline_axis": self.inline[idx].tolist(),
            "twt_axis_ms": self.twt_axis_ms.tolist(),
            "amplitude": amplitude.tolist(),
        }

    def get_time_slice(self, time_ms: float) -> dict:
        """Nearest-sample lookup -- a requested time outside [twt_start_ms,
        twt_end_ms] clamps to the nearest edge sample rather than erroring,
        since argmin-of-absolute-difference already picks the closest
        sample regardless of which side of the range it's on."""
        sample_idx = int(np.argmin(np.abs(self.twt_axis_ms - time_ms)))
        actual_time_ms = float(self.twt_axis_ms[sample_idx])

        grid = np.full(self._grid_trace_idx.shape, np.nan, dtype=float)
        valid = self._grid_trace_idx >= 0
        grid[valid] = self._traces[self._grid_trace_idx[valid], sample_idx]

        # Exact zero at this sample is padding, not a real amplitude
        # reading -- see the best_time_ms comment in __init__. Masking it
        # to NaN (same as an out-of-grid cell) renders it as transparent
        # "no data" instead of a flat mid-colorscale blob that visually
        # competes with real structure.
        grid[grid == 0.0] = np.nan

        return {
            "time_ms": actual_time_ms,
            "requested_time_ms": float(time_ms),
            "inline_axis": self._inlines_sorted.tolist(),
            "crossline_axis": self._crosslines_sorted.tolist(),
            "amplitude": grid.tolist(),
        }

    def get_amplitude_spectrum(
        self, inline_number: int | None = None, max_traces: int = DEFAULT_SPECTRUM_SAMPLE_TRACES
    ) -> dict:
        if inline_number is not None:
            idx = self._inline_index.get(int(inline_number))
            if idx is None:
                raise SegyVolumeError(
                    f"Inline {inline_number} not found. Valid range: {self.inline_min}-{self.inline_max}."
                )
            sample_traces = self._traces[idx]
        else:
            # Systematic (evenly spaced) sample across the whole volume
            # rather than every trace -- representative of the volume's
            # frequency content without an FFT over all 61k+ traces.
            step = max(1, self.n_traces // max_traces)
            sample_traces = self._traces[::step]

        n_samples = sample_traces.shape[1]
        dt_s = self.sample_interval_ms / 1000.0
        freqs = np.fft.rfftfreq(n_samples, d=dt_s)
        spectra = np.abs(np.fft.rfft(sample_traces, axis=1))
        avg_spectrum = spectra.mean(axis=0)

        # Skip the DC bin (index 0) when finding the dominant frequency --
        # it's driven by trace mean/drift, not seismic signal content.
        dominant_idx = int(np.argmax(avg_spectrum[1:])) + 1 if len(avg_spectrum) > 1 else 0
        dominant_freq_hz = float(freqs[dominant_idx])

        peak_amp = avg_spectrum[dominant_idx]
        half_power = peak_amp / np.sqrt(2)
        above_half_power = np.where(avg_spectrum >= half_power)[0]
        bandwidth_hz = (
            float(freqs[above_half_power[-1]] - freqs[above_half_power[0]])
            if len(above_half_power) > 1
            else 0.0
        )

        # Simple, uncalibrated S/N proxy: mean amplitude within the
        # half-power band vs. mean amplitude outside it. Not a real S/N
        # measurement (would need a noise-only reference window), just a
        # QC signal of how concentrated the spectrum's energy is.
        signal_power = float(avg_spectrum[above_half_power].mean()) if len(above_half_power) else 0.0
        noise_mask = np.ones(len(avg_spectrum), dtype=bool)
        noise_mask[above_half_power] = False
        noise_power = float(avg_spectrum[noise_mask].mean()) if noise_mask.any() else 0.0
        snr_proxy = (signal_power / noise_power) if noise_power > 0 else None

        return {
            "inline_number": inline_number,
            "n_traces_sampled": int(sample_traces.shape[0]),
            "freq_hz": freqs.tolist(),
            "amplitude": avg_spectrum.tolist(),
            "dominant_freq_hz": dominant_freq_hz,
            "bandwidth_hz": bandwidth_hz,
            "snr_proxy": snr_proxy,
        }

    # ---- spectral decomposition -----------------------------------------
    @property
    def _nyquist_hz(self) -> float:
        return 1000.0 / (2 * self.sample_interval_ms)

    def _decompose_stft(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched Short-Time Fourier Transform. traces: (n_pos, n_samples).
        Returns (freq_hz, time_ms, energy) where energy has shape
        (n_time, n_freq, n_pos) -- see module docstring for the window
        choice and its time/frequency-resolution tradeoff."""
        dt_s = self.sample_interval_ms / 1000.0
        fs = 1.0 / dt_s
        n_samples = traces.shape[-1]
        nperseg = min(STFT_WINDOW_SAMPLES, n_samples)
        noverlap = int(nperseg * STFT_OVERLAP_FRACTION)

        freqs, times_s, Zxx = stft(
            traces, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
            boundary=None, padded=False, axis=-1,
        )
        # rfft-based freqs already top out at fs/2 (Nyquist), but filter
        # explicitly so the Nyquist limit is enforced by construction, not
        # just by scipy's default behavior.
        keep = freqs <= self._nyquist_hz + 1e-9
        freqs = freqs[keep]
        Zxx = Zxx[..., keep, :] if Zxx.ndim == 3 else Zxx[keep, :]
        if Zxx.ndim == 2:  # single-trace input -> add back the position axis
            Zxx = Zxx[np.newaxis, ...]

        energy = np.abs(Zxx)  # (n_pos, n_freq, n_time)
        time_ms = self.twt_axis_ms[0] + times_s * 1000.0
        return freqs, time_ms, energy.transpose(2, 1, 0)  # (n_time, n_freq, n_pos)

    def _decompose_cwt(self, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous Wavelet Transform (hand-rolled complex Morlet -- see
        module docstring for why, instead of scipy.signal.cwt/morlet2).
        traces: (n_pos, n_samples). Returns (freq_hz, time_ms, energy) where
        energy has shape (n_time, n_freq, n_pos), at the trace's native
        sample resolution (unlike STFT, no windowing time-resolution loss)."""
        dt_s = self.sample_interval_ms / 1000.0
        freqs = np.array([f for f in CWT_DEFAULT_FREQS_HZ if f <= self._nyquist_hz], dtype=float)
        if freqs.size == 0:
            freqs = np.array([self._nyquist_hz * 0.5])

        n_pos, n_samples = traces.shape
        energy = np.empty((n_pos, len(freqs), n_samples), dtype=float)
        for i, freq_hz in enumerate(freqs):
            wavelet = _morlet_wavelet(float(freq_hz), dt_s)
            conv = fftconvolve(traces, wavelet[np.newaxis, :], mode="same", axes=1)
            energy[:, i, :] = np.abs(conv)

        return freqs, self.twt_axis_ms.copy(), energy.transpose(2, 1, 0)  # (n_time, n_freq, n_pos)

    def _swt_band_hz(self, level: int) -> tuple[float, float]:
        """Approximate dyadic frequency band for an SWT detail level, from
        this survey's own Nyquist frequency: level N ~= [Nyquist/2^N,
        Nyquist/2^(N-1)]. Just a labeling convenience for the UI -- unlike
        STFT/CWT's FFT-derived bins, this isn't an exact per-bin frequency,
        it's the nominal octave band a dyadic wavelet decomposition splits
        out at that level."""
        nyquist = self._nyquist_hz
        return nyquist / (2**level), nyquist / (2 ** (level - 1))

    def _decompose_swt(
        self, traces: np.ndarray, wavelet: str = SWT_DEFAULT_WAVELET
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stationary (undecimated) Wavelet Transform via PyWavelets.
        traces: (n_pos, n_samples). Returns (bands_hz, time_ms, energy)
        where bands_hz has shape (SWT_MAX_LEVEL, 2) -- one [lo, hi] Hz pair
        per level, see _swt_band_hz -- and energy has shape
        (n_time, n_level, n_pos), at the trace's native sample resolution
        (shift-invariant/no decimation, unlike a plain DWT).

        pywt.swt requires the transformed axis length to be a multiple of
        2**level; traces are reflect-padded up to the next multiple of
        2**SWT_MAX_LEVEL (covering every level up to the max in one call)
        and trimmed back to the original length afterward -- padding is
        appended only at the end, so trimming is a plain slice, no
        left-edge bookkeeping needed.

        Amplitude is the Hilbert envelope of each level's detail
        coefficients (cD), not a raw |coefficient| -- the standard seismic
        "instantaneous amplitude" convention, and visually consistent with
        STFT/CWT's smooth energy maps rather than a jagged real-valued
        wavelet coefficient.
        """
        n_pos, n_samples = traces.shape
        multiple = 2**SWT_MAX_LEVEL
        pad_len = (multiple - (n_samples % multiple)) % multiple

        if pad_len == 0:
            padded = traces
        else:
            # reflect-padding can't exceed the signal's own length; fall
            # back to edge-padding for a pathologically short trace rather
            # than letting numpy raise.
            pad_mode = "reflect" if pad_len < n_samples else "edge"
            padded = np.pad(traces, ((0, 0), (0, pad_len)), mode=pad_mode)

        # trim_approx=True -> [cA_max, cD_max, cD_(max-1), ..., cD_1], so
        # detail coefficients for level L sit at index (SWT_MAX_LEVEL+1-L).
        coeffs = pywt.swt(padded, wavelet, level=SWT_MAX_LEVEL, axis=-1, trim_approx=True)

        energy = np.empty((n_pos, SWT_MAX_LEVEL, n_samples), dtype=float)
        bands_hz = np.empty((SWT_MAX_LEVEL, 2), dtype=float)
        for level in range(SWT_MIN_LEVEL, SWT_MAX_LEVEL + 1):
            cD = coeffs[SWT_MAX_LEVEL + 1 - level][:, :n_samples]  # trim padding back off
            energy[:, level - 1, :] = np.abs(hilbert(cD, axis=-1))
            bands_hz[level - 1] = self._swt_band_hz(level)

        return bands_hz, self.twt_axis_ms.copy(), energy.transpose(2, 1, 0)  # (n_time, n_level, n_pos)

    def _decompose(self, traces: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._decompose_stft(traces) if method == "stft" else self._decompose_cwt(traces)

    def _decompose_sswt(self, trace_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Synchrosqueezed Wavelet Transform of a SINGLE trace (1D array),
        via ssqueezepy's ssq_cwt -- see the SSWT_MIN_SAMPLES comment above
        for why this stays trace-level-only and opt-in rather than joining
        _decompose/_decompose_cwt's batched, cached inline path.

        Returns (freq_hz, amplitude, compute_seconds) where amplitude has
        shape (n_time, n_freq): ssq_cwt itself returns Tx with shape
        (n_freq, n_time) and frequencies DESCENDING from Nyquist, both
        transposed/reversed here to match every other decomposition
        method's (n_time, n_freq) ascending-frequency convention.
        """
        try:
            from ssqueezepy import ssq_cwt
        except ImportError as exc:
            raise SegyVolumeError(
                "ssqueezepy is not installed -- SSWT (include_sswt=true) is an optional, opt-in "
                "dependency, separate from the rest of Spectral Decomposition. Run `pip install "
                "-r requirements.txt` (or `pip install ssqueezepy`) to enable it."
            ) from exc

        n_samples = trace_1d.shape[-1]
        if n_samples < SSWT_MIN_SAMPLES:
            raise SegyVolumeError(
                f"Trace too short ({n_samples} samples) for SSWT -- need >= {SSWT_MIN_SAMPLES}."
            )

        # ssq_cwt zeroes NaN/inf itself (with a printed warning) rather than
        # raising, but do it explicitly first so behavior doesn't depend on
        # that internal, undocumented fallback.
        clean_trace = np.nan_to_num(trace_1d, nan=0.0, posinf=0.0, neginf=0.0)
        fs = 1000.0 / self.sample_interval_ms

        t0 = time.perf_counter()
        Tx, _Wx, ssq_freqs, _scales = ssq_cwt(clean_trace, wavelet="morlet", fs=fs)
        compute_s = time.perf_counter() - t0

        amplitude = np.abs(Tx).T  # (n_freq, n_time) -> (n_time, n_freq)
        freq_hz = np.asarray(ssq_freqs, dtype=float)
        if freq_hz.size > 1 and freq_hz[0] > freq_hz[-1]:
            freq_hz = freq_hz[::-1]
            amplitude = amplitude[:, ::-1]

        return freq_hz, amplitude, compute_s

    def get_spectral_decomposition_inline(
        self,
        inline_number: int,
        method: str = "stft",
        frequency_hz: float | None = None,
        level: int | None = None,
        wavelet: str = SWT_DEFAULT_WAVELET,
    ) -> dict:
        """Time-frequency decomposition for every trace along an inline.

        For method='stft'/'cwt': with frequency_hz omitted, returns the
        full (time x freq x position) volume -- heavier, meant for an
        initial load or export. With frequency_hz given, returns just that
        single frequency's energy across the section (time x position,
        same shape convention as get_inline_section's "amplitude") -- the
        fast path meant for a frontend frequency slider, reusing the
        cached full decomposition after the first call for a given
        (inline, method) rather than recomputing the STFT/CWT on every
        slider tick.

        For method='swt': always returns a single level's detail-
        coefficient envelope (same "amplitude" shape as the STFT/CWT fast
        path above) -- SWT has no continuous frequency axis to browse the
        "full volume" of, only a handful of discrete dyadic levels (see
        _decompose_swt), so there's no heavier all-levels response to
        return here. level defaults to SWT_DEFAULT_LEVEL if omitted. All
        SWT_MAX_LEVEL levels are computed and cached together on first
        call per (inline, wavelet) -- cheap, since pywt.swt computes every
        level in one pass -- so switching levels on a slider is also just
        a cache slice, not a recompute.
        """
        method = _validate_spectral_method(method)
        idx = self._inline_index.get(int(inline_number))
        if idx is None:
            raise SegyVolumeError(
                f"Inline {inline_number} not found. Valid range: {self.inline_min}-{self.inline_max}."
            )

        if method == "swt":
            wavelet = _validate_swt_wavelet(wavelet)
            cache_key = (int(inline_number), method, wavelet)
            cached = self._spectral_cache.get(cache_key)
            if cached is None:
                bands_hz, time_ms, energy = self._decompose_swt(self._traces[idx].astype(float), wavelet)
                cached = {
                    "crossline_axis": self.crossline[idx].tolist(),
                    "bands_hz": bands_hz,  # (SWT_MAX_LEVEL, 2)
                    "time_ms": time_ms,
                    "energy": energy,  # (n_time, n_level, n_pos)
                }
                self._spectral_cache[cache_key] = cached

            lvl = _validate_swt_level(level if level is not None else SWT_DEFAULT_LEVEL)
            band = cached["bands_hz"][lvl - 1]
            amplitude_slice = cached["energy"][:, lvl - 1, :]  # (n_time, n_pos)
            return {
                "inline_number": int(inline_number),
                "method": method,
                "level": lvl,
                "wavelet": wavelet,
                "band_hz": [float(band[0]), float(band[1])],
                "nyquist_hz": self._nyquist_hz,
                "crossline_axis": cached["crossline_axis"],
                "time_ms": cached["time_ms"].tolist(),
                "amplitude": amplitude_slice.tolist(),
            }

        cache_key = (int(inline_number), method)
        cached = self._spectral_cache.get(cache_key)
        if cached is None:
            freq_hz, time_ms, energy = self._decompose(self._traces[idx].astype(float), method)
            cached = {
                "crossline_axis": self.crossline[idx].tolist(),
                "freq_hz": freq_hz,
                "time_ms": time_ms,
                "energy": energy,  # (n_time, n_freq, n_pos)
            }
            self._spectral_cache[cache_key] = cached

        if frequency_hz is None:
            return {
                "inline_number": int(inline_number),
                "method": method,
                "crossline_axis": cached["crossline_axis"],
                "time_ms": cached["time_ms"].tolist(),
                "freq_hz": cached["freq_hz"].tolist(),
                "nyquist_hz": self._nyquist_hz,
                "typical_band_hz": list(TYPICAL_USEFUL_BAND_HZ),
                "energy": cached["energy"].tolist(),
            }

        freq_idx = int(np.argmin(np.abs(cached["freq_hz"] - frequency_hz)))
        actual_freq = float(cached["freq_hz"][freq_idx])
        amplitude_slice = cached["energy"][:, freq_idx, :]  # (n_time, n_pos)
        return {
            "inline_number": int(inline_number),
            "method": method,
            "requested_frequency_hz": float(frequency_hz),
            "frequency_hz": actual_freq,
            "crossline_axis": cached["crossline_axis"],
            "time_ms": cached["time_ms"].tolist(),
            "amplitude": amplitude_slice.tolist(),
        }

    def get_spectral_decomposition_trace(
        self,
        inline_number: int,
        crossline_number: int,
        method: str = "stft",
        wavelet: str = SWT_DEFAULT_WAVELET,
        include_sswt: bool = False,
    ) -> dict:
        """Time-frequency decomposition for a single trace, e.g. for a
        trace-inspection view or well-tie context. For method='swt', energy
        covers all SWT_MAX_LEVEL levels (there's no frequency slider to
        page through here, so the full small set is returned directly).

        include_sswt (CWT only, ignored otherwise): additionally compute
        the Synchrosqueezed Wavelet Transform (see _decompose_sswt) and
        return it as extra sswt_freq_hz/sswt_amplitude/sswt_compute_ms
        fields ALONGSIDE the existing freq_hz/energy (the plain CWT) --
        additive, not a replacement, so existing CWT-only callers are
        unaffected. Logs a compute-time comparison against the plain CWT
        call, since SSWT is meaningfully more expensive (see module-level
        SSWT_MIN_SAMPLES comment) and that cost needs to be visible before
        anyone considers it for a full-volume or model-feature path.
        """
        method = _validate_spectral_method(method)
        idx = self._inline_index.get(int(inline_number))
        if idx is None:
            raise SegyVolumeError(
                f"Inline {inline_number} not found. Valid range: {self.inline_min}-{self.inline_max}."
            )
        match = np.where(self.crossline[idx] == int(crossline_number))[0]
        if match.size == 0:
            raise SegyVolumeError(
                f"No trace at inline={inline_number}, crossline={crossline_number}. Valid "
                f"crossline range: {self.crossline_min}-{self.crossline_max}."
            )
        trace_idx = int(idx[match[0]])

        trace = self._traces[trace_idx : trace_idx + 1].astype(float)  # (1, n_samples)

        if method == "swt":
            wavelet = _validate_swt_wavelet(wavelet)
            bands_hz, time_ms, energy = self._decompose_swt(trace, wavelet)
            energy_2d = energy[:, :, 0]  # (n_time, n_level)
            return {
                "inline_number": int(inline_number),
                "crossline_number": int(crossline_number),
                "method": method,
                "wavelet": wavelet,
                "time_ms": time_ms.tolist(),
                "levels": list(range(SWT_MIN_LEVEL, SWT_MAX_LEVEL + 1)),
                "bands_hz": bands_hz.tolist(),
                "nyquist_hz": self._nyquist_hz,
                "energy": energy_2d.tolist(),
            }

        cwt_t0 = time.perf_counter()
        freq_hz, time_ms, energy = self._decompose(trace, method)
        cwt_elapsed_s = time.perf_counter() - cwt_t0
        energy_2d = energy[:, :, 0]  # (n_time, n_freq)

        result = {
            "inline_number": int(inline_number),
            "crossline_number": int(crossline_number),
            "method": method,
            "time_ms": time_ms.tolist(),
            "freq_hz": freq_hz.tolist(),
            "nyquist_hz": self._nyquist_hz,
            "typical_band_hz": list(TYPICAL_USEFUL_BAND_HZ),
            "energy": energy_2d.tolist(),
        }

        if method == "cwt" and include_sswt:
            sswt_freq_hz, sswt_amplitude, sswt_compute_s = self._decompose_sswt(trace[0])
            logger.info(
                "SSWT benchmark -- trace inline=%s crossline=%s (%d samples): existing CWT "
                "(%d freq bins) took %.4fs, SSWT/ssq_cwt (%d freq bins) took %.4fs (%.1fx as long). "
                "Note: ssq_cwt's first call in a fresh process is far slower than this due to "
                "one-time numba JIT compilation.",
                inline_number, crossline_number, trace.shape[-1],
                len(freq_hz), cwt_elapsed_s, len(sswt_freq_hz), sswt_compute_s,
                sswt_compute_s / max(cwt_elapsed_s, 1e-9),
            )
            result["sswt_freq_hz"] = sswt_freq_hz.tolist()
            result["sswt_amplitude"] = sswt_amplitude.tolist()
            result["sswt_compute_ms"] = sswt_compute_s * 1000.0

        return result

    def get_grid_geometry(self) -> dict:
        """Public accessor for the internal dense (inline, crossline) ->
        trace index lookup table and its sorted axes, for callers outside
        this class that need to work across the full grid (e.g.
        well_zone_tie_service's IDW interpolation) without duplicating the
        grid-building logic from __init__. grid_trace_idx entries are -1
        for a gap (see __init__)."""
        return {
            "grid_trace_idx": self._grid_trace_idx,
            "inlines_sorted": self._inlines_sorted,
            "crosslines_sorted": self._crosslines_sorted,
        }

    # ---- well tie -------------------------------------------------------
    def check_crs_alignment(self, well_id: str, well_x: float, well_y: float) -> None:
        x_min, x_max = float(self.source_x.min()), float(self.source_x.max())
        y_min, y_max = float(self.source_y.min()), float(self.source_y.max())
        # Generous buffer (20% of the survey's extent, floored at 500 m) so
        # a well just outside the survey footprint but genuinely in the
        # same CRS isn't rejected -- this is a coarse "same ballpark" check,
        # not a precise inside/outside-survey test.
        buffer_x = max(0.2 * (x_max - x_min), 500.0)
        buffer_y = max(0.2 * (y_max - y_min), 500.0)
        if not (
            x_min - buffer_x <= well_x <= x_max + buffer_x
            and y_min - buffer_y <= well_y <= y_max + buffer_y
        ):
            raise CrsMismatchError(
                f"Well '{well_id}' surface coordinates (X={well_x:.1f}, Y={well_y:.1f}) fall far "
                f"outside this seismic survey's coordinate extent (X: {x_min:.1f}-{x_max:.1f}, "
                f"Y: {y_min:.1f}-{y_max:.1f}). This almost always means the well's LAS "
                "coordinates and the SEG-Y's trace coordinates are in different coordinate "
                "reference systems (e.g. a placeholder/local grid vs. the survey's real "
                "projected CRS), not that the well is genuinely outside the survey -- verify "
                "both are in the same CRS before trusting a tie. Refusing to guess a nearest "
                "trace across a likely CRS mismatch."
            )

    def get_well_tie(self, well_id: str, wavelet_freq_hz: float = 25.0) -> dict:
        # Reuses well_service (LAS loading/repository) and well_seismic_tie
        # (impedance/reflectivity/Ricker-wavelet synthetic) rather than
        # re-implementing either -- see tie_service.py for the analogous
        # flow against the upload-pipeline's seismic data. Well location is
        # resolved via coordinate_calibration_service, NOT a direct
        # find_nearest_trace_index(well_x, well_y, source_x, source_y) call
        # -- well and seismic coordinates are on different, unknown
        # coordinate reference systems (see that module's docstring), so a
        # raw distance comparison between them is meaningless without the
        # calibrated transform.
        well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent, before coordinate resolution
        trace_idx, distance_m, tie_method = ccs.resolve_well_trace_index(self, well_id)

        curves_response = well_service.get_well_curves(well_id)
        rows = curves_response["data"]

        def _extract(curve_name: str) -> np.ndarray:
            arr = np.array(
                [row.get(curve_name) if row.get(curve_name) is not None else np.nan for row in rows],
                dtype=float,
            )
            arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
            return arr

        depth = _extract("DEPT")
        dt_log = _extract("DT")
        rhob = _extract("RHOB")

        if not np.isfinite(dt_log).any():
            raise MissingCurveError(well_id, "DT")
        if not np.isfinite(rhob).any():
            raise MissingCurveError(well_id, "RHOB")

        result = wst.build_synthetic(
            depth_m=depth,
            dt_log=dt_log,
            rhob=rhob,
            seismic_dt_ms=self.sample_interval_ms,
            seismic_twt_axis_ms=self.twt_axis_ms,
            wavelet_freq_hz=wavelet_freq_hz,
            dt_unit="us_per_ft",
        )
        real_trace = self._traces[trace_idx].astype(float)

        note = (
            "Depth-time relationship comes from integrating the sonic (DT) log only -- no "
            "checkshot/VSP survey is available for this well, so this is a simplification "
            "(accumulates sonic logging error with depth and ignores velocity anisotropy), "
            "not a calibrated depth-time tie. Anchored to this survey's own first sample time "
            "as a non-degenerate starting point (arbitrary, not physically derived), since the "
            "sonic-integrated curve alone has no absolute time reference. "
            "See well_seismic_tie.depth_to_twt."
        )

        return {
            "well_id": well_id,
            "wavelet_freq_hz": wavelet_freq_hz,
            "twt_ms": self.twt_axis_ms.tolist(),
            "synthetic": result.synthetic.tolist(),
            "real_trace": real_trace.tolist(),
            "nearest_inline": int(self.inline[trace_idx]),
            "nearest_crossline": int(self.crossline[trace_idx]),
            "distance_m": distance_m,
            "tie_method": tie_method,
            "note": note,
        }


def _discover_segy_path() -> Path:
    candidates = sorted(RAW_SEISMIC_DIR.glob("*.sgy")) + sorted(RAW_SEISMIC_DIR.glob("*.segy"))
    if not candidates:
        raise SegyFileNotFoundError(
            f"No .sgy/.segy file found in {RAW_SEISMIC_DIR}. Drop the seismic volume there "
            "to use the seismic visualization endpoints."
        )
    if len(candidates) == 1:
        return candidates[0]
    # This feature serves a single active volume (unlike the separate
    # upload/attribute pipeline in segy_loader.py, which supports many
    # named datasets at once) -- if several raw files are present, use the
    # most recently modified one.
    return max(candidates, key=lambda p: p.stat().st_mtime)


_volume_cache: dict[str, SegyVolume] = {}


def get_segy_volume(refresh: bool = False) -> SegyVolume:
    """Process-wide singleton accessor: the SEG-Y file is opened and fully
    read into memory once, then reused across requests. Pass refresh=True
    to force re-opening (e.g. after replacing the raw file)."""
    path = _discover_segy_path()
    key = str(path)
    if refresh or key not in _volume_cache:
        _volume_cache.clear()  # only one volume is ever cached at a time
        _volume_cache[key] = SegyVolume(path)
    return _volume_cache[key]
