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

NON-STANDARD TRACE HEADER LAYOUT (confirmed by inspecting this file's
textual header): inline number lives at trace header bytes 9-12 and
crossline number at bytes 13-16. segyio's usual TraceField.INLINE_3D /
CROSSLINE_3D constants point at bytes 189/193 instead, and would silently
read the wrong values for this file -- so this module reads bytes 9-12 and
13-16 explicitly instead (those happen to be segyio's FieldRecord/
TraceNumber trace-header fields, which are just names for those same byte
offsets; _verify_header_layout() double-checks this with a raw
struct.unpack against trace 0 at file-open time so a segyio behavior
change would fail loudly rather than silently mis-read the geometry).
SourceX/SourceY (bytes 73-76/77-80) and the sample interval/delay
recording time (bytes 117-118/109-110) ARE the standard SEG-Y rev1
locations, so segyio's normal parsing (f.samples, TraceField.SourceX/Y) is
used for those.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import segyio
from scipy.signal import fftconvolve, stft

from app import well_seismic_tie as wst
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

# Typical usable seismic bandwidth, returned alongside the full frequency
# axis so the frontend can highlight/default-zoom to this band rather than
# the full 0-Nyquist range.
TYPICAL_USEFUL_BAND_HZ = (5.0, 80.0)

VALID_SPECTRAL_METHODS = ("stft", "cwt")


def _validate_spectral_method(method: str) -> str:
    method = method.lower()
    if method not in VALID_SPECTRAL_METHODS:
        raise SegyVolumeError(
            f"Unknown spectral decomposition method '{method}' -- expected one of "
            f"{VALID_SPECTRAL_METHODS}."
        )
    return method


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

        f = segyio.open(str(self.path), "r", ignore_geometry=True)
        try:
            f.mmap()
            self.n_traces = f.tracecount
            self.twt_axis_ms = np.array(f.samples, dtype=float)
            self.n_samples = len(self.twt_axis_ms)
            self.sample_interval_ms = (
                float(self.twt_axis_ms[1] - self.twt_axis_ms[0]) if self.n_samples > 1 else 0.0
            )

            # NON-STANDARD: inline at bytes 9-12 (FieldRecord), crossline at
            # bytes 13-16 (TraceNumber) -- NOT INLINE_3D/CROSSLINE_3D. See
            # module docstring.
            self.inline = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=int)
            self.crossline = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:], dtype=int)
            self.source_x = np.asarray(f.attributes(segyio.TraceField.SourceX)[:], dtype=float)
            self.source_y = np.asarray(f.attributes(segyio.TraceField.SourceY)[:], dtype=float)

            self._verify_header_layout(f)

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
        trace 0's raw header bytes at the documented offsets, so a segyio
        version/behavior difference fails loudly instead of silently
        mis-reading this file's non-standard inline/crossline layout."""
        raw = bytes(f.header[0].buf)
        inline0 = struct.unpack(">i", raw[8:12])[0]
        crossline0 = struct.unpack(">i", raw[12:16])[0]
        srcx0 = struct.unpack(">i", raw[72:76])[0]
        srcy0 = struct.unpack(">i", raw[76:80])[0]
        expected = (inline0, crossline0, srcx0, srcy0)
        actual = (int(self.inline[0]), int(self.crossline[0]), int(self.source_x[0]), int(self.source_y[0]))
        if expected != actual:
            raise SegyVolumeError(
                "Trace header layout mismatch: manual byte-offset read of trace 0 "
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

    def _decompose(self, traces: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._decompose_stft(traces) if method == "stft" else self._decompose_cwt(traces)

    def get_spectral_decomposition_inline(
        self, inline_number: int, method: str = "stft", frequency_hz: float | None = None
    ) -> dict:
        """Time-frequency decomposition for every trace along an inline.

        With frequency_hz omitted, returns the full (time x freq x position)
        volume -- heavier, meant for an initial load or export. With
        frequency_hz given, returns just that single frequency's energy
        across the section (time x position, same shape convention as
        get_inline_section's "amplitude") -- this is the fast path meant for
        a frontend frequency slider, and reuses the cached full decomposition
        after the first call for a given (inline, method) rather than
        recomputing the STFT/CWT on every slider tick.
        """
        method = _validate_spectral_method(method)
        idx = self._inline_index.get(int(inline_number))
        if idx is None:
            raise SegyVolumeError(
                f"Inline {inline_number} not found. Valid range: {self.inline_min}-{self.inline_max}."
            )

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
        self, inline_number: int, crossline_number: int, method: str = "stft"
    ) -> dict:
        """Time-frequency decomposition for a single trace, e.g. for a
        trace-inspection view or well-tie context."""
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
        freq_hz, time_ms, energy = self._decompose(trace, method)
        energy_2d = energy[:, :, 0]  # (n_time, n_freq)

        return {
            "inline_number": int(inline_number),
            "crossline_number": int(crossline_number),
            "method": method,
            "time_ms": time_ms.tolist(),
            "freq_hz": freq_hz.tolist(),
            "nyquist_hz": self._nyquist_hz,
            "typical_band_hz": list(TYPICAL_USEFUL_BAND_HZ),
            "energy": energy_2d.tolist(),
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
        # (impedance/reflectivity/Ricker-wavelet synthetic + nearest-trace
        # search) rather than re-implementing either -- see tie_service.py
        # for the analogous flow against the upload-pipeline's seismic data.
        well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
        if well_summary.well_x is None or well_summary.well_y is None:
            raise SegyVolumeError(
                f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot "
                "locate it within the seismic survey."
            )

        self.check_crs_alignment(well_id, well_summary.well_x, well_summary.well_y)

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

        trace_idx, distance_m = wst.find_nearest_trace_index(
            well_summary.well_x, well_summary.well_y, self.source_x, self.source_y
        )

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
