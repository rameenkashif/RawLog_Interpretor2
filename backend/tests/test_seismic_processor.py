"""
test_seismic_processor.py
----------------------------
Unit + integration tests for app/services/seismic_processor.py (the
"Seismic Visualization" feature) and its router (app/routers/seismic_viz.py).

Builds a small synthetic SEG-Y file with the SAME non-standard trace header
layout as the real production file (inline at bytes 9-12/FieldRecord,
crossline at bytes 13-16/TraceNumber, SourceX/Y at the standard 73-80,
delay recording time + sample interval set explicitly), so the tests
exercise the real byte-offset parsing without needing the real ~90 MB file.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app.las_loader import FT_TO_M
from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import well_service


def _set_las_coordinate(las_text: str, mnemonic: str, value: float) -> str:
    """Overwrite a ~Well section header value (e.g. X, Y, KB, TD) in raw LAS
    text, regardless of what value/whitespace/unit is currently there --
    regex on the mnemonic plus its (variable-width) unit field rather than
    an exact-string match on the old value, so this doesn't silently no-op
    if the checked-in LAS file's coordinates change."""
    pattern = re.compile(rf"^{re.escape(mnemonic)}\s*\.\S*\s+[-\d.]+", re.MULTILINE)
    new_text, n = pattern.subn(f"{mnemonic}.m {value:.2f}", las_text, count=1)
    assert n == 1, f"Expected exactly one {mnemonic} header line to replace, found {n}"
    return new_text


def _set_las_coordinate_meters(las_text: str, mnemonic: str, meters: float) -> str:
    """Like _set_las_coordinate, but for the real Z-02 file's X/Y specifically:
    that file's X/Y/KB/TD are stored in feet (mislabeled '.m') and get
    auto-converted to meters by las_loader's unit standardization (detected
    via its TD/STOP ratio -- see test_las_loader.py), so this injects the
    feet-equivalent of the desired final meters value."""
    return _set_las_coordinate(las_text, mnemonic, meters / FT_TO_M)

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"

# Matches the real survey's non-standard layout at a much smaller scale:
# 5 inlines x 4 crosslines = 20 traces, 6 samples at 2 ms starting at 2030 ms
# (2030-2040 ms), so boundary-clamp behavior (< 2030 ms) is testable exactly
# like the real 2030-2654 ms volume.
INLINES = list(range(382, 387))  # 382..386
CROSSLINES = list(range(46, 50))  # 46..49
N_SAMPLES = 6
DELAY_MS = 2030
INTERVAL_MS = 2


def _write_synthetic_segy(
    path: Path,
    source_x_base: float = 363000.0,
    source_y_base: float = 2949800.0,
    n_samples: int = N_SAMPLES,
    interval_ms: int = INTERVAL_MS,
    delay_ms: int = DELAY_MS,
) -> None:
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(n_samples) * interval_ms + delay_ms
    spec.tracecount = len(INLINES) * len(CROSSLINES)

    rng = np.random.default_rng(42)
    i = 0
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = interval_ms * 1000
        # Mirrors the real production file's vendor-declared (non-rev1)
        # byte locations for inline/crossline, so tests exercise
        # segy_header_parser's byte-location parsing the same way the real
        # file does rather than falling back to the rev1 defaults (which
        # would silently break every inline=9/crossline=13-based test
        # below). SourceX/Y are left undeclared -- like the real file,
        # they're at the rev1 standard location (73/77) already.
        f.text[0] = (
            "C 1 CLIENT LMKR SURVEY TEST TRACE INLINE AT 9 AND SIZE 4 "
            "TRACE CROSSLINE AT 13 AND SIZE 4"
        ).ljust(3200)
        for il in INLINES:
            for xl in CROSSLINES:
                f.header[i] = {
                    segyio.TraceField.FieldRecord: il,
                    segyio.TraceField.TraceNumber: xl,
                    segyio.TraceField.SourceX: int(source_x_base + il * 10),
                    segyio.TraceField.SourceY: int(source_y_base + xl * 10),
                    segyio.TraceField.DelayRecordingTime: delay_ms,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: interval_ms * 1000,
                }
                # Distinct-ish waveform per trace so section/spectrum tests
                # have something non-degenerate to check.
                f.trace[i] = (
                    np.sin(np.linspace(0, 2 * np.pi, n_samples)) + rng.normal(0, 0.05, n_samples)
                ).astype(np.float32)
                i += 1


def _write_windowed_synthetic_segy(path: Path) -> None:
    """Half the traces are zero-padded at the shallow end (samples 0-1),
    half at the deep end (samples 4-5) -- mimics a horizon-windowed
    extraction subvolume where no single absolute time has full coverage,
    but the middle samples (2-3, where both halves have real data) do.
    Used to test best_time_ms and get_time_slice's exact-zero-as-NaN
    masking (see seismic_processor.SegyVolume._best_time_sample_idx)."""
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(N_SAMPLES) * INTERVAL_MS + DELAY_MS
    spec.tracecount = len(INLINES) * len(CROSSLINES)

    shallow_padded = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    deep_padded = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0], dtype=np.float32)

    i = 0
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = INTERVAL_MS * 1000
        # See _write_synthetic_segy's comment above -- without a declared
        # byte location, segy_header_parser falls back to the rev1
        # standard (189/193) instead of this fixture's FieldRecord/
        # TraceNumber (9/13) convention.
        f.text[0] = (
            "C 1 TRACE INLINE AT 9 AND SIZE 4 TRACE CROSSLINE AT 13 AND SIZE 4"
        ).ljust(3200)
        for il in INLINES:
            for xl in CROSSLINES:
                f.header[i] = {
                    segyio.TraceField.FieldRecord: il,
                    segyio.TraceField.TraceNumber: xl,
                    segyio.TraceField.SourceX: int(363000.0 + il * 10),
                    segyio.TraceField.SourceY: int(2949800.0 + xl * 10),
                    segyio.TraceField.DelayRecordingTime: DELAY_MS,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: INTERVAL_MS * 1000,
                }
                f.trace[i] = shallow_padded if i < 10 else deep_padded
                i += 1


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path)
    return sp.SegyVolume(path)


class TestGeometry:
    def test_survey_info_matches_known_dimensions(self, volume):
        info = volume.survey_info()
        assert info.n_traces == len(INLINES) * len(CROSSLINES) == 20
        assert info.n_samples == N_SAMPLES
        assert info.inline_min == 382 and info.inline_max == 386
        assert info.crossline_min == 46 and info.crossline_max == 49

    def test_survey_info_surfaces_header_diagnostics(self, volume):
        info = volume.survey_info()
        # This fixture's textual header declares inline/crossline at bytes
        # 9/13 (see _write_synthetic_segy) -- resolved dynamically, not
        # hardcoded (segy_header_parser).
        assert info.byte_locations["inline"] == 9
        assert info.byte_locations["crossline"] == 13
        assert info.byte_locations_declared["inline"] is True
        assert info.byte_locations_declared["crossline"] is True
        # source_x/source_y weren't declared -- defaulted to rev1 standard.
        assert info.byte_locations_declared["source_x"] is False
        assert info.byte_locations_declared["source_y"] is False
        assert info.textual_header_encoding in ("cp037", "ascii", "latin-1")
        assert info.delay_recording_time_ms == DELAY_MS
        assert info.delay_recording_time_uniform is True
        assert info.n_inlines == 5 and info.n_crosslines == 4
        assert info.twt_start_ms == 2030.0
        assert info.sample_interval_ms == 2.0
        # No zero-padded samples in this fixture's waveform, so every
        # sample is equally "fully covered" and argmin picks the first.
        assert info.best_time_ms == 2030.0

    def test_inline_section_known_trace_count(self, volume):
        section = volume.get_inline_section(384)
        assert section["crossline_axis"] == CROSSLINES
        assert len(section["amplitude"]) == N_SAMPLES  # rows = samples
        assert all(len(row) == len(CROSSLINES) for row in section["amplitude"])

    def test_crossline_section_known_trace_count(self, volume):
        section = volume.get_crossline_section(47)
        assert section["inline_axis"] == INLINES
        assert len(section["amplitude"]) == N_SAMPLES
        assert all(len(row) == len(INLINES) for row in section["amplitude"])

    def test_inline_section_out_of_range_raises(self, volume):
        with pytest.raises(sp.SegyVolumeError):
            volume.get_inline_section(999)

    def test_crossline_section_out_of_range_raises(self, volume):
        with pytest.raises(sp.SegyVolumeError):
            volume.get_crossline_section(999)


class TestTimeSlice:
    def test_exact_sample_time(self, volume):
        ts = volume.get_time_slice(2034.0)
        assert ts["time_ms"] == 2034.0
        assert len(ts["amplitude"]) == 5  # n_inlines
        assert len(ts["amplitude"][0]) == 4  # n_crosslines

    def test_boundary_value_clamps_to_first_sample(self, volume):
        """time_ms=2029 (below the 2030 ms start) should clamp to the
        nearest sample, 2030 ms -- not error and not extrapolate."""
        ts = volume.get_time_slice(2029.0)
        assert ts["time_ms"] == 2030.0
        assert ts["requested_time_ms"] == 2029.0

    def test_value_past_end_clamps_to_last_sample(self, volume):
        ts = volume.get_time_slice(9999.0)
        assert ts["time_ms"] == 2040.0  # DELAY_MS + (N_SAMPLES-1)*INTERVAL_MS


class TestWindowedCoverage:
    """Some real surveys are exported as a horizon-windowed extraction
    rather than a raw full cube -- each trace is only "live" in a window
    around its own horizon pick, zero-padded outside it. No single
    absolute time then has every trace on at once. See best_time_ms and
    get_time_slice's exact-zero-as-NaN masking in seismic_processor.py."""

    def test_best_time_ms_picks_fullest_overlap_sample(self, tmp_path):
        path = tmp_path / "windowed.sgy"
        _write_windowed_synthetic_segy(path)
        volume = sp.SegyVolume(path)
        info = volume.survey_info()
        # Sample index 2 (2034 ms) is the only one where neither the
        # shallow-padded nor deep-padded half is zero.
        assert info.best_time_ms == 2034.0

    def test_get_time_slice_masks_padding_as_nan(self, tmp_path):
        path = tmp_path / "windowed.sgy"
        _write_windowed_synthetic_segy(path)
        volume = sp.SegyVolume(path)

        shallow = volume.get_time_slice(2030.0)  # sample 0: 10/20 traces padded
        flat = [v for row in shallow["amplitude"] for v in row]
        assert sum(1 for v in flat if v != v) == 10  # NaN != NaN

        full = volume.get_time_slice(2034.0)  # sample 2: fully covered
        flat_full = [v for row in full["amplitude"] for v in row]
        assert all(v == v for v in flat_full)


class TestAmplitudeSpectrum:
    def test_whole_volume_spectrum_shape(self, volume):
        result = volume.get_amplitude_spectrum()
        assert result["n_traces_sampled"] == 20
        assert len(result["freq_hz"]) == len(result["amplitude"])
        assert result["dominant_freq_hz"] >= 0

    def test_single_inline_spectrum(self, volume):
        result = volume.get_amplitude_spectrum(inline_number=384)
        assert result["n_traces_sampled"] == len(CROSSLINES)

    def test_unknown_inline_raises(self, volume):
        with pytest.raises(sp.SegyVolumeError):
            volume.get_amplitude_spectrum(inline_number=999)


class TestWellTie:
    @pytest.fixture
    def well_repo(self, tmp_path):
        return FileWellRepository(base_dir=tmp_path / "wells")

    @pytest.fixture
    def loaded_well(self, well_repo, monkeypatch):
        monkeypatch.setattr(
            well_service, "get_well_summary",
            lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
        )
        monkeypatch.setattr(
            well_service, "get_well_curves",
            lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
        )
        las_bytes = Z02_PATH.read_bytes()
        return well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)

    def test_well_far_outside_survey_raises_crs_mismatch(self, volume, loaded_well):
        # Z-02's real (unit-standardized) coordinates are ~700m past this
        # synthetic test survey's narrow SourceX/Y range (by construction --
        # see INLINES/CROSSLINES/source_x_base/source_y_base above) --
        # get_well_tie must flag this as a likely CRS mismatch rather than
        # silently tying to whatever the nearest (very distant) trace
        # happens to be.
        with pytest.raises(sp.CrsMismatchError):
            volume.get_well_tie(loaded_well.well_id)

    def test_well_tie_succeeds_when_coordinates_align(self, volume, well_repo, monkeypatch):
        monkeypatch.setattr(
            well_service, "get_well_summary",
            lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
        )
        monkeypatch.setattr(
            well_service, "get_well_curves",
            lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
        )
        # Patch the LAS bytes' X/Y to land inside this synthetic survey's
        # coordinate extent before loading, simulating a well whose
        # coordinates really are in the same CRS as the seismic.
        las_text = Z02_PATH.read_text()
        las_text = _set_las_coordinate_meters(las_text, "X", 366840.0)
        las_text = _set_las_coordinate_meters(las_text, "Y", 2950275.0)
        result = well_service.process_and_store_las_bytes(
            las_text.encode(), "Z-02_raw.las", repo=well_repo
        )
        assert result.well_x == pytest.approx(366840.0)

        tie = volume.get_well_tie(result.well_id, wavelet_freq_hz=25.0)
        assert tie["well_id"] == result.well_id
        assert len(tie["twt_ms"]) == len(tie["synthetic"]) == len(tie["real_trace"]) == N_SAMPLES
        assert tie["distance_m"] >= 0
        assert "note" in tie and "sonic" in tie["note"].lower()

    def test_missing_dt_curve_raises_clear_error(self, well_repo, volume, monkeypatch):
        monkeypatch.setattr(
            well_service, "get_well_summary",
            lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
        )
        monkeypatch.setattr(
            well_service, "get_well_curves",
            lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
        )
        las_text = Z02_PATH.read_text()
        las_text = _set_las_coordinate_meters(las_text, "X", 366840.0)
        las_text = _set_las_coordinate_meters(las_text, "Y", 2950275.0)
        result = well_service.process_and_store_las_bytes(
            las_text.encode(), "Z-02_raw.las", repo=well_repo
        )

        # Null out the DT curve in the stored (already-interpreted) DataFrame
        # to simulate a well that has no usable sonic log, then re-save --
        # las_loader.py requires DT at LAS-parse time, so this is the
        # realistic way a "missing" curve shows up after processing (all
        # null after cleaning, rather than absent as a column).
        metadata, df = well_repo.get_well(result.well_id)
        df["DT"] = np.nan
        well_repo.save_well(metadata, df)

        with pytest.raises(sp.MissingCurveError) as exc_info:
            volume.get_well_tie(result.well_id)
        assert exc_info.value.curve == "DT"


class TestSpectralDecomposition:
    @pytest.fixture
    def wide_volume(self, tmp_path) -> sp.SegyVolume:
        """More samples than the base `volume` fixture (80 vs. 6) at the
        real survey's 2 ms interval, so STFT's 32-sample window has
        something meaningful to slide across -- Nyquist here is still
        250 Hz."""
        path = tmp_path / "wide_survey.sgy"
        _write_synthetic_segy(path, n_samples=80, interval_ms=2)
        return sp.SegyVolume(path)

    @pytest.fixture
    def low_nyquist_volume(self, tmp_path) -> sp.SegyVolume:
        """20 ms sample interval -> fs=50 Hz -> Nyquist=25 Hz, well below
        CWT's default 5-100 Hz frequency grid -- exercises the actual
        Nyquist-clamping logic (not just "the default grid happens to be
        under 250 Hz")."""
        path = tmp_path / "low_nyquist_survey.sgy"
        _write_synthetic_segy(path, n_samples=80, interval_ms=20)
        return sp.SegyVolume(path)

    def test_stft_frequencies_never_exceed_nyquist(self, wide_volume):
        result = wide_volume.get_spectral_decomposition_inline(384, method="stft")
        assert result["nyquist_hz"] == pytest.approx(250.0)
        assert max(result["freq_hz"]) <= result["nyquist_hz"] + 1e-9

    def test_cwt_frequencies_never_exceed_nyquist(self, low_nyquist_volume):
        result = low_nyquist_volume.get_spectral_decomposition_inline(384, method="cwt")
        assert result["nyquist_hz"] == pytest.approx(25.0)
        assert max(result["freq_hz"]) <= result["nyquist_hz"] + 1e-9
        # CWT's default grid goes up to 100 Hz -- confirm it was actually
        # clamped down, not just coincidentally already low.
        assert max(result["freq_hz"]) < 100.0

    def test_trace_decomposition_also_respects_nyquist(self, low_nyquist_volume):
        result = low_nyquist_volume.get_spectral_decomposition_trace(384, 47, method="cwt")
        assert max(result["freq_hz"]) <= result["nyquist_hz"] + 1e-9

    def test_frequency_slice_matches_inline_section_position_shape(self, wide_volume):
        section = wide_volume.get_inline_section(384)
        slice_result = wide_volume.get_spectral_decomposition_inline(
            384, method="stft", frequency_hz=50.0
        )
        assert slice_result["crossline_axis"] == section["crossline_axis"]
        n_pos = len(section["crossline_axis"])
        assert all(len(row) == n_pos for row in slice_result["amplitude"])

    def test_frequency_slice_matches_inline_section_position_shape_cwt(self, wide_volume):
        section = wide_volume.get_inline_section(384)
        slice_result = wide_volume.get_spectral_decomposition_inline(
            384, method="cwt", frequency_hz=30.0
        )
        assert slice_result["crossline_axis"] == section["crossline_axis"]
        n_pos = len(section["crossline_axis"])
        assert all(len(row) == n_pos for row in slice_result["amplitude"])

    def test_stft_and_cwt_produce_internally_consistent_but_different_resolutions(self, wide_volume):
        stft_result = wide_volume.get_spectral_decomposition_trace(384, 47, method="stft")
        cwt_result = wide_volume.get_spectral_decomposition_trace(384, 47, method="cwt")

        # Each method's own energy array must be internally consistent with
        # its own time/freq axes...
        for result in (stft_result, cwt_result):
            n_time, n_freq = len(result["time_ms"]), len(result["freq_hz"])
            assert len(result["energy"]) == n_time
            assert all(len(row) == n_freq for row in result["energy"])

        # ...even though the two methods are NOT required to share the same
        # time/frequency resolution (STFT is windowed/coarser in time; CWT
        # runs at native sample resolution) -- the response model must
        # handle both without forcing a common grid.
        assert len(stft_result["time_ms"]) != len(cwt_result["time_ms"])

    def test_unknown_method_raises(self, wide_volume):
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_inline(384, method="bogus")

    def test_unknown_trace_raises(self, wide_volume):
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_trace(384, 9999, method="stft")

    def test_full_decomposition_is_cached_across_calls(self, wide_volume):
        wide_volume.get_spectral_decomposition_inline(384, method="stft")
        cache_key = (384, "stft")
        assert cache_key in wide_volume._spectral_cache
        cached_energy = wide_volume._spectral_cache[cache_key]["energy"]
        # A second call (including a frequency-slice request) must reuse
        # the same cached array rather than recomputing it.
        wide_volume.get_spectral_decomposition_inline(384, method="stft", frequency_hz=40.0)
        assert wide_volume._spectral_cache[cache_key]["energy"] is cached_energy


class TestSwtSpectralDecomposition:
    """SWT (Stationary Wavelet Transform, via PyWavelets) -- a third
    spectral decomposition method alongside STFT/CWT, with a discrete
    level (not continuous frequency) axis. See seismic_processor.py's
    _decompose_swt for the padding/level/band-mapping rationale."""

    @pytest.fixture
    def wide_volume(self, tmp_path) -> sp.SegyVolume:
        path = tmp_path / "wide_survey.sgy"
        _write_synthetic_segy(path, n_samples=80, interval_ms=2)
        return sp.SegyVolume(path)

    def test_default_level_and_wavelet(self, wide_volume):
        result = wide_volume.get_spectral_decomposition_inline(384, method="swt")
        assert result["level"] == sp.SWT_DEFAULT_LEVEL == 3
        assert result["wavelet"] == sp.SWT_DEFAULT_WAVELET == "sym8"

    def test_band_hz_matches_formula(self, wide_volume):
        # Nyquist here is 250 Hz (2 ms interval) -- level N band should be
        # [Nyquist/2^N, Nyquist/2^(N-1)].
        for level in range(1, 7):
            result = wide_volume.get_spectral_decomposition_inline(384, method="swt", level=level)
            lo, hi = result["band_hz"]
            assert lo == pytest.approx(250.0 / (2**level))
            assert hi == pytest.approx(250.0 / (2 ** (level - 1)))

    def test_band_hz_decreases_as_level_increases(self, wide_volume):
        # Level 1 = finest scale = closest to Nyquist (highest band);
        # level 6 = coarsest = lowest band.
        bands = [
            wide_volume.get_spectral_decomposition_inline(384, method="swt", level=lvl)["band_hz"]
            for lvl in range(1, 7)
        ]
        highs = [b[1] for b in bands]
        assert highs == sorted(highs, reverse=True)

    def test_coif3_wavelet_selectable(self, wide_volume):
        result = wide_volume.get_spectral_decomposition_inline(384, method="swt", wavelet="coif3")
        assert result["wavelet"] == "coif3"

    def test_unknown_wavelet_raises(self, wide_volume):
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_inline(384, method="swt", wavelet="bogus")

    def test_level_out_of_range_raises(self, wide_volume):
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_inline(384, method="swt", level=7)
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_inline(384, method="swt", level=0)

    def test_amplitude_matches_inline_section_position_shape(self, wide_volume):
        section = wide_volume.get_inline_section(384)
        result = wide_volume.get_spectral_decomposition_inline(384, method="swt", level=2)
        assert result["crossline_axis"] == section["crossline_axis"]
        n_pos = len(section["crossline_axis"])
        assert all(len(row) == n_pos for row in result["amplitude"])
        # Time axis must match the trace's own native sample resolution
        # (no windowing loss, like CWT) -- same length as the section.
        assert len(result["time_ms"]) == len(section["twt_axis_ms"])
        assert len(result["amplitude"]) == len(result["time_ms"])

    def test_amplitude_is_non_negative(self, wide_volume):
        # Hilbert-envelope amplitude, not a raw signed coefficient.
        result = wide_volume.get_spectral_decomposition_inline(384, method="swt")
        assert all(v >= 0.0 for row in result["amplitude"] for v in row)

    def test_full_decomposition_is_cached_across_levels(self, wide_volume):
        wide_volume.get_spectral_decomposition_inline(384, method="swt", level=1)
        cache_key = (384, "swt", "sym8")
        assert cache_key in wide_volume._spectral_cache
        cached_energy = wide_volume._spectral_cache[cache_key]["energy"]
        # Switching levels must reuse the cached all-levels array, not
        # recompute pywt.swt from scratch.
        wide_volume.get_spectral_decomposition_inline(384, method="swt", level=5)
        assert wide_volume._spectral_cache[cache_key]["energy"] is cached_energy

    def test_different_wavelets_cached_separately(self, wide_volume):
        wide_volume.get_spectral_decomposition_inline(384, method="swt", wavelet="sym8")
        wide_volume.get_spectral_decomposition_inline(384, method="swt", wavelet="coif3")
        assert (384, "swt", "sym8") in wide_volume._spectral_cache
        assert (384, "swt", "coif3") in wide_volume._spectral_cache

    def test_trace_decomposition_returns_all_levels(self, wide_volume):
        result = wide_volume.get_spectral_decomposition_trace(384, 47, method="swt")
        assert result["levels"] == [1, 2, 3, 4, 5, 6]
        assert len(result["bands_hz"]) == 6
        n_time = len(result["time_ms"])
        assert len(result["energy"]) == n_time
        assert all(len(row) == 6 for row in result["energy"])

    def test_short_trace_falls_back_to_edge_padding(self, tmp_path):
        # The base fixture's 6-sample trace is far shorter than 2**6=64,
        # so reflect-padding (which can't exceed the signal's own length)
        # isn't possible -- must fall back to edge-padding rather than
        # raising.
        path = tmp_path / "short_survey.sgy"
        _write_synthetic_segy(path)  # default N_SAMPLES = 6
        volume = sp.SegyVolume(path)
        result = volume.get_spectral_decomposition_inline(384, method="swt", level=1)
        assert len(result["time_ms"]) == N_SAMPLES
        assert all(len(row) == len(result["amplitude"][0]) for row in result["amplitude"])

    def test_nan_trace_does_not_crash(self, wide_volume):
        # Mirrors how STFT/CWT are exercised -- no special NaN masking,
        # just confirms it propagates without raising.
        volume = wide_volume
        idx = volume._inline_index[384]
        volume._traces[idx[0], 5] = np.nan
        result = volume.get_spectral_decomposition_inline(384, method="swt")
        assert len(result["amplitude"]) > 0

    def test_unknown_method_still_raises(self, wide_volume):
        with pytest.raises(sp.SegyVolumeError):
            wide_volume.get_spectral_decomposition_inline(384, method="bogus")


class TestSegyVolumeSingleton:
    def test_no_file_raises_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
        sp._volume_cache.clear()
        with pytest.raises(sp.SegyFileNotFoundError):
            sp.get_segy_volume()

    def test_discovers_and_caches_single_file(self, tmp_path, monkeypatch):
        path = tmp_path / "survey.sgy"
        _write_synthetic_segy(path)
        monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
        sp._volume_cache.clear()
        try:
            vol1 = sp.get_segy_volume()
            vol2 = sp.get_segy_volume()
            assert vol1 is vol2  # cached, not re-opened
        finally:
            sp._volume_cache.clear()
