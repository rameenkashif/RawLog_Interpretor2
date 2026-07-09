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

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import well_service

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


def _write_synthetic_segy(path: Path, source_x_base: float = 363000.0, source_y_base: float = 2949800.0) -> None:
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(N_SAMPLES) * INTERVAL_MS + DELAY_MS
    spec.tracecount = len(INLINES) * len(CROSSLINES)

    rng = np.random.default_rng(42)
    i = 0
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = INTERVAL_MS * 1000
        for il in INLINES:
            for xl in CROSSLINES:
                f.header[i] = {
                    segyio.TraceField.FieldRecord: il,
                    segyio.TraceField.TraceNumber: xl,
                    segyio.TraceField.SourceX: int(source_x_base + il * 10),
                    segyio.TraceField.SourceY: int(source_y_base + xl * 10),
                    segyio.TraceField.DelayRecordingTime: DELAY_MS,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: INTERVAL_MS * 1000,
                }
                # Distinct-ish waveform per trace so section/spectrum tests
                # have something non-degenerate to check.
                f.trace[i] = (
                    np.sin(np.linspace(0, 2 * np.pi, N_SAMPLES)) + rng.normal(0, 0.05, N_SAMPLES)
                ).astype(np.float32)
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
        assert info.n_inlines == 5 and info.n_crosslines == 4
        assert info.twt_start_ms == 2030.0
        assert info.sample_interval_ms == 2.0

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
        # Z-02's LAS coordinates are a placeholder "field local grid" far
        # from this synthetic survey's SourceX/Y range -- get_well_tie must
        # flag this as a likely CRS mismatch rather than silently tying to
        # whatever the nearest (very distant) trace happens to be.
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
        # Patch the LAS bytes' XWELL/YWELL to land inside this synthetic
        # survey's coordinate extent before loading, simulating a well
        # whose coordinates really are in the same CRS as the seismic.
        las_text = Z02_PATH.read_text()
        las_text = las_text.replace("XWELL.m    512340.00", "XWELL.m    366840.00")
        las_text = las_text.replace("YWELL.m   6543210.00", "YWELL.m   2950275.00")
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
        las_text = las_text.replace("XWELL.m    512340.00", "XWELL.m    366840.00")
        las_text = las_text.replace("YWELL.m   6543210.00", "YWELL.m   2950275.00")
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
