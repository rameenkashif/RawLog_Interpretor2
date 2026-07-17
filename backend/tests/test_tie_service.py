"""
test_tie_service.py
---------------------
Integration tests for services/tie_service.py's tie-strategy selection:
prefer a real spatial nearest-trace match when both the well (LAS header)
and the seismic dataset (trace headers) carry coordinates, otherwise fall
back to the manually configured trace_index in tie_config.yaml.

Uses real LAS bytes (Z-02_raw.las, whose X/Y/KB/TD header fields are
unit-standardized from feet to meters on load -- see test_las_loader.py's
TestUnitStandardization) and a synthetic SEG-Y file written with segyio so
the whole pipeline (loaders -> repositories -> tie_service) is exercised,
without touching the shared backend/data/processed or
backend/data/seismic_processed caches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app.repository import FileWellRepository
from app.seismic_repository import FileSeismicRepository
from app.services import seismic_service, tie_service, well_service

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"


def _write_segy(
    tmp_path: Path, headers: list[dict], n_samples: int = 400, dt_us: int = 2000, delay_ms: int = 2000
) -> bytes:
    # delay_ms defaults to 2000 so the synthetic seismic window (2000ms +
    # n_samples*dt_us/1000) brackets Z-02's real DPTM range (~2101-2173ms,
    # see data/raw/Z-02_raw.las) -- the tie search now uses each well's own
    # DPTM curve directly as an absolute time axis (see
    # well_seismic_tie.search_best_tie_full_window), so a synthetic test
    # SEG-Y window that doesn't overlap the real well's real time range
    # would never find a tie, regardless of trace-index/coordinate wiring.
    path = tmp_path / "survey.sgy"
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(n_samples)
    spec.tracecount = len(headers)
    rng = np.random.default_rng(11)
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = dt_us
        for i, hdr in enumerate(headers):
            merged = {segyio.TraceField.DelayRecordingTime: delay_ms, **hdr}
            f.header[i] = merged
            f.trace[i] = rng.normal(0, 1, n_samples).astype(np.float32)
    return path.read_bytes()


@pytest.fixture
def repos(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells"), FileSeismicRepository(
        base_dir=tmp_path / "seismic"
    )


@pytest.fixture
def loaded_well(repos):
    well_repo, _ = repos
    las_bytes = Z02_PATH.read_bytes()
    return well_service.process_and_store_las_bytes(las_bytes, "Z-02_raw.las", repo=well_repo)


def _patch_services(monkeypatch, well_repo, seismic_repo):
    monkeypatch.setattr(
        well_service, "get_well_summary",
        lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_curves",
        lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "list_well_summaries",
        lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
    )
    monkeypatch.setattr(
        seismic_service, "get_seismic_dataset",
        lambda dataset_id, repo=None, _f=seismic_service.get_seismic_dataset: _f(dataset_id, repo=seismic_repo),
    )


class TestTieMethodSelection:
    def test_uses_nearest_trace_when_both_sides_have_coordinates(
        self, monkeypatch, repos, loaded_well
    ):
        well_repo, seismic_repo = repos
        _patch_services(monkeypatch, well_repo, seismic_repo)

        well_x, well_y = loaded_well.well_x, loaded_well.well_y
        assert well_x is not None and well_y is not None

        n_traces = 50
        xs = well_x - 1000 + np.arange(n_traces) * 40.0
        headers = [
            {
                segyio.TraceField.CDP_X: int(xs[i]),
                segyio.TraceField.CDP_Y: int(well_y),
                segyio.TraceField.SourceGroupScalar: 1,
            }
            for i in range(n_traces)
        ]
        segy_bytes = _write_segy(seismic_repo.base_dir, headers)
        summary = seismic_service.process_and_store_segy_bytes(
            segy_bytes, "coords.sgy", repo=seismic_repo
        )

        result = tie_service.get_well_seismic_tie(loaded_well.well_id, summary.dataset_id)

        assert result.tie_method == "nearest_trace"
        assert result.geometry_warning is None
        assert result.distance_m is not None
        assert result.distance_m < 40.0  # within half the trace spacing

    def test_falls_back_to_manual_override_when_dataset_has_no_coordinates(
        self, monkeypatch, repos, loaded_well
    ):
        well_repo, seismic_repo = repos
        _patch_services(monkeypatch, well_repo, seismic_repo)

        # tie_config.yaml has a Z-02_RAW override at trace_index 5000, so use
        # enough traces for that index to be valid.
        n_traces = 6000
        headers = [{} for _ in range(n_traces)]
        segy_bytes = _write_segy(seismic_repo.base_dir, headers, n_samples=150)
        summary = seismic_service.process_and_store_segy_bytes(
            segy_bytes, "no_coords.sgy", repo=seismic_repo
        )

        result = tie_service.get_well_seismic_tie(loaded_well.well_id, summary.dataset_id)

        assert result.tie_method == "manual_override"
        assert result.distance_m is None
        assert result.trace_index == 5000
        assert result.geometry_warning is not None


class TestTieResultFields:
    def test_response_carries_full_window_search_fields(self, monkeypatch, repos, loaded_well):
        well_repo, seismic_repo = repos
        _patch_services(monkeypatch, well_repo, seismic_repo)

        well_x, well_y = loaded_well.well_x, loaded_well.well_y
        n_traces = 50
        xs = well_x - 1000 + np.arange(n_traces) * 40.0
        headers = [
            {
                segyio.TraceField.CDP_X: int(xs[i]),
                segyio.TraceField.CDP_Y: int(well_y),
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.INLINE_3D: 100,
                segyio.TraceField.CROSSLINE_3D: 200 + i,
            }
            for i in range(n_traces)
        ]
        segy_bytes = _write_segy(seismic_repo.base_dir, headers)
        summary = seismic_service.process_and_store_segy_bytes(
            segy_bytes, "coords2.sgy", repo=seismic_repo
        )

        result = tie_service.get_well_seismic_tie(loaded_well.well_id, summary.dataset_id)

        assert result.polarity in (1, -1)
        assert result.best_freq_hz > 0
        assert -100.0 <= result.bulk_shift_ms <= 100.0
        assert result.max_shift_ms == 100.0
        assert result.n_used > 0
        assert len(result.time_ms) == len(result.synthetic_amplitude) == len(result.seismic_amplitude) == len(result.reflectivity)
        assert result.inline == 100
        assert result.crossline is not None


class TestBatchTie:
    @pytest.fixture
    def loaded_wells(self, repos):
        well_repo, _ = repos
        summaries = []
        for name in ("Z-02_raw.las", "Z-03_raw.las"):
            las_bytes = (RAW_LAS_DIR / name).read_bytes()
            summaries.append(well_service.process_and_store_las_bytes(las_bytes, name, repo=well_repo))
        return summaries

    def test_batch_tie_covers_every_well(self, monkeypatch, repos, loaded_wells):
        well_repo, seismic_repo = repos
        _patch_services(monkeypatch, well_repo, seismic_repo)

        # No trace coordinates -> falls back to tie_config.yaml's manual
        # overrides (Z-02_RAW: 5000, Z-03_RAW: 15000), so enough traces are
        # needed for both indices to be valid. A 2000-2500ms window covers
        # both wells' real DPTM ranges (Z-02 ~2101-2173ms, Z-03 ~2249-2331ms).
        n_traces = 15100
        headers = [{segyio.TraceField.SourceGroupScalar: 1} for _ in range(n_traces)]
        segy_bytes = _write_segy(seismic_repo.base_dir, headers, n_samples=250)
        summary = seismic_service.process_and_store_segy_bytes(
            segy_bytes, "batch.sgy", repo=seismic_repo
        )

        batch = tie_service.get_all_well_ties(summary.dataset_id)

        assert batch.dataset_id == summary.dataset_id
        well_ids = {row.well_id for row in batch.rows}
        assert well_ids == {w.well_id for w in loaded_wells}
        for row in batch.rows:
            assert row.error is None
            assert row.correlation is not None
            assert row.best_freq_hz is not None
            assert row.tie_method == "manual_override"  # no seismic trace coordinates in this test

    def test_batch_tie_reports_error_row_instead_of_dropping(self, monkeypatch, repos, loaded_wells):
        well_repo, seismic_repo = repos
        _patch_services(monkeypatch, well_repo, seismic_repo)

        # Tiny window far from any real well's DPTM range -- every well's
        # tie search should fail, but still show up as an error row.
        headers = [{segyio.TraceField.SourceGroupScalar: 1} for _ in range(200)]
        segy_bytes = _write_segy(seismic_repo.base_dir, headers, n_samples=10, delay_ms=0)
        summary = seismic_service.process_and_store_segy_bytes(
            segy_bytes, "tiny.sgy", repo=seismic_repo
        )

        batch = tie_service.get_all_well_ties(summary.dataset_id)

        assert len(batch.rows) == len(loaded_wells)
        for row in batch.rows:
            assert row.error is not None
            assert row.correlation is None
        assert len(batch.warnings) == len(loaded_wells)
