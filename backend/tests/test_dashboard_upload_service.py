"""
test_dashboard_upload_service.py
------------------------------------
Unit tests for services/dashboard_upload_service.py's orchestration logic:
status transitions, low_confidence thresholding, the run_token
compare-and-swap on a same-well re-upload race, and the cache-first/
live-fallback summary readers used by the new agent tools.

Mocks seismic_service/tie_service/synthetic_seismogram_service/
seismic_processor directly rather than exercising real LAS/SEG-Y parsing --
each of those already has its own dedicated test suite (test_tie_service.py,
test_synthetic_seismogram_service.py, test_seismic_processor.py); this
module's own job is orchestration (sequencing, caching, flagging), which is
what these tests target.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import dashboard_upload_service as dus
from app.well_processing_cache_repository import get_well_processing_cache_repository


class _FakeSegySummary:
    dataset_id = "DS-1"


class _FakeVolume:
    def __init__(self, spectrum=None, raise_on_spectrum=False):
        self._spectrum = spectrum or {"dominant_freq_hz": 25.0, "bandwidth_hz": 10.0, "snr_proxy": 3.2}
        self._raise = raise_on_spectrum

    def get_amplitude_spectrum(self, inline_number=None):
        if self._raise:
            raise RuntimeError("spectrum computation failed")
        return self._spectrum


def _fake_tie(correlation=0.9, boundary_pinned=False, inline=100):
    return SimpleNamespace(
        correlation=correlation,
        boundary_pinned=boundary_pinned,
        best_freq_hz=30.0,
        polarity=1,
        bulk_shift_ms=5.0,
        distance_m=12.0,
        trace_index=42,
        inline=inline,
        crossline=200,
    )


def _fake_synthetic(correlation=0.9, boundary_pinned=False, datum_plausible=True):
    return {
        "correlation": correlation,
        "boundary_pinned": boundary_pinned,
        "datum_check": {"plausible": datum_plausible},
        "washout_flag": [False, False, True, False],
        "polarity": 1,
        "best_shift_ms": 3.0,
    }


@pytest.fixture
def raw_seismic_dir(tmp_path, monkeypatch):
    from app.services import seismic_processor as sp

    d = tmp_path / "seismic_raw"
    d.mkdir()
    monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", d)
    return d


def _patch_pipeline_deps(
    monkeypatch,
    segy_summary=_FakeSegySummary(),
    segy_error=None,
    volume=None,
    tie=None,
    tie_error=None,
    synthetic=None,
    synthetic_error=None,
):
    from app.services import seismic_processor as sp
    from app.services import seismic_service, synthetic_seismogram_service, tie_service

    def _process_segy(segy_bytes, filename):
        if segy_error is not None:
            raise segy_error
        return segy_summary

    monkeypatch.setattr(seismic_service, "process_and_store_segy_bytes", _process_segy)
    monkeypatch.setattr(sp, "get_segy_volume", lambda refresh=False: volume or _FakeVolume())

    def _get_tie(well_id, dataset_id):
        if tie_error is not None:
            raise tie_error
        return tie or _fake_tie()

    monkeypatch.setattr(tie_service, "get_well_seismic_tie", _get_tie)

    def _generate(well_id, **kwargs):
        if synthetic_error is not None:
            raise synthetic_error
        return synthetic or _fake_synthetic()

    monkeypatch.setattr(synthetic_seismogram_service, "generate", _generate)


class TestRunUploadPipelineSuccess:
    def test_full_success_marks_ready_with_all_sections_available(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch)
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"fake-bytes", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.status == "ready"
        assert record.error is None
        assert record.dataset_id == "DS-1"
        assert record.tie_available is True
        assert record.tie_correlation == 0.9
        assert record.tie_low_confidence is False
        assert record.synthetic_available is True
        assert record.synthetic_low_confidence is False
        assert record.spectral_available is True
        assert record.spectral_dominant_freq_hz == 25.0
        # The uploaded file becomes the active volume on disk.
        assert (raw_seismic_dir / "survey.sgy").read_bytes() == b"fake-bytes"

    def test_low_correlation_flags_low_confidence(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch, tie=_fake_tie(correlation=0.15))
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.tie_correlation == 0.15
        assert record.tie_low_confidence is True

    def test_boundary_pinned_flags_low_confidence_even_with_high_correlation(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch, tie=_fake_tie(correlation=0.95, boundary_pinned=True))
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.tie_low_confidence is True

    def test_prunes_stale_raw_segy_files(self, monkeypatch, raw_seismic_dir):
        (raw_seismic_dir / "old_survey.sgy").write_bytes(b"stale")
        _patch_pipeline_deps(monkeypatch)
        token = dus.start_upload("Z-02", "new_survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"new", "new_survey.sgy")

        remaining = sorted(p.name for p in raw_seismic_dir.glob("*.sgy"))
        assert remaining == ["new_survey.sgy"]


class TestRunUploadPipelinePartialFailure:
    def test_segy_validation_error_marks_whole_pipeline_failed(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch, segy_error=__import__("app.segy_loader", fromlist=["SegyValidationError"]).SegyValidationError("bad file"))
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.status == "failed"
        assert "bad file" in record.error
        assert record.tie_available is False

    def test_tie_error_does_not_abort_whole_pipeline(self, monkeypatch, raw_seismic_dir):
        from app.well_seismic_tie import TieError

        _patch_pipeline_deps(monkeypatch, tie_error=TieError("no coordinates available"))
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.status == "ready"
        assert record.tie_available is False
        assert "no coordinates available" in record.tie_error
        # Synthetic still ran even though tie failed.
        assert record.synthetic_available is True
        # No tie -> no inline to anchor a spectral summary to.
        assert record.spectral_available is False

    def test_synthetic_error_does_not_abort_whole_pipeline(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch, synthetic_error=RuntimeError("missing DT curve"))
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.status == "ready"
        assert record.tie_available is True
        assert record.synthetic_available is False
        assert "missing DT curve" in record.synthetic_error

    def test_unhandled_exception_marks_failed_instead_of_stuck_processing(self, monkeypatch, raw_seismic_dir):
        from app.services import seismic_service

        def _boom(*args, **kwargs):
            raise RuntimeError("totally unexpected")

        monkeypatch.setattr(seismic_service, "process_and_store_segy_bytes", _boom)
        token = dus.start_upload("Z-02", "survey.sgy")

        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        record = get_well_processing_cache_repository().get("Z-02")
        assert record.status == "failed"
        assert "totally unexpected" in record.error


class TestRunTokenRace:
    def test_stale_run_does_not_clobber_a_newer_upload(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch)
        repo = get_well_processing_cache_repository()

        stale_token = dus.start_upload("Z-02", "old.sgy")
        # A second, newer upload for the same well starts while the first
        # (stale_token) run is conceptually still "in flight" -- this
        # overwrites the stored run_token/segy_filename.
        newer_token = dus.start_upload("Z-02", "new.sgy")

        # The stale run now finishes and tries to write its result.
        dus.run_upload_pipeline("Z-02", stale_token, b"old-bytes", "old.sgy")

        record = repo.get("Z-02")
        # The stale run's writes must all have been no-ops (run_token
        # mismatch), so the record still reflects the newer run's token
        # and is not stuck reporting the older upload's filename/status.
        assert record.run_token == newer_token
        assert record.segy_filename == "new.sgy"


class TestSummaryReadersCacheFirst:
    def test_get_tie_summary_reads_from_cache_without_calling_live_service(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch)
        token = dus.start_upload("Z-02", "survey.sgy")
        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        from app.services import tie_service

        def _explode(*a, **k):
            raise AssertionError("should not recompute -- cache hit expected")

        monkeypatch.setattr(tie_service, "get_well_seismic_tie", _explode)

        summary = dus.get_tie_summary("Z-02")
        assert summary["correlation"] == 0.9
        assert summary["low_confidence"] is False

    def test_get_tie_summary_falls_back_live_for_uncached_well_and_writes_back(self, monkeypatch, raw_seismic_dir):
        from app.services import seismic_service, tie_service

        monkeypatch.setattr(
            seismic_service, "list_seismic_summaries", lambda: [SimpleNamespace(dataset_id="DS-9")]
        )
        monkeypatch.setattr(tie_service, "get_well_seismic_tie", lambda well_id, dataset_id: _fake_tie(correlation=0.6))

        summary = dus.get_tie_summary("Z-05")
        assert summary["correlation"] == 0.6
        assert summary["dataset_id"] == "DS-9"

        # Opportunistically cached for next time, even though Z-05 was
        # never uploaded through the dashboard pipeline.
        record = get_well_processing_cache_repository().get("Z-05")
        assert record is not None
        assert record.tie_available is True
        assert record.tie_correlation == 0.6

    def test_get_spectral_summary_unavailable_without_a_tie(self, monkeypatch, raw_seismic_dir):
        from app.services import seismic_service

        monkeypatch.setattr(seismic_service, "list_seismic_summaries", lambda: [])

        result = dus.get_spectral_summary("Z-99")
        assert result["available"] is False
        assert "error" in result
