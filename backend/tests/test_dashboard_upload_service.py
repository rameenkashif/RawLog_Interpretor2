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


class TestSyntheticTieSearchRigor:
    """generate()'s plain defaults (statistical wavelet, no search at
    all) are weaker than they could be -- both call sites opt into
    auto_optimize_tie=True for a polarity search on top of the same
    statistical wavelet. wavelet_method is deliberately left at its
    default ("statistical"), NOT forced to "ricker": that was tried
    first and made real results WORSE (lower correlation, more
    boundary-pinned wells), because a statistically-extracted wavelet's
    trace-matching advantage outweighs a frequency/polarity search over
    a generic Ricker shape that may not fit this trace at all. Polarity
    search against the SAME (statistical) wavelet, by contrast, is a
    strictly monotonic improvement -- it can only find a correlation >=
    what always-assuming +1 polarity gets. See README.md/AGENT_BRIEF.md
    for the full story, including the reverted wavelet_method="ricker"
    attempt."""

    def test_run_upload_pipeline_requests_auto_optimize_with_statistical_wavelet(self, monkeypatch, raw_seismic_dir):
        _patch_pipeline_deps(monkeypatch)
        from app.services import synthetic_seismogram_service

        calls = []

        def _spy(well_id, **kwargs):
            calls.append(kwargs)
            return _fake_synthetic()

        monkeypatch.setattr(synthetic_seismogram_service, "generate", _spy)

        token = dus.start_upload("Z-02", "survey.sgy")
        dus.run_upload_pipeline("Z-02", token, b"x", "survey.sgy")

        assert len(calls) == 1
        assert calls[0] == {"auto_optimize_tie": True}

    def test_get_synthetic_summary_live_fallback_requests_auto_optimize_with_statistical_wavelet(
        self, monkeypatch, raw_seismic_dir
    ):
        from app.services import synthetic_seismogram_service

        calls = []

        def _spy(well_id, **kwargs):
            calls.append(kwargs)
            return _fake_synthetic()

        monkeypatch.setattr(synthetic_seismogram_service, "generate", _spy)

        dus.get_synthetic_summary("Z-09")

        assert len(calls) == 1
        assert calls[0] == {"auto_optimize_tie": True}


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


class TestFieldOverview:
    """get_field_overview -- the cross-well tool backing the agent's
    reasoning-workflow guidance (see anthropic_agent.SYSTEM_PROMPT) so a
    ranking/comparison question doesn't require looping per-well tool
    calls."""

    @pytest.fixture
    def well_repo(self, tmp_path):
        from app.repository import FileWellRepository

        return FileWellRepository(base_dir=tmp_path / "wells")

    @pytest.fixture
    def loaded_wells(self, well_repo):
        from pathlib import Path

        from app.services import well_service

        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        summaries = []
        for name in ("Z-02_raw.las", "Z-03_raw.las"):
            las_bytes = (raw_dir / name).read_bytes()
            summaries.append(well_service.process_and_store_las_bytes(las_bytes, name, repo=well_repo))
        return summaries

    def _patch_well_service(self, monkeypatch, well_repo):
        from app.services import well_service

        monkeypatch.setattr(
            well_service, "list_well_summaries",
            lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
        )
        monkeypatch.setattr(
            well_service, "get_well_zones",
            lambda well_id, repo=None, _f=well_service.get_well_zones: _f(well_id, repo=well_repo),
        )

    def test_returns_every_loaded_well_with_pay_zone_data(self, monkeypatch, well_repo, loaded_wells):
        self._patch_well_service(monkeypatch, well_repo)
        monkeypatch.setattr(dus, "seismic_deps_available", lambda: False)

        result = dus.get_field_overview()

        well_ids = {w["well_id"] for w in result["wells"]}
        assert well_ids == {w.well_id for w in loaded_wells}
        for well in result["wells"]:
            assert well["pay_zone"] is None or "thickness_m" in well["pay_zone"]
            assert well["net_pay_thickness"] is not None

    def test_omits_tie_fields_when_seismic_unavailable(self, monkeypatch, well_repo, loaded_wells):
        self._patch_well_service(monkeypatch, well_repo)
        monkeypatch.setattr(dus, "seismic_deps_available", lambda: False)

        result = dus.get_field_overview()

        for well in result["wells"]:
            assert well["tie"] is None
            assert well["tie_error"] == "Seismic module unavailable."
            assert well["synthetic"] is None
            assert well["synthetic_error"] == "Seismic module unavailable."

    def test_includes_tie_and_synthetic_when_available(self, monkeypatch, well_repo, loaded_wells):
        self._patch_well_service(monkeypatch, well_repo)
        monkeypatch.setattr(dus, "seismic_deps_available", lambda: True)
        monkeypatch.setattr(
            dus, "get_tie_summary", lambda well_id: {"well_id": well_id, "correlation": 0.8, "low_confidence": False}
        )
        monkeypatch.setattr(
            dus, "get_synthetic_summary", lambda well_id: {"correlation": 0.75, "low_confidence": False}
        )

        result = dus.get_field_overview()

        for well in result["wells"]:
            assert well["tie"]["correlation"] == 0.8
            assert well["tie_error"] is None
            assert well["synthetic"]["correlation"] == 0.75
            assert well["synthetic_error"] is None

    def test_one_well_failing_tie_does_not_drop_it_or_other_wells(self, monkeypatch, well_repo, loaded_wells):
        self._patch_well_service(monkeypatch, well_repo)
        monkeypatch.setattr(dus, "seismic_deps_available", lambda: True)

        def _tie(well_id):
            if well_id == loaded_wells[0].well_id:
                return {"error": "no coordinates available"}
            return {"correlation": 0.9, "low_confidence": False}

        monkeypatch.setattr(dus, "get_tie_summary", _tie)
        monkeypatch.setattr(dus, "get_synthetic_summary", lambda well_id: {"correlation": 0.9, "low_confidence": False})

        result = dus.get_field_overview()

        assert len(result["wells"]) == len(loaded_wells)
        by_id = {w["well_id"]: w for w in result["wells"]}
        assert by_id[loaded_wells[0].well_id]["tie"] is None
        assert by_id[loaded_wells[0].well_id]["tie_error"] == "no coordinates available"
        assert by_id[loaded_wells[1].well_id]["tie"]["correlation"] == 0.9

    def test_zone_lookup_failure_is_isolated_per_well(self, monkeypatch, well_repo, loaded_wells):
        from app.services import well_service

        monkeypatch.setattr(
            well_service, "list_well_summaries",
            lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
        )

        def _broken_zones(well_id, repo=None):
            raise RuntimeError("zone computation exploded")

        monkeypatch.setattr(well_service, "get_well_zones", _broken_zones)
        monkeypatch.setattr(dus, "seismic_deps_available", lambda: False)

        result = dus.get_field_overview()

        assert len(result["wells"]) == len(loaded_wells)
        for well in result["wells"]:
            assert well["pay_zone"] is None
            assert well["zone_error"] == "zone computation exploded"
