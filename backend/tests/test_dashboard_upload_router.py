"""
test_dashboard_upload_router.py
-----------------------------------
HTTP-level tests for the new POST /dashboard/upload and
GET /dashboard/upload/{well_id}/status endpoints (routers/dashboard.py).

Monkeypatches well_service.process_and_store_las_bytes and
dashboard_upload_service's scheduling entry points rather than exercising
a real LAS/SEG-Y round trip through the default (real-directory) well
repository singleton -- routers/wells.py's own /wells/upload endpoint has
no dedicated HTTP-level test in this suite for the same reason (it writes
through get_repository()'s real singleton, same as this new endpoint
does). Orchestration correctness (status transitions, low_confidence
thresholding, run_token races) is covered end-to-end at the service layer
in test_dashboard_upload_service.py; this file targets the router's own
new logic: request validation, response shape, and status lookup.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from app.models.schemas import WellSummary
from app.services import dashboard_upload_service, well_service
from app.well_processing_cache_repository import (
    WellProcessingCacheRecord,
    get_well_processing_cache_repository,
)


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _fake_well_summary(well_id="Z-02_RAW") -> WellSummary:
    return WellSummary(
        well_id=well_id,
        well_name=well_id,
        start_depth=1000.0,
        stop_depth=1100.0,
        step=0.1524,
        n_samples=100,
        footage_logged=100.0,
    )


class TestDashboardUploadValidation:
    def test_rejects_non_las_well_file(self, client):
        resp = client.post(
            "/dashboard/upload",
            files={
                "las_file": ("well.txt", b"not a las file", "text/plain"),
                "segy_file": ("survey.sgy", b"fake", "application/octet-stream"),
            },
        )
        assert resp.status_code == 422

    def test_rejects_non_segy_seismic_file(self, client):
        resp = client.post(
            "/dashboard/upload",
            files={
                "las_file": ("well.las", b"~V\n", "text/plain"),
                "segy_file": ("survey.txt", b"fake", "text/plain"),
            },
        )
        assert resp.status_code == 422


class TestDashboardUploadSuccess:
    def test_schedules_background_pipeline_and_returns_processing(self, client, monkeypatch):
        monkeypatch.setattr(
            well_service, "process_and_store_las_bytes", lambda *a, **k: _fake_well_summary()
        )
        monkeypatch.setattr(dashboard_upload_service, "seismic_deps_available", lambda: True)
        monkeypatch.setattr(dashboard_upload_service, "start_upload", lambda well_id, filename: "tok-123")

        scheduled = {}

        def _fake_run_pipeline(well_id, run_token, segy_bytes, segy_filename):
            scheduled["args"] = (well_id, run_token, segy_filename)

        monkeypatch.setattr(dashboard_upload_service, "run_upload_pipeline", _fake_run_pipeline)

        resp = client.post(
            "/dashboard/upload",
            files={
                "las_file": ("Z-02_raw.las", b"~V\n", "text/plain"),
                "segy_file": ("survey.sgy", b"fake-segy-bytes", "application/octet-stream"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["well_id"] == "Z-02_RAW"
        assert body["status"] == "processing"
        assert body["well_summary"]["well_id"] == "Z-02_RAW"
        # TestClient runs BackgroundTasks before returning the response.
        assert scheduled["args"] == ("Z-02_RAW", "tok-123", "survey.sgy")

    def test_503_when_seismic_deps_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(
            well_service, "process_and_store_las_bytes", lambda *a, **k: _fake_well_summary()
        )
        monkeypatch.setattr(dashboard_upload_service, "seismic_deps_available", lambda: False)

        resp = client.post(
            "/dashboard/upload",
            files={
                "las_file": ("Z-02_raw.las", b"~V\n", "text/plain"),
                "segy_file": ("survey.sgy", b"fake", "application/octet-stream"),
            },
        )
        assert resp.status_code == 503


class TestDashboardUploadStatus:
    def test_404_for_unknown_well(self, client):
        resp = client.get("/dashboard/upload/DOES_NOT_EXIST/status")
        assert resp.status_code == 404

    def test_returns_stored_record(self, client, monkeypatch):
        monkeypatch.setattr(dashboard_upload_service, "is_active_volume_stale", lambda record: False)
        repo = get_well_processing_cache_repository()
        repo.save(
            WellProcessingCacheRecord(
                well_id="Z-02_RAW",
                status="ready",
                run_token="tok-1",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:01+00:00",
                dataset_id="DS-1",
                segy_filename="survey.sgy",
                tie_available=True,
                tie_correlation=0.12,
                tie_boundary_pinned=False,
                tie_low_confidence=True,
            )
        )

        resp = client.get("/dashboard/upload/Z-02_RAW/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["tie_available"] is True
        assert body["tie_correlation"] == 0.12
        assert body["tie_low_confidence"] is True
        assert body["stale"] is False
