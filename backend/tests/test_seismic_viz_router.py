"""
test_seismic_viz_router.py
-----------------------------
HTTP-level tests for routers/seismic_viz.py (/api/seismic/*), using
FastAPI's TestClient against a small synthetic SEG-Y file (see
test_seismic_processor.py for why the file is synthetic and how its
header layout mirrors the real one) so the routing/response-model/
error-mapping layer is exercised, not just the SegyVolume class directly.
"""

from __future__ import annotations

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from fastapi.testclient import TestClient

from app.services import seismic_processor as sp
from tests.test_seismic_processor import _write_synthetic_segy


@pytest.fixture
def client(tmp_path, monkeypatch):
    _write_synthetic_segy(tmp_path / "survey.sgy")
    monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
    sp._volume_cache.clear()

    import main  # imported here so the monkeypatched RAW_SEISMIC_DIR is in effect before any request

    with TestClient(main.app) as c:
        yield c

    sp._volume_cache.clear()


class TestSurveyInfo:
    def test_returns_geometry(self, client):
        resp = client.get("/api/seismic/survey-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_traces"] == 20
        assert body["inline_min"] == 382
        assert body["crossline_max"] == 49


class TestSections:
    def test_inline_section_ok(self, client):
        resp = client.get("/api/seismic/inline/384")
        assert resp.status_code == 200
        body = resp.json()
        assert body["inline_number"] == 384
        assert len(body["crossline_axis"]) == 4

    def test_inline_section_out_of_range_is_422(self, client):
        resp = client.get("/api/seismic/inline/9999")
        assert resp.status_code == 422

    def test_crossline_section_ok(self, client):
        resp = client.get("/api/seismic/crossline/47")
        assert resp.status_code == 200
        assert resp.json()["crossline_number"] == 47


class TestTimeSlice:
    def test_boundary_clamp_via_http(self, client):
        resp = client.get("/api/seismic/timeslice", params={"time_ms": 2029})
        assert resp.status_code == 200
        body = resp.json()
        assert body["time_ms"] == 2030.0
        assert body["requested_time_ms"] == 2029.0


class TestSpectrum:
    def test_whole_volume(self, client):
        resp = client.get("/api/seismic/spectrum")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_traces_sampled"] == 20

    def test_bad_inline_is_422(self, client):
        resp = client.get("/api/seismic/spectrum", params={"inline_number": 9999})
        assert resp.status_code == 422


class TestWellTieEndpoint:
    def test_unknown_well_is_404(self, client):
        resp = client.get("/api/seismic/well-tie/DOES_NOT_EXIST")
        assert resp.status_code == 404

    def test_no_segy_file_is_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
        sp._volume_cache.clear()
        import main

        with TestClient(main.app) as c:
            resp = c.get("/api/seismic/survey-info")
        assert resp.status_code == 404
        sp._volume_cache.clear()
