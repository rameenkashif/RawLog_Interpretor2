"""
test_synthetic_router.py
---------------------------
HTTP-level tests for routers/synthetic.py (/api/synthetic/*), using
FastAPI's TestClient. Reuses the same synthetic-SEG-Y + coordinate-patching
helpers as test_synthetic_seismogram_service.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from fastapi.testclient import TestClient

from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import synthetic_seismogram_service as sss
from app.services import well_service
from app.synthetic_tie_repository import FileSyntheticTieRepository
from tests.test_seismic_processor import _set_las_coordinate_meters, _write_synthetic_segy

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"
ALIGNED_X = 366840.0
ALIGNED_Y = 2950275.0


@pytest.fixture
def client(tmp_path, monkeypatch):
    _write_synthetic_segy(tmp_path / "survey.sgy", n_samples=80, interval_ms=2)
    monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
    sp._volume_cache.clear()

    well_repo = FileWellRepository(base_dir=tmp_path / "wells")
    tie_repo = FileSyntheticTieRepository(base_dir=tmp_path / "ties")

    las_text = Z02_PATH.read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", ALIGNED_X)
    las_text = _set_las_coordinate_meters(las_text, "Y", ALIGNED_Y)
    well_service.process_and_store_las_bytes(las_text.encode(), "Z-02_raw.las", repo=well_repo)

    monkeypatch.setattr(
        well_service, "get_well_summary",
        lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_curves",
        lambda well_id, repo=None, _f=well_service.get_well_curves: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(sss, "get_synthetic_tie_repository", lambda: tie_repo)

    import main

    with TestClient(main.app) as c:
        yield c

    sp._volume_cache.clear()


class TestGenerateEndpoint:
    def test_generate_ok(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/generate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["well_id"] == "Z-02_RAW"
        assert body["well_header"]["coordinate_unit_detected"] == "feet"
        assert len(body["seismic_twt_ms"]) == len(body["synthetic"])
        assert "correlation" in body

    def test_generate_with_query_params(self, client):
        resp = client.get(
            "/api/synthetic/Z-02_RAW/generate",
            params={"wavelet_method": "ricker", "wavelet_freq_hz": 30, "density_method": "gardner"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["wavelet_method"] == "ricker"
        assert body["density_method"] == "gardner"
        assert body["gardner_coefficients"] is not None

    def test_auto_optimize_tie_ok(self, client):
        resp = client.get(
            "/api/synthetic/Z-02_RAW/generate",
            params={"wavelet_method": "ricker", "wavelet_freq_hz": 25, "auto_optimize_tie": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["auto_optimize_tie"] is True
        assert body["polarity"] in (1, -1)
        assert body["tie_search_note"] is not None

    def test_auto_optimize_tie_default_off(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/generate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["auto_optimize_tie"] is False
        assert body["tie_search_note"] is None

    def test_unknown_well_is_404(self, client):
        resp = client.get("/api/synthetic/DOES_NOT_EXIST/generate")
        assert resp.status_code == 404

    def test_bad_density_method_is_422(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/generate", params={"density_method": "bogus"})
        assert resp.status_code == 422

    def test_no_segy_file_is_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
        sp._volume_cache.clear()
        import main

        with TestClient(main.app) as c:
            resp = c.get("/api/synthetic/Z-02_RAW/generate")
        assert resp.status_code == 404
        sp._volume_cache.clear()


class TestNearestTraceEndpoint:
    def test_ok(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/nearest-trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["well_id"] == "Z-02_RAW"
        assert body["distance_m"] >= 0


class TestTiePointsEndpoints:
    def test_get_before_save_returns_null(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/tie")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_save_and_get_round_trip(self, client):
        put_resp = client.put(
            "/api/synthetic/Z-02_RAW/tie",
            json={
                "points": [{"md_m": 3500.0, "time_shift_ms": 5.0}],
                "wavelet_method": "ricker",
                "wavelet_freq_hz": 20.0,
            },
        )
        assert put_resp.status_code == 200
        assert len(put_resp.json()["points"]) == 1

        get_resp = client.get("/api/synthetic/Z-02_RAW/tie")
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body["points"][0]["md_m"] == 3500.0
        assert body["wavelet_freq_hz"] == 20.0

    def test_delete(self, client):
        client.put("/api/synthetic/Z-02_RAW/tie", json={"points": [{"md_m": 1.0, "time_shift_ms": 1.0}]})
        del_resp = client.delete("/api/synthetic/Z-02_RAW/tie")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        get_resp = client.get("/api/synthetic/Z-02_RAW/tie")
        assert get_resp.json() is None


class TestExportEndpoint:
    def test_export_returns_csv(self, client):
        resp = client.get("/api/synthetic/Z-02_RAW/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers["content-disposition"]
        text = resp.text
        assert "Synthetic seismogram tie report" in text
        assert "twt_ms,synthetic,shifted_synthetic,real_trace" in text
