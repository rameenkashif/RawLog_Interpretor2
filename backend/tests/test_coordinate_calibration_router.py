"""
test_coordinate_calibration_router.py
------------------------------------------
Router-level tests for the /api/seismic/coordinate-calibration* endpoints
(coordinate calibration report, recalibrate, manual tie-point overrides
CRUD) -- fixes #4/#5 in the calibration audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

segyio = pytest.importorskip("segyio")

from fastapi.testclient import TestClient

import main
from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import well_service
from tests.test_seismic_processor import _set_las_coordinate_meters, _write_synthetic_segy

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Test survey extent from _write_synthetic_segy's defaults (see
# test_coordinate_calibration_service.py for the derivation): X
# 366820-366860, Y 2950260-2950290.
WELL_A_XY = (366825.0, 2950262.0)
WELL_B_XY = (366858.0, 2950288.0)


@pytest.fixture
def well_repo(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells")


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path)
    return sp.SegyVolume(path)


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch, well_repo, volume):
    monkeypatch.setattr(
        well_service, "list_well_summaries",
        lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_summary",
        lambda well_id, repo=None, _f=well_service.get_well_summary: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(sp, "get_segy_volume", lambda refresh=False: volume)


def _load_well(well_repo, las_filename: str, xy: tuple[float, float]):
    las_text = (RAW_LAS_DIR / las_filename).read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", xy[0])
    las_text = _set_las_coordinate_meters(las_text, "Y", xy[1])
    return well_service.process_and_store_las_bytes(las_text.encode(), las_filename, repo=well_repo)


class TestCalibrationReportEndpoint:
    def test_returns_report(self, well_repo):
        _load_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        _load_well(well_repo, "Z-03_raw.las", WELL_B_XY)

        client = TestClient(main.app)
        response = client.get("/api/seismic/coordinate-calibration")
        assert response.status_code == 200
        body = response.json()
        assert len(body["wells"]) == 2
        assert "method_note" in body


class TestRecalibrateEndpoint:
    def test_recalibrate_with_all_wells(self, well_repo):
        _load_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        _load_well(well_repo, "Z-03_raw.las", WELL_B_XY)

        client = TestClient(main.app)
        response = client.post("/api/seismic/coordinate-calibration/recalibrate", json={})
        assert response.status_code == 200
        body = response.json()
        assert len(body["well_ids_used"]) == 2

    def test_recalibrate_with_curated_subset(self, well_repo):
        w1 = _load_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        w2 = _load_well(well_repo, "Z-03_raw.las", WELL_B_XY)

        client = TestClient(main.app)
        response = client.post(
            "/api/seismic/coordinate-calibration/recalibrate",
            json={"well_ids": [w1.well_id, w2.well_id]},
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body["well_ids_used"]) == {w1.well_id, w2.well_id}


class TestOverrideEndpoints:
    def test_save_list_delete_override(self, well_repo):
        _load_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        client = TestClient(main.app)

        save_resp = client.put(
            "/api/seismic/coordinate-calibration/overrides/Z-99",
            json={"inline": 384, "crossline": 47, "note": "confirmed by geologist"},
        )
        assert save_resp.status_code == 200
        assert save_resp.json()["inline"] == 384

        list_resp = client.get("/api/seismic/coordinate-calibration/overrides")
        assert list_resp.status_code == 200
        assert any(o["well_id"] == "Z-99" for o in list_resp.json())

        delete_resp = client.delete("/api/seismic/coordinate-calibration/overrides/Z-99")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

        list_resp_2 = client.get("/api/seismic/coordinate-calibration/overrides")
        assert not any(o["well_id"] == "Z-99" for o in list_resp_2.json())
