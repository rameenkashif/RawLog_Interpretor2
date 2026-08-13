"""
test_sweet_spot_router.py
------------------------------
HTTP-level tests for routers/sweet_spot.py (/api/sweet-spot/*), using
FastAPI's TestClient. The underlying pipeline is monkeypatched at
get_or_train_cascade/predict_region -- routers/sweet_spot.py is thin
wiring only, already covered end-to-end by
test_sweet_spot_prediction_service.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import well_seismic_tie as wst
from app.services import sweet_spot_prediction_service as ssp
from app.services.well_service import WellNotFoundError


@pytest.fixture
def client():
    import main

    with TestClient(main.app) as c:
        yield c


class TestTrainEndpoint:
    def test_ok_validated(self, client, monkeypatch):
        def _fake(blind_well_id, refresh=False):
            return {
                "status": "validated", "message": None, "blind_well_id": blind_well_id,
                "training_well_ids": ["Z-03_RAW", "Z-04_RAW", "Z-05_RAW"], "excluded_wells": [],
                "feature_names": ["attr_amp_center"],
                "blind_results": {
                    "ai": {
                        "status": "validated", "model_name": "xgb_shallow", "cv_r2": 0.1, "facies_alpha": None,
                        "blind_well_r2": 0.05, "blind_well_rmse": 100.0, "n_blind_samples": 20,
                        "depth_m": [3500.0], "time_ms": [2100.0], "y_true": [5000.0], "y_pred": [5010.0],
                    },
                    "gr": {
                        "status": "validated", "model_name": "ridge", "cv_r2": -0.02, "facies_alpha": 0.75,
                        "blind_well_r2": 0.1, "blind_well_rmse": 10.0, "n_blind_samples": 20,
                        "depth_m": [3500.0], "time_ms": [2100.0], "y_true": [60.0], "y_pred": [58.0],
                    },
                },
            }

        monkeypatch.setattr(ssp, "get_or_train_cascade", _fake)
        resp = client.get("/api/sweet-spot/train", params={"blind_well_id": "Z-02_RAW"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "validated"
        assert body["blind_well_id"] == "Z-02_RAW"
        assert "Z-02_RAW" not in body["training_well_ids"]
        assert body["results"]["ai"]["blind_well_r2"] == 0.05
        assert body["results"]["gr"]["facies_alpha"] == 0.75

    def test_blind_well_unusable(self, client, monkeypatch):
        def _fake(blind_well_id, refresh=False):
            return {
                "status": "blind_well_unusable", "message": "no usable tie", "blind_well_id": blind_well_id,
                "training_well_ids": [], "excluded_wells": [], "feature_names": [], "blind_results": None,
            }

        monkeypatch.setattr(ssp, "get_or_train_cascade", _fake)
        resp = client.get("/api/sweet-spot/train", params={"blind_well_id": "Z-02_RAW"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "blind_well_unusable"

    def test_passes_refresh_param(self, client, monkeypatch):
        seen = {}

        def _fake(blind_well_id, refresh=False):
            seen["refresh"] = refresh
            return {
                "status": "insufficient_data", "message": "too few wells", "blind_well_id": blind_well_id,
                "training_well_ids": [], "excluded_wells": [], "feature_names": [], "blind_results": None,
            }

        monkeypatch.setattr(ssp, "get_or_train_cascade", _fake)
        resp = client.get("/api/sweet-spot/train", params={"blind_well_id": "Z-02_RAW", "refresh": "true"})
        assert resp.status_code == 200
        assert seen["refresh"] is True

    def test_unknown_well_is_404(self, client, monkeypatch):
        def _fake(blind_well_id, refresh=False):
            raise WellNotFoundError(blind_well_id)

        monkeypatch.setattr(ssp, "get_or_train_cascade", _fake)
        resp = client.get("/api/sweet-spot/train", params={"blind_well_id": "DOES_NOT_EXIST"})
        assert resp.status_code == 404

    def test_tie_error_is_422(self, client, monkeypatch):
        def _fake(blind_well_id, refresh=False):
            raise wst.TieError("no correlation")

        monkeypatch.setattr(ssp, "get_or_train_cascade", _fake)
        resp = client.get("/api/sweet-spot/train", params={"blind_well_id": "Z-02_RAW"})
        assert resp.status_code == 422


class TestRegionPredictionEndpoint:
    def test_ok(self, client, monkeypatch):
        def _fake(inline_range, crossline_range, property_names, blind_well_id):
            return {
                "blind_well_id": blind_well_id, "inline_axis": [400, 401], "crossline_axis": [100, 101],
                "twt_axis_ms": [2000.0, 2002.0],
                "predictions": {p: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] for p in property_names},
            }

        monkeypatch.setattr(ssp, "predict_region", _fake)
        resp = client.get(
            "/api/sweet-spot/region-prediction",
            params={"inline_min": 400, "inline_max": 401, "crossline_min": 100, "crossline_max": 101, "properties": "gr,vsh"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["predictions"].keys()) == {"gr", "vsh"}
        assert body["inline_axis"] == [400, 401]

    def test_passes_parsed_property_list_and_ranges(self, client, monkeypatch):
        seen = {}

        def _fake(inline_range, crossline_range, property_names, blind_well_id):
            seen["inline_range"] = inline_range
            seen["crossline_range"] = crossline_range
            seen["property_names"] = property_names
            seen["blind_well_id"] = blind_well_id
            return {
                "blind_well_id": blind_well_id, "inline_axis": [], "crossline_axis": [], "twt_axis_ms": [],
                "predictions": {},
            }

        monkeypatch.setattr(ssp, "predict_region", _fake)
        resp = client.get(
            "/api/sweet-spot/region-prediction",
            params={
                "inline_min": 400, "inline_max": 410, "crossline_min": 100, "crossline_max": 110,
                "properties": " gr , vsh ,phie", "blind_well_id": "Z-05_RAW",
            },
        )
        assert resp.status_code == 200
        assert seen["inline_range"] == (400, 410)
        assert seen["crossline_range"] == (100, 110)
        assert seen["property_names"] == ["gr", "vsh", "phie"]
        assert seen["blind_well_id"] == "Z-05_RAW"

    def test_prediction_error_is_422(self, client, monkeypatch):
        def _fake(inline_range, crossline_range, property_names, blind_well_id):
            raise ssp.SweetSpotPredictionError("region too large")

        monkeypatch.setattr(ssp, "predict_region", _fake)
        resp = client.get(
            "/api/sweet-spot/region-prediction",
            params={"inline_min": 400, "inline_max": 401, "crossline_min": 100, "crossline_max": 101, "properties": "gr"},
        )
        assert resp.status_code == 422

    def test_unknown_well_is_404(self, client, monkeypatch):
        def _fake(inline_range, crossline_range, property_names, blind_well_id):
            raise WellNotFoundError(blind_well_id)

        monkeypatch.setattr(ssp, "predict_region", _fake)
        resp = client.get(
            "/api/sweet-spot/region-prediction",
            params={"inline_min": 400, "inline_max": 401, "crossline_min": 100, "crossline_max": 101, "properties": "gr", "blind_well_id": "NOPE"},
        )
        assert resp.status_code == 404

    def test_missing_required_query_params_is_422(self, client):
        resp = client.get("/api/sweet-spot/region-prediction", params={"properties": "gr"})
        assert resp.status_code == 422
