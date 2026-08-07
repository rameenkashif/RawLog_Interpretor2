"""
test_prediction_router.py
----------------------------
HTTP-level tests for routers/prediction.py (/api/prediction/*), using
FastAPI's TestClient. The underlying pipeline is monkeypatched at
run_blind_well_prediction -- routers/prediction.py is thin wiring only,
already covered end-to-end by test_blind_well_prediction_service.py.
"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from app.services import blind_well_prediction_service as bwp
from app.services.well_service import WellNotFoundError


@pytest.fixture
def client(monkeypatch):
    import main

    with TestClient(main.app) as c:
        yield c


class TestBlindWellEndpoint:
    def test_ok(self, client, monkeypatch):
        monkeypatch.setattr(
            bwp, "run_blind_well_prediction",
            lambda blind_well_id="Z-02_RAW": {
                "status": "validated",
                "message": None,
                "blind_well_id": blind_well_id,
                "training_well_ids": ["Z-03_RAW", "Z-04_RAW", "Z-05_RAW"],
                "excluded_wells": [],
                "neighborhood_radius_m": bwp.NEIGHBORHOOD_RADIUS_M,
                "results": {
                    "vsh": {
                        "status": "validated",
                        "message": None,
                        "selected_features": ["cwt_dominant_freq_hz"],
                        "feature_diagnostics": [
                            {"feature": "cwt_dominant_freq_hz", "pooled_corr": 0.6, "stable": True, "selected": True}
                        ],
                        "decision_gate_r2": 0.2,
                        "stack_loocv_r2": 0.3,
                        "blind_well_r2": 0.15,
                        "blind_well_rmse": 0.1,
                        "n_training_samples": 900,
                        "n_blind_samples": 30,
                        "depth_m": [3500.0, 3500.5],
                        "y_true": [0.3, 0.35],
                        "y_pred": [0.28, 0.31],
                    },
                    "phie": {"status": "insufficient_data", "message": "too few samples"},
                    "swe": {"status": "no_stable_features", "message": "no stable feature"},
                },
            },
        )
        resp = client.get("/api/prediction/blind-well")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "validated"
        assert body["blind_well_id"] == "Z-02_RAW"
        assert "Z-02_RAW" not in body["training_well_ids"]
        assert body["results"]["vsh"]["blind_well_r2"] == 0.15
        assert body["results"]["phie"]["status"] == "insufficient_data"

    def test_accepts_blind_well_id_query_param(self, client, monkeypatch):
        seen = {}

        def _fake(blind_well_id="Z-02_RAW"):
            seen["blind_well_id"] = blind_well_id
            return {
                "status": "blind_well_unusable",
                "message": "no usable tie",
                "blind_well_id": blind_well_id,
                "training_well_ids": [],
                "excluded_wells": [],
                "neighborhood_radius_m": bwp.NEIGHBORHOOD_RADIUS_M,
                "results": None,
            }

        monkeypatch.setattr(bwp, "run_blind_well_prediction", _fake)
        resp = client.get("/api/prediction/blind-well", params={"blind_well_id": "Z-05_RAW"})
        assert resp.status_code == 200
        assert seen["blind_well_id"] == "Z-05_RAW"
        assert resp.json()["status"] == "blind_well_unusable"

    def test_unknown_well_is_404(self, client, monkeypatch):
        def _fake(blind_well_id="Z-02_RAW"):
            raise WellNotFoundError(blind_well_id)

        monkeypatch.setattr(bwp, "run_blind_well_prediction", _fake)
        resp = client.get("/api/prediction/blind-well", params={"blind_well_id": "DOES_NOT_EXIST"})
        assert resp.status_code == 404
