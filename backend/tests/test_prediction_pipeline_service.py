"""
test_prediction_pipeline_service.py
--------------------------------------
Tests for services/prediction_pipeline_service.py. _predict_blind is
tested as a pure function against constructed WellPredictionFeatures (no
real LAS/SEG-Y needed) -- real well/tie resolution (_build_well_features)
needs a real SEG-Y volume and is exercised indirectly via
test_spectral_property_prediction_service.py's coverage of
direct_tie_service, which this module's tie resolution is identical to.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import prediction_pipeline_service as pps


def _well(well_id: str, n: int, slope: float, seed: int) -> pps.WellPredictionFeatures:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=0.1, size=(n, 4))
    vsh = np.clip(slope * X[:, 0] + 0.5 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    return pps.WellPredictionFeatures(
        well_id=well_id,
        inline_number=100,
        crossline_number=200,
        correlation=0.9,
        depths_m=np.arange(n, dtype=float),
        twt_ms=np.arange(n, dtype=float) * 2.0,
        targets={"vsh": vsh, "phie": rng.uniform(0.05, 0.25, n), "swe": rng.uniform(0, 1, n)},
        features={"cwt": X, "sswt": X.copy()},
        freq_hz={"cwt": np.array([5.0, 10.0, 15.0, 20.0]), "sswt": np.array([5.0, 10.0, 15.0, 20.0])},
    )


class TestPredictBlind:
    def test_blind_well_never_appears_in_training_set(self):
        wells = {
            "A": _well("A", 20, slope=2.0, seed=1),
            "B": _well("B", 25, slope=2.0, seed=2),
            "C": _well("C", 30, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "cwt", "vsh")

        assert result["n_train_wells"] == 2
        assert result["n_train_samples"] == 20 + 25
        assert result["n_test_samples"] == 30
        assert len(result["y_true"]) == 30
        assert len(result["y_pred"]) == 30

    def test_predictions_are_clipped_to_0_1(self):
        wells = {
            "A": _well("A", 20, slope=50.0, seed=1),  # exaggerated slope to try to escape [0,1]
            "B": _well("B", 20, slope=50.0, seed=2),
        }
        result = pps._predict_blind(wells, "B", "cwt", "vsh")

        assert all(0.0 <= v <= 1.0 for v in result["y_pred"])

    def test_recovers_a_known_relationship_reasonably(self):
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "cwt", "vsh")
        assert result["r2"] is not None
        assert result["r2"] > 0.3  # same relationship in every well -- should generalize

    def test_depths_and_twt_are_the_blind_wells_own(self):
        wells = {"A": _well("A", 10, slope=1.0, seed=1), "B": _well("B", 12, slope=1.0, seed=2)}
        result = pps._predict_blind(wells, "B", "sswt", "phie")
        assert result["depths_m"] == wells["B"].depths_m.tolist()
        assert result["twt_ms"] == wells["B"].twt_ms.tolist()


class TestGetPredictionResultValidation:
    def test_rejects_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            pps.get_prediction_result("Z-04_RAW", "not_a_target", "cwt")

    def test_rejects_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            pps.get_prediction_result("Z-04_RAW", "vsh", "not_a_method")
