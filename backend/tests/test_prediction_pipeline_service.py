"""
test_prediction_pipeline_service.py
--------------------------------------
Tests for services/prediction_pipeline_service.py. _predict_blind is
tested as a pure function against constructed WellPredictionFeatures (no
real LAS/SEG-Y needed) -- real well/tie resolution (_build_well_features)
needs a real SEG-Y volume and is exercised indirectly via
test_spectral_property_prediction_service.py's coverage of
direct_tie_service, which this module's tie resolution is identical to.

Each BEST_CONFIG model path (phie: sswt-only + Ridge, no PCA input beyond
the spectrum; vsh: sswt + instantaneous attrs + PCA-3 + RandomForest; swe:
sswt + instantaneous attrs + PCA-3 + Ridge) is exercised by at least one
test, but "recovers a known relationship" is checked against 'phie' only
-- the simplest path (no PCA-diluting instantaneous-attribute columns,
no RandomForest's own step-function approximation quirks) is the one
whose R^2 threshold can be reasoned about confidently without running the
actual model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import prediction_pipeline_service as pps


def _well(well_id: str, n: int, slope: float, seed: int) -> pps.WellPredictionFeatures:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=0.1, size=(n, 4))
    vsh = np.clip(slope * X[:, 0] + 0.5 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    phie = np.clip(slope * X[:, 0] + 0.15 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    # Small-scale, low-variance relative to the spectral features (scale
    # 0.1) so PCA-3 (vsh/swe) doesn't get dominated by uninformative
    # instantaneous-attribute variance and squeeze out the real signal.
    inst_attrs = rng.normal(scale=0.01, size=(n, 6))
    return pps.WellPredictionFeatures(
        well_id=well_id,
        inline_number=100,
        crossline_number=200,
        correlation=0.9,
        best_freq_hz=25.0,
        polarity=1,
        bulk_shift_ms=0.0,
        depths_m=np.arange(n, dtype=float),
        twt_ms=np.arange(n, dtype=float) * 2.0,
        targets={"vsh": vsh, "phie": phie, "swe": rng.uniform(0, 1, n)},
        features={"cwt": X, "sswt": X.copy()},
        freq_hz={"cwt": np.array([5.0, 10.0, 15.0, 20.0]), "sswt": np.array([5.0, 10.0, 15.0, 20.0])},
        inst_attrs=inst_attrs,
    )


class TestPredictBlind:
    def test_blind_well_never_appears_in_training_set(self):
        wells = {
            "A": _well("A", 20, slope=2.0, seed=1),
            "B": _well("B", 25, slope=2.0, seed=2),
            "C": _well("C", 30, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "phie")

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
        result = pps._predict_blind(wells, "B", "swe")  # exercises instantaneous-attrs + PCA-3 + Ridge

        assert all(0.0 <= v <= 1.0 for v in result["y_pred"])

    def test_recovers_a_known_relationship_reasonably(self):
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "phie")
        assert result["r2"] is not None
        assert result["r2"] > 0.3  # same relationship in every well -- should generalize

    def test_depths_and_twt_are_the_blind_wells_own(self):
        wells = {"A": _well("A", 10, slope=1.0, seed=1), "B": _well("B", 12, slope=1.0, seed=2)}
        result = pps._predict_blind(wells, "B", "vsh")  # exercises instantaneous-attrs + PCA-3 + RandomForest
        assert result["depths_m"] == wells["B"].depths_m.tolist()
        assert result["twt_ms"] == wells["B"].twt_ms.tolist()


class TestGetPredictionResultValidation:
    def test_rejects_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            pps.get_prediction_result("Z-04_RAW", "not_a_target")
