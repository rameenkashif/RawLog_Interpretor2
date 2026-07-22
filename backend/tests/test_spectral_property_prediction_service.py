"""
test_spectral_property_prediction_service.py
-------------------------------------------------
Tests for services/spectral_property_prediction_service.py -- multi-
frequency CWT/SSWT -> VSH/PHIE/SWE prediction, validated with leave-one-
well-out cross-validation.

Layered coverage:
- _train_and_loocv is tested as a pure function against constructed
  feature dicts (no real LAS/SEG-Y needed) -- this is where the actual
  ML mechanics (no leakage between folds, per-well/pooled scoring,
  per-property independence) are verified precisely and fast.
- get_property_models' orchestration (eligibility gating, insufficient-
  data branching, excluded-well reasons) is tested by monkeypatching
  _eligible_wells/_extract_well_features directly, since those are this
  module's own seams -- real well/tie resolution is already covered by
  test_spectral_petro_correlation_service.py's TestTimeShiftCorrection
  (the shift-sign-convention correctness this module depends on) and
  test_dashboard_upload_service.py (get_synthetic_summary itself).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import spectral_property_prediction_service as sppp


def _well_features(n_samples: int, freq_offset: float, slope: float = 1.0, n_freq: int = 4, seed: int = 0) -> dict:
    """A well's feature dict with a KNOWN, controllable linear
    relationship between the first feature column and the target, so
    LOOCV recovery can be checked against ground truth rather than just
    "does it run"."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=freq_offset, scale=0.1, size=(n_samples, n_freq))
    y_vsh = slope * X[:, 0] + rng.normal(scale=1e-3, size=n_samples)
    return {
        "sswt_freq_hz": np.array([5.0, 10.0, 15.0, 20.0][:n_freq]),
        "sswt_X": X,
        "cwt_freq_hz": np.array([5.0, 10.0, 15.0, 20.0][:n_freq]),
        "cwt_X": X.copy(),
        "y": {
            "vsh": y_vsh,
            "phie": np.full(n_samples, np.nan),  # deliberately insufficient for phie
            "swe": rng.normal(size=n_samples),
        },
    }


class TestTrainAndLoocv:
    def test_recovers_a_known_linear_relationship(self):
        wells = {
            "A": _well_features(30, freq_offset=0.0, slope=2.0, seed=1),
            "B": _well_features(30, freq_offset=0.0, slope=2.0, seed=2),
            "C": _well_features(30, freq_offset=0.0, slope=2.0, seed=3),
        }
        result = sppp._train_and_loocv(wells, "sswt", "vsh")
        assert result is not None
        assert result["n_wells_used"] == 3
        assert result["loocv_r2"] is not None
        assert result["loocv_r2"] > 0.5  # same linear relationship in every well -- should generalize well
        assert len(result["per_well"]) == 3
        assert {r["well_id"] for r in result["per_well"]} == {"A", "B", "C"}
        assert len(result["feature_importance"]) == 4

    def test_held_out_well_never_appears_in_its_own_training_fold(self, monkeypatch):
        from sklearn.ensemble import RandomForestRegressor

        wells = {
            "A": _well_features(20, freq_offset=0.0, seed=1),
            "B": _well_features(25, freq_offset=0.0, seed=2),
            "C": _well_features(30, freq_offset=0.0, seed=3),
        }
        fit_sizes: list[int] = []
        original_fit = RandomForestRegressor.fit

        def _spy_fit(self, X, y, *a, **k):
            fit_sizes.append(len(X))
            return original_fit(self, X, y, *a, **k)

        monkeypatch.setattr(RandomForestRegressor, "fit", _spy_fit)

        sppp._train_and_loocv(wells, "sswt", "vsh")

        # 3 LOOCV folds (20+25+30 minus the held-out well each time) + 1
        # final all-wells fit for feature importance = 4 total .fit calls.
        total = 20 + 25 + 30
        loocv_sizes = sorted(fit_sizes[:3])
        assert loocv_sizes == sorted([total - 20, total - 25, total - 30])
        assert fit_sizes[3] == total  # the final in-sample importance model uses everyone

    def test_returns_none_when_fewer_than_two_wells_have_enough_samples(self):
        wells = {"A": _well_features(30, freq_offset=0.0, seed=1)}
        assert sppp._train_and_loocv(wells, "sswt", "vsh") is None

    def test_well_with_too_few_valid_samples_for_this_property_is_excluded_from_that_property_only(self):
        wells = {
            "A": _well_features(30, freq_offset=0.0, seed=1),
            "B": _well_features(30, freq_offset=0.0, seed=2),
        }
        # phie is all-NaN in the fixture -- must return None, not crash,
        # while vsh (same wells) still computes fine.
        assert sppp._train_and_loocv(wells, "sswt", "phie") is None
        assert sppp._train_and_loocv(wells, "sswt", "vsh") is not None

    def test_methods_are_independent(self):
        wells = {
            "A": _well_features(30, freq_offset=0.0, seed=1),
            "B": _well_features(30, freq_offset=0.0, seed=2),
        }
        sswt_result = sppp._train_and_loocv(wells, "sswt", "vsh")
        cwt_result = sppp._train_and_loocv(wells, "cwt", "vsh")
        assert sswt_result is not None
        assert cwt_result is not None


class TestEligibleWells:
    def test_excludes_low_confidence_and_error_wells_with_reasons(self, monkeypatch):
        from app.services import dashboard_upload_service as dus
        from app.services import well_service

        class _Summary:
            def __init__(self, well_id):
                self.well_id = well_id

        monkeypatch.setattr(
            well_service, "list_well_summaries", lambda: [_Summary("GOOD"), _Summary("LOW_CONF"), _Summary("NO_TIE")]
        )

        def _fake_summary(well_id):
            if well_id == "GOOD":
                return {"correlation": 0.9, "boundary_pinned": False, "low_confidence": False, "best_shift_ms": 5.0}
            if well_id == "LOW_CONF":
                return {"correlation": 0.1, "boundary_pinned": False, "low_confidence": True, "best_shift_ms": 5.0}
            return {"error": "no coordinates available"}

        monkeypatch.setattr(dus, "get_synthetic_summary", _fake_summary)

        eligible, excluded = sppp._eligible_wells()

        assert eligible == ["GOOD"]
        excluded_ids = {e["well_id"] for e in excluded}
        assert excluded_ids == {"LOW_CONF", "NO_TIE"}
        assert all(e["reason"] for e in excluded)  # never an empty/silent reason

    def test_boundary_pinned_excluded_with_specific_reason(self, monkeypatch):
        from app.services import dashboard_upload_service as dus
        from app.services import well_service

        class _Summary:
            well_id = "PINNED"

        monkeypatch.setattr(well_service, "list_well_summaries", lambda: [_Summary()])
        monkeypatch.setattr(
            dus, "get_synthetic_summary",
            lambda well_id: {"correlation": 0.9, "boundary_pinned": True, "low_confidence": True, "best_shift_ms": 0.0},
        )

        eligible, excluded = sppp._eligible_wells()
        assert eligible == []
        assert "boundary" in excluded[0]["reason"].lower()


class TestGetPropertyModelsOrchestration:
    def test_insufficient_data_when_fewer_than_two_eligible_wells(self, monkeypatch):
        monkeypatch.setattr(sppp, "_eligible_wells", lambda: (["Z-02_RAW"], [{"well_id": "Z-03_RAW", "reason": "low confidence"}]))

        result = sppp.get_property_models()

        assert result["status"] == "insufficient_data"
        assert result["results"] is None
        assert result["n_wells_used"] == 1
        assert "Z-03_RAW" in {e["well_id"] for e in result["excluded_wells"]}
        assert result["message"] is not None

    def test_insufficient_data_with_zero_eligible_wells(self, monkeypatch):
        monkeypatch.setattr(sppp, "_eligible_wells", lambda: ([], []))
        result = sppp.get_property_models()
        assert result["status"] == "insufficient_data"
        assert result["n_wells_used"] == 0

    def test_well_failing_feature_extraction_is_moved_to_excluded_not_crashed(self, monkeypatch):
        from app.services import dashboard_upload_service as dus
        from app.services import seismic_processor as sp

        monkeypatch.setattr(sppp, "_eligible_wells", lambda: (["A", "B"], []))
        monkeypatch.setattr(sp, "get_segy_volume", lambda: object())
        monkeypatch.setattr(dus, "get_synthetic_summary", lambda well_id: {"best_shift_ms": 3.0})

        def _fake_extract(volume, well_id, shift_ms):
            if well_id == "A":
                return None  # simulates a resolution failure despite passing eligibility
            return _well_features(30, freq_offset=0.0, seed=42)

        monkeypatch.setattr(sppp, "_extract_well_features", _fake_extract)

        result = sppp.get_property_models()

        # Only 1 well had extractable features -> still insufficient for LOOCV.
        assert result["status"] == "insufficient_data"
        assert "A" in {e["well_id"] for e in result["excluded_wells"]}

    def test_validated_status_with_enough_wells(self, monkeypatch):
        from app.services import dashboard_upload_service as dus
        from app.services import seismic_processor as sp

        monkeypatch.setattr(sppp, "_eligible_wells", lambda: (["A", "B"], []))
        monkeypatch.setattr(sp, "get_segy_volume", lambda: object())
        monkeypatch.setattr(dus, "get_synthetic_summary", lambda well_id: {"best_shift_ms": 3.0})
        monkeypatch.setattr(
            sppp, "_extract_well_features",
            lambda volume, well_id, shift_ms: _well_features(30, freq_offset=0.0, slope=2.0, seed=hash(well_id) % 100),
        )

        result = sppp.get_property_models()

        assert result["status"] == "validated"
        assert result["n_wells_used"] == 2
        assert set(result["results"].keys()) == {"vsh", "phie", "swe"}
        assert result["results"]["vsh"]["sswt"] is not None
        assert result["results"]["phie"]["sswt"] is None  # all-NaN in the fixture
