"""
test_prediction_pipeline_service.py
--------------------------------------
Tests for services/prediction_pipeline_service.py.

- _predict_blind is tested as a pure function against constructed
  WellPredictionFeatures (no real LAS/SEG-Y needed) -- real well/tie
  resolution (_build_well_features) needs a real SEG-Y volume and is
  exercised indirectly via test_spectral_property_prediction_service.py's
  coverage of direct_tie_service, which this module's tie resolution is
  identical to.
- run_grid_search/get_best_config are tested against the SAME constructed
  wells, monkeypatching _eligible_well_features so no real volume is
  needed there either.

Several model paths (raw spectrum + Ridge; instantaneous-attrs + PCA-3 +
RandomForest; instantaneous-attrs + PCA-3 + Ridge) are exercised across
tests, but "recovers a known relationship"/grid-search-picks-signal tests
use a plain (spectrum-only, no PCA, Ridge) config -- the simplest path,
whose R^2 behavior can be reasoned about confidently without running the
actual model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import prediction_pipeline_service as pps

PLAIN_RIDGE_CONFIG: pps.Config = ("sswt", False, None, "ridge")


def _well(well_id: str, n: int, slope: float, seed: int) -> pps.WellPredictionFeatures:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=0.1, size=(n, 4))
    vsh = np.clip(slope * X[:, 0] + 0.5 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    phie = np.clip(slope * X[:, 0] + 0.15 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    # Small-scale, low-variance relative to the spectral features (scale
    # 0.1) so PCA-3 doesn't get dominated by uninformative instantaneous-
    # attribute variance and squeeze out the real signal.
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
        result = pps._predict_blind(wells, "C", "phie", PLAIN_RIDGE_CONFIG)

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
        config = ("sswt", True, 3, "ridge")  # instantaneous-attrs + PCA-3 + Ridge
        result = pps._predict_blind(wells, "B", "swe", config)

        assert all(0.0 <= v <= 1.0 for v in result["y_pred"])

    def test_recovers_a_known_relationship_reasonably(self):
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "phie", PLAIN_RIDGE_CONFIG)
        assert result["r2"] is not None
        assert result["r2"] > 0.3  # same relationship in every well -- should generalize

    def test_depths_and_twt_are_the_blind_wells_own(self):
        wells = {"A": _well("A", 10, slope=1.0, seed=1), "B": _well("B", 12, slope=1.0, seed=2)}
        config = ("sswt", True, 3, "rf")  # instantaneous-attrs + PCA-3 + RandomForest
        result = pps._predict_blind(wells, "B", "vsh", config)
        assert result["depths_m"] == wells["B"].depths_m.tolist()
        assert result["twt_ms"] == wells["B"].twt_ms.tolist()


class TestGetPredictionResultValidation:
    def test_rejects_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            pps.get_prediction_result("Z-04_RAW", "not_a_target")


class TestGridSearch:
    """run_grid_search/get_best_config, against constructed wells with
    _eligible_well_features monkeypatched so no real SEG-Y volume is
    needed. Uses a small subset of the real grid (monkeypatching the
    GRID_* constants) so these tests run in a reasonable time -- the
    full 48-candidate x n_wells-fold search is exercised implicitly by
    running with the real constants at least once
    (test_full_grid_runs_and_picks_a_scored_winner).
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        pps._GRID_SEARCH_CACHE.clear()
        yield
        pps._GRID_SEARCH_CACHE.clear()

    def _patch_wells(self, monkeypatch, wells):
        monkeypatch.setattr(pps, "_eligible_well_features", lambda volume: (wells, []))
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: object())

    def test_picks_the_config_with_the_best_pooled_r2(self, monkeypatch):
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        self._patch_wells(monkeypatch, wells)
        # Small search space so this test is fast and deterministic to reason about.
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge", "pls"))

        result = pps.run_grid_search("phie")

        assert result["best_config"] is not None
        assert result["best_r2"] is not None
        scored = [e for e in result["leaderboard"] if e["r2"] is not None]
        assert result["best_r2"] == max(e["r2"] for e in scored)
        assert len(result["leaderboard"]) == 2  # ridge, pls

    def test_get_best_config_caches_across_calls(self, monkeypatch):
        wells = {"A": _well("A", 30, slope=2.0, seed=1), "B": _well("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))

        calls = {"n": 0}
        original = pps.run_grid_search

        def _counting_search(target):
            calls["n"] += 1
            return original(target)

        monkeypatch.setattr(pps, "run_grid_search", _counting_search)

        first = pps.get_best_config("phie")
        second = pps.get_best_config("phie")

        assert first == second
        assert calls["n"] == 1  # second call reused the cache, didn't re-search

    def test_force_true_bypasses_the_cache(self, monkeypatch):
        wells = {"A": _well("A", 30, slope=2.0, seed=1), "B": _well("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))

        calls = {"n": 0}
        original = pps.run_grid_search

        def _counting_search(target):
            calls["n"] += 1
            return original(target)

        monkeypatch.setattr(pps, "run_grid_search", _counting_search)

        pps.get_best_config("phie")
        pps.get_best_config("phie", force=True)

        assert calls["n"] == 2

    def test_a_failing_candidate_does_not_crash_the_search(self, monkeypatch):
        wells = {"A": _well("A", 30, slope=2.0, seed=1), "B": _well("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge", "not_a_real_model"))

        result = pps.run_grid_search("phie")

        failed = [e for e in result["leaderboard"] if e["error"] is not None]
        assert len(failed) == 1
        assert result["best_config"] is not None  # ridge still won despite the other candidate failing

    def test_insufficient_wells_returns_no_winner(self, monkeypatch):
        wells = {"A": _well("A", 30, slope=2.0, seed=1)}
        self._patch_wells(monkeypatch, wells)

        result = pps.run_grid_search("phie")

        assert result["best_config"] is None
        assert result["leaderboard"] == []

    def test_full_grid_runs_and_picks_a_scored_winner(self, monkeypatch):
        """Exercises the REAL (un-monkeypatched) 48-candidate grid at
        least once, so a shape/API mismatch in any (spectrum, PCA,
        inst_attrs, model) combination -- not just the ones the faster
        tests above cover -- would fail here."""
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        self._patch_wells(monkeypatch, wells)

        result = pps.run_grid_search("vsh")

        assert len(result["leaderboard"]) == (
            len(pps.GRID_SPECTRA) * len(pps.GRID_PCA_OPTIONS) * len(pps.GRID_USE_INST_OPTIONS) * len(pps.GRID_MODEL_NAMES)
        )
        assert result["best_config"] is not None
        assert result["best_r2"] is not None
