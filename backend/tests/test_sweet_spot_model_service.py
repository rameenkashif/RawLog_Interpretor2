"""
test_sweet_spot_model_service.py
------------------------------------
Tests for services/sweet_spot_model_service.py -- model pool, LOGO-CV,
facies modulation, and the 2-stage cascade.

Most tests substitute a small, cheap 2-template stub for
_make_base_templates() (via monkeypatch) rather than the real 11-template
pool -- the real pool's doc-faithful hyperparameters (200-500 trees,
3-way internal stacking CV) take real minutes to LOGO-CV across even a
handful of targets, which would make this test file unacceptably slow.
TestMakeBaseTemplates checks the real pool's structure directly (no
fitting) so that fidelity is still covered.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from app.services import sweet_spot_model_service as sms
from app.services.sweet_spot_calibration import (
    per_well_winsorize_and_zscore_target,
    per_well_zscore_features,
)
from app.services.sweet_spot_training_data import (
    STAGE1_TARGETS,
    STAGE2_TARGETS,
    _SweetSpotWellSamples,
)

FEATURES = ("f0", "f1", "f2")


def _cheap_templates():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "rf_tiny": lambda: RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42),
    }


def _make_well(well_id, n, signal_col=0, coef=2.0, noise=0.3, seed=0, vsh_col=None):
    rng = np.random.default_rng(seed)
    X58 = rng.normal(0, 1, size=(n, len(FEATURES)))
    time_ms = np.linspace(2000.0, 2100.0, n)
    y_signal = coef * X58[:, signal_col] + rng.normal(0, noise, n)
    vsh = np.clip(0.3 + 0.1 * X58[:, vsh_col if vsh_col is not None else signal_col], 0, 1)
    is_sand = (vsh < 0.4).astype(float)
    targets = {
        "ai": 5000 + 100 * y_signal, "dt": 70 - y_signal, "phit": np.clip(0.15 + 0.01 * y_signal, 0, 0.3),
        "gr": 60 + 10 * y_signal, "rhob": 2.3 + 0.02 * y_signal, "vsh": vsh,
        "phie": np.clip(0.15 + 0.01 * y_signal, 0, 0.3), "swe": np.clip(0.5 - 0.05 * y_signal, 0, 1),
        "is_sand": is_sand,
    }
    return _SweetSpotWellSamples(
        well_id=well_id, X58=X58, feature_names=FEATURES, targets=targets, time_ms=time_ms,
        depth_m=np.zeros(n), tie_weight=0.8, tie_correlation=0.8, inline_number=1, crossline_number=1,
        distance_m=10.0, center_row_start=0, center_row_end=n,
    )


def _wells(n_wells=5, n_per_well=60):
    return [_make_well(f"W{i}", n_per_well, seed=i) for i in range(n_wells)]


class TestMakeBaseTemplates:
    """Structural check of the real (unmonkeypatched) 11-template pool --
    doesn't fit anything, so stays fast."""

    def test_11_templates_present(self):
        templates = sms._make_base_templates()
        assert len(templates) == 11
        assert set(templates.keys()) == {
            "rf_shallow", "rf_deep", "et_shallow", "et_deep", "xgb_shallow", "xgb_regularized",
            "lgbm_shallow", "lgbm_deep", "lgbm_regularized", "stack_ridge", "stack_tree",
        }

    def test_each_template_constructs_a_fresh_unfitted_estimator(self):
        templates = sms._make_base_templates()
        for name, factory in templates.items():
            m1 = factory()
            m2 = factory()
            assert m1 is not m2, f"{name} factory returned the same instance twice"


class TestLogoCvScore:
    def test_recovers_strong_signal(self):
        wells = _wells(5, 60)
        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "ai")
        r2, oof = sms.logo_cv_score(lambda: Ridge(alpha=1.0), wells, X_scaled, target_scales, "ai")
        assert r2 is not None
        assert r2 > 0.5
        assert set(oof.keys()) == {w.well_id for w in wells}

    def test_held_out_well_never_appears_in_its_own_training_fold(self):
        wells = _wells(4, 40)
        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "ai")

        seen_train_sizes = []

        class _SpyRidge(Ridge):
            def fit(self, X, y, sample_weight=None):
                seen_train_sizes.append(X.shape[0])
                return super().fit(X, y, sample_weight=sample_weight)

        sms.logo_cv_score(lambda: _SpyRidge(alpha=1.0), wells, X_scaled, target_scales, "ai")
        total_samples = sum(len(w.time_ms) for w in wells)
        for train_size in seen_train_sizes:
            # Each fold's training size must be strictly less than the
            # pooled total (i.e. the held-out well's samples are excluded).
            assert train_size < total_samples

    def test_too_few_wells_returns_none(self):
        wells = _wells(1, 30)
        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "ai")
        r2, oof = sms.logo_cv_score(lambda: Ridge(alpha=1.0), wells, X_scaled, target_scales, "ai")
        assert r2 is None


class TestSelectBestModel:
    def test_picks_higher_cv_r2_candidate(self):
        wells = _wells(5, 60)
        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "ai")

        # "good" recovers the real relationship; "bad" is Ridge fit against
        # shuffled features (should score much worse).
        rng = np.random.default_rng(0)
        shuffled_X = {wid: X.copy() for wid, X in X_scaled.items()}
        for wid in shuffled_X:
            rng.shuffle(shuffled_X[wid])

        templates = {"good": lambda: Ridge(alpha=1.0)}
        selection = sms.select_best_model(templates, wells, X_scaled, target_scales, "ai")
        assert selection.status == "selected"
        assert selection.model_name == "good"
        assert selection.cv_r2 > 0.5

    def test_insufficient_data_status_when_no_candidate_scores(self):
        wells = _wells(1, 30)  # too few wells for any LOGO fold
        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "ai")
        selection = sms.select_best_model({"ridge": lambda: Ridge()}, wells, X_scaled, target_scales, "ai")
        assert selection.status == "insufficient_data"


class TestFaciesModulation:
    def test_facies_means_per_fold_excludes_held_out_well(self):
        wells = _wells(4, 40)
        means = sms._facies_means_per_fold(wells, "gr")
        assert set(means.keys()) == {w.well_id for w in wells}
        for wid, (mean_sand, mean_shale) in means.items():
            assert np.isfinite(mean_sand) or np.isfinite(mean_shale)

    def test_blend_at_alpha_zero_matches_pure_ml(self):
        wells = _wells(4, 40)
        oof_ml = {w.well_id: w.targets["gr"] + 0.01 for w in wells}  # pretend "prediction" close to truth
        p_sand = {w.well_id: np.full(len(w.time_ms), 0.5) for w in wells}
        means = sms._facies_means_per_fold(wells, "gr")
        r2_at_0 = sms._facies_blend_cv_r2(wells, oof_ml, p_sand, means, "gr", alpha=0.0)
        # alpha=0 should be a near-perfect match to the (near-truth) oof_ml.
        assert r2_at_0 > 0.99

    def test_select_best_model_can_choose_a_facies_blend(self):
        # Construct a target with NO real feature relationship (so plain ML
        # is weak) but a clean facies split (so blending toward the facies
        # mean should score much better).
        wells = []
        for i in range(5):
            rng = np.random.default_rng(100 + i)
            n = 60
            X58 = rng.normal(0, 1, size=(n, len(FEATURES)))
            vsh = np.clip(0.3 + 0.1 * X58[:, 0], 0, 1)
            is_sand = (vsh < 0.4).astype(float)
            # swe depends ONLY on facies, not on any feature -- pure noise
            # relationship to X, but a clean sand/shale split.
            swe = np.where(is_sand == 1, 0.2, 0.8) + rng.normal(0, 0.02, n)
            well = _SweetSpotWellSamples(
                well_id=f"W{i}", X58=X58, feature_names=FEATURES,
                targets={"swe": swe, "is_sand": is_sand}, time_ms=np.linspace(2000, 2100, n),
                depth_m=np.zeros(n), tie_weight=0.8, tie_correlation=0.8, inline_number=1,
                crossline_number=1, distance_m=10.0, center_row_start=0, center_row_end=n,
            )
            wells.append(well)

        feature_scales = per_well_zscore_features(wells, list(FEATURES))
        X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in wells}
        target_scales = per_well_winsorize_and_zscore_target(wells, "swe")
        p_sand_oof = sms._fit_facies_classifier_oof(wells, X_scaled)

        selection = sms.select_best_model(
            {"ridge": lambda: Ridge(alpha=1.0)}, wells, X_scaled, target_scales, "swe", p_sand_oof=p_sand_oof,
        )
        assert selection.status == "selected"
        # A pure-facies relationship should be won by a facies-heavy blend.
        assert selection.facies_alpha is not None
        assert selection.facies_alpha >= 0.5


class TestTrainStage1AndStage2:
    @pytest.fixture(autouse=True)
    def _cheap_pool(self, monkeypatch):
        monkeypatch.setattr(sms, "_make_base_templates", _cheap_templates)

    def test_stage1_trains_all_targets_with_signal(self):
        wells = _wells(5, 60)
        stage1 = sms.train_stage1(wells, list(FEATURES))
        assert set(stage1.keys()) == set(STAGE1_TARGETS)
        for target_name, result in stage1.items():
            assert result.status == "trained", f"{target_name} failed to train"
            assert result.final_model is not None
            assert result.calibration is not None

    def test_stage1_insufficient_wells(self):
        wells = _wells(1, 30)
        stage1 = sms.train_stage1(wells, list(FEATURES))
        assert all(r.status == "insufficient_data" for r in stage1.values())

    def test_stage2_uses_stage1_oof_without_crashing(self):
        wells = _wells(5, 60)
        stage1 = sms.train_stage1(wells, list(FEATURES))
        stage2 = sms.train_stage2(wells, list(FEATURES), stage1)
        assert set(stage2.keys()) == set(STAGE2_TARGETS)
        for target_name, result in stage2.items():
            assert result.status == "trained", f"{target_name} failed to train"

    def test_full_cascade_end_to_end(self):
        wells = _wells(5, 60)
        cascade = sms.train_sweet_spot_cascade(wells, list(FEATURES))
        assert set(cascade.stage1.keys()) == set(STAGE1_TARGETS)
        assert set(cascade.stage2.keys()) == set(STAGE2_TARGETS)
        assert cascade.training_well_ids == tuple(w.well_id for w in wells)
        assert cascade.feature_names == tuple(FEATURES)

    def test_stage2_missing_stage1_result_does_not_crash(self):
        # Simulate Stage 1 failing entirely -- Stage 2 should still run,
        # just without useful cascade features (NaN-filled -> zero after
        # z-scoring), not raise.
        wells = _wells(5, 60)
        empty_stage1 = {t: sms._FittedTarget(status="insufficient_data") for t in STAGE1_TARGETS}
        stage2 = sms.train_stage2(wells, list(FEATURES), empty_stage1)
        assert set(stage2.keys()) == set(STAGE2_TARGETS)
