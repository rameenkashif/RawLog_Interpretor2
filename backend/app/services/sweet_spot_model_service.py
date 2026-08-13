"""
services/sweet_spot_model_service.py
----------------------------------------
Model pool, LOGO-CV, and the 2-stage cascade for the Sweet-Spot prediction
module (source doc Section 9). Model SELECTION is always by maximum CV R^2
only -- blind-well R^2 is computed and reported but never fed back into
which model wins, mirroring blind_well_prediction_service.py's own
"decision gate is diagnostic, not selection" philosophy (see its module
docstring) and the source doc's own explicit rule (Section 9: "Best model
is selected by maximum CV R-squared only... Blind well R-squared is
computed but never used for selection").

Facies modulation (doc Section 9's "Plus Facies Modulation variants" for
GR/VSH/PHIE/SWE) is a cheap POST-HOC blend of a base template's own
already-fitted LOGO-CV out-of-fold predictions with a facies-conditioned
baseline -- NOT a separate model refit per alpha. The doc's own formula
(verbatim):

    pred_modulated = (1-alpha)*pred_ML + alpha*(P_sand*mean_sand + (1-P_sand)*mean_shale)

so sweeping alpha in {0, .25, .5, .75, 1.0} costs one blend + one R^2
computation each, not 5x the training cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Imported eagerly (not inside _make_base_templates) so a missing install
# is caught at router registration by main.py's try/except around `from
# app.routers import sweet_spot`, which disables just this module with a
# clear log message, instead of surfacing as a raw 500 on the first
# request that reaches it (see blind_well_prediction_service.py's own
# identical fix for the same xgboost lazy-import issue).
from lightgbm import LGBMRegressor  # noqa: F401
from xgboost import XGBRegressor  # noqa: F401

from app.services.sweet_spot_calibration import (
    DynamicCalibration,
    _WellTargetScale,
    apply_calibration,
    fit_dynamic_calibration,
    per_well_winsorize_and_zscore_target,
    per_well_zscore_features,
)
from app.services.sweet_spot_training_data import (
    STAGE1_TARGETS,
    STAGE2_TARGETS,
    _SweetSpotWellSamples,
)

RANDOM_STATE = 42
FACIES_ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
# Stage 2 targets the doc applies facies modulation to.
FACIES_TARGETS: tuple[str, ...] = ("gr", "vsh", "phie", "swe")
MIN_TRAINING_WELLS = 3


def _make_base_templates() -> dict[str, object]:
    """11 base model templates (doc Section 9's pool), as zero-arg
    constructors so LOGO-CV gets a fresh, unfitted estimator per fold."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    def _xgb(**kw):
        return XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", verbosity=0, **kw)

    def _lgbm(**kw):
        return LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, **kw)

    return {
        "rf_shallow": lambda: RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE),
        "rf_deep": lambda: RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
        "et_shallow": lambda: ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE),
        "et_deep": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
        "xgb_shallow": lambda: _xgb(n_estimators=200, max_depth=3, learning_rate=0.03),
        "xgb_regularized": lambda: _xgb(n_estimators=200, max_depth=4, learning_rate=0.02, reg_alpha=0.5, reg_lambda=3.0),
        "lgbm_shallow": lambda: _lgbm(n_estimators=300, max_depth=3, learning_rate=0.03),
        "lgbm_deep": lambda: _lgbm(n_estimators=300, max_depth=6, learning_rate=0.05),
        "lgbm_regularized": lambda: _lgbm(n_estimators=300, max_depth=4, learning_rate=0.02, reg_alpha=0.5, reg_lambda=3.0),
        "stack_ridge": lambda: StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
                ("xgb", _xgb(n_estimators=150, max_depth=4, learning_rate=0.05)),
            ],
            final_estimator=Ridge(alpha=10.0), cv=3,
        ),
        "stack_tree": lambda: StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
                ("et", ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
                ("xgb", _xgb(n_estimators=150, max_depth=4, learning_rate=0.05)),
            ],
            final_estimator=ExtraTreesRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE), cv=3,
        ),
    }


def _fit_with_optional_weight(model, X, y, weight):
    try:
        model.fit(X, y, sample_weight=weight)
    except (TypeError, ValueError):
        model.fit(X, y)
    return model


def logo_cv_score(
    model_factory,
    wells: list[_SweetSpotWellSamples],
    X_scaled: dict[str, np.ndarray],
    target_scales: dict[str, _WellTargetScale],
    target_name: str,
) -> tuple[float | None, dict[str, np.ndarray]]:
    """LeaveOneGroupOut CV (groups=well_id, via sklearn.LeaveOneGroupOut
    semantics -- implemented directly here since the calibration
    back-conversion step needs to run PER FOLD, not just the model fit).
    Per fold: train on per-well-z-scored pooled OTHER wells, predict the
    held-out well using ITS OWN feature z-scoring, invert to physical
    units via a calibration fit on the (N-1) training wells only, score
    R^2 in PHYSICAL units over every fold's pooled predictions. Returns
    (mean_cv_r2, oof_predictions_physical) -- oof needed as Stage 2's
    cascade features and for facies-modulation blending.
    """
    oof_pred_physical: dict[str, np.ndarray] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for held_out in wells:
        train_wells = [w for w in wells if w.well_id != held_out.well_id]
        if len(train_wells) < 2:
            continue
        calib = fit_dynamic_calibration(train_wells, target_name, target_scales)
        if calib is None:
            continue

        X_parts, y_parts, w_parts = [], [], []
        for w in train_wells:
            X = X_scaled[w.well_id]
            y = target_scales[w.well_id].y_scaled
            weight = np.full(len(y), w.tie_weight)
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if valid.sum() == 0:
                continue
            X_parts.append(X[valid])
            y_parts.append(y[valid])
            w_parts.append(weight[valid])
        if not X_parts:
            continue
        X_train = np.concatenate(X_parts)
        y_train = np.concatenate(y_parts)
        w_train = np.concatenate(w_parts)
        if len(np.unique(y_train)) < 2:
            continue

        X_test = X_scaled[held_out.well_id]
        valid_test = np.all(np.isfinite(X_test), axis=1)
        if not valid_test.any():
            continue

        model = model_factory()
        _fit_with_optional_weight(model, X_train, y_train, w_train)
        z_pred = model.predict(X_test[valid_test])
        y_pred_physical = apply_calibration(z_pred, held_out.time_ms[valid_test], calib)

        pred_full = np.full(len(held_out.time_ms), np.nan)
        pred_full[np.where(valid_test)[0]] = y_pred_physical
        oof_pred_physical[held_out.well_id] = pred_full

        y_true_physical = held_out.targets[target_name][valid_test]
        valid_true = np.isfinite(y_true_physical)
        if valid_true.any():
            all_true.append(y_true_physical[valid_true])
            all_pred.append(y_pred_physical[valid_true])

    if not all_true:
        return None, oof_pred_physical
    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    if len(all_true_arr) < 2 or np.std(all_true_arr) == 0:
        return None, oof_pred_physical

    from sklearn.metrics import r2_score

    return float(r2_score(all_true_arr, all_pred_arr)), oof_pred_physical


def _fit_facies_classifier_oof(
    wells: list[_SweetSpotWellSamples], X_scaled: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """LOGO-CV RandomForestClassifier(IS_SAND) -- target-independent (used
    by every facies-modulated target), so computed once and reused.
    Returns {well_id: P_sand array}, NaN where not scoreable."""
    from sklearn.ensemble import RandomForestClassifier

    oof: dict[str, np.ndarray] = {}
    for held_out in wells:
        train_wells = [w for w in wells if w.well_id != held_out.well_id]
        X_parts, y_parts = [], []
        for w in train_wells:
            X = X_scaled[w.well_id]
            y = w.targets["is_sand"]
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if valid.sum() == 0:
                continue
            X_parts.append(X[valid])
            y_parts.append(y[valid])
        p_full = np.full(len(held_out.time_ms), np.nan)
        if X_parts:
            X_train = np.concatenate(X_parts)
            y_train = np.concatenate(y_parts).astype(int)
            if len(np.unique(y_train)) >= 2:
                clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
                clf.fit(X_train, y_train)
                X_test = X_scaled[held_out.well_id]
                valid_test = np.all(np.isfinite(X_test), axis=1)
                if valid_test.any() and 1 in clf.classes_:
                    sand_col = list(clf.classes_).index(1)
                    p_full[valid_test] = clf.predict_proba(X_test[valid_test])[:, sand_col]
        oof[held_out.well_id] = p_full
    return oof


def _facies_means_per_fold(wells: list[_SweetSpotWellSamples], target_name: str) -> dict[str, tuple[float, float]]:
    """{held_out_well_id: (mean_sand, mean_shale)}, computed from that
    fold's TRAINING wells only -- the held-out well never contributes to
    its own facies baseline, same leakage discipline as everything else."""
    out: dict[str, tuple[float, float]] = {}
    for held_out in wells:
        train_wells = [w for w in wells if w.well_id != held_out.well_id]
        sand_vals, shale_vals = [], []
        for w in train_wells:
            y = w.targets[target_name]
            is_sand = w.targets["is_sand"]
            valid = np.isfinite(y) & np.isfinite(is_sand)
            sand_vals.append(y[valid][is_sand[valid] == 1])
            shale_vals.append(y[valid][is_sand[valid] == 0])
        sand_concat = np.concatenate(sand_vals) if sand_vals else np.array([])
        shale_concat = np.concatenate(shale_vals) if shale_vals else np.array([])
        mean_sand = float(np.mean(sand_concat)) if len(sand_concat) else float("nan")
        mean_shale = float(np.mean(shale_concat)) if len(shale_concat) else float("nan")
        out[held_out.well_id] = (mean_sand, mean_shale)
    return out


def _facies_blend_cv_r2(
    wells: list[_SweetSpotWellSamples],
    oof_ml_physical: dict[str, np.ndarray],
    p_sand_oof: dict[str, np.ndarray],
    facies_means: dict[str, tuple[float, float]],
    target_name: str,
    alpha: float,
) -> float | None:
    all_true, all_pred = [], []
    for w in wells:
        pred_ml = oof_ml_physical.get(w.well_id)
        p_sand = p_sand_oof.get(w.well_id)
        if pred_ml is None or p_sand is None:
            continue
        mean_sand, mean_shale = facies_means.get(w.well_id, (float("nan"), float("nan")))
        if not (np.isfinite(mean_sand) and np.isfinite(mean_shale)):
            continue
        facies_baseline = p_sand * mean_sand + (1.0 - p_sand) * mean_shale
        pred_blend = (1.0 - alpha) * pred_ml + alpha * facies_baseline
        y_true = w.targets[target_name]
        valid = np.isfinite(y_true) & np.isfinite(pred_blend)
        if not valid.any():
            continue
        all_true.append(y_true[valid])
        all_pred.append(pred_blend[valid])
    if not all_true:
        return None
    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    if len(all_true_arr) < 2 or np.std(all_true_arr) == 0:
        return None
    from sklearn.metrics import r2_score

    return float(r2_score(all_true_arr, all_pred_arr))


@dataclass
class _ModelSelection:
    status: str  # "selected" or "insufficient_data"
    model_name: str | None
    cv_r2: float | None
    oof_predictions_physical: dict[str, np.ndarray]
    facies_alpha: float | None = None  # None unless this selection is a facies-blended variant


def select_best_model(
    templates: dict[str, object],
    wells: list[_SweetSpotWellSamples],
    X_scaled: dict[str, np.ndarray],
    target_scales: dict[str, _WellTargetScale],
    target_name: str,
    p_sand_oof: dict[str, np.ndarray] | None = None,
) -> _ModelSelection:
    """Evaluates every base template (and, if p_sand_oof is given and
    target_name is facies-eligible, every (template, alpha) blend on top
    of that SAME template's own OOF predictions -- no extra model fits)
    via LOGO-CV, picks by MAX CV R^2 ONLY."""
    best: _ModelSelection | None = None
    facies_means = _facies_means_per_fold(wells, target_name) if (p_sand_oof and target_name in FACIES_TARGETS) else None

    for name, factory in templates.items():
        r2, oof = logo_cv_score(factory, wells, X_scaled, target_scales, target_name)
        if r2 is not None and (best is None or r2 > best.cv_r2):
            best = _ModelSelection(status="selected", model_name=name, cv_r2=r2, oof_predictions_physical=oof)

        if facies_means is not None and r2 is not None:
            for alpha in FACIES_ALPHAS:
                if alpha == 0.0:
                    continue  # alpha=0 reduces exactly to the un-blended candidate already scored above
                blended_r2 = _facies_blend_cv_r2(wells, oof, p_sand_oof, facies_means, target_name, alpha)
                if blended_r2 is not None and (best is None or blended_r2 > best.cv_r2):
                    best = _ModelSelection(
                        status="selected", model_name=name, cv_r2=blended_r2,
                        oof_predictions_physical=oof, facies_alpha=alpha,
                    )

    if best is None:
        return _ModelSelection(status="insufficient_data", model_name=None, cv_r2=None, oof_predictions_physical={})
    return best


@dataclass
class _FittedTarget:
    status: str  # "trained" or "insufficient_data"
    model_name: str | None = None
    cv_r2: float | None = None
    facies_alpha: float | None = None
    final_model: object | None = None
    calibration: DynamicCalibration | None = None
    oof_predictions_physical: dict[str, np.ndarray] | None = None
    facies_final: dict | None = None  # {"classifier": ..., "mean_sand": float, "mean_shale": float} if facies-blended


def _fit_final(model_factory, wells, X_scaled, target_scales, target_name) -> tuple[object, DynamicCalibration | None]:
    calib = fit_dynamic_calibration(wells, target_name, target_scales)
    X_parts, y_parts, w_parts = [], [], []
    for w in wells:
        X = X_scaled[w.well_id]
        y = target_scales[w.well_id].y_scaled
        weight = np.full(len(y), w.tie_weight)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() == 0:
            continue
        X_parts.append(X[valid])
        y_parts.append(y[valid])
        w_parts.append(weight[valid])
    model = model_factory()
    if X_parts:
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        w_all = np.concatenate(w_parts)
        _fit_with_optional_weight(model, X_all, y_all, w_all)
    return model, calib


def _fit_final_facies_classifier(wells, X_scaled) -> object | None:
    from sklearn.ensemble import RandomForestClassifier

    X_parts, y_parts = [], []
    for w in wells:
        X = X_scaled[w.well_id]
        y = w.targets["is_sand"]
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() == 0:
            continue
        X_parts.append(X[valid])
        y_parts.append(y[valid])
    if not X_parts:
        return None
    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts).astype(int)
    if len(np.unique(y_all)) < 2:
        return None
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    clf.fit(X_all, y_all)
    return clf


def _facies_means_all_training(wells: list[_SweetSpotWellSamples], target_name: str) -> tuple[float, float]:
    sand_vals, shale_vals = [], []
    for w in wells:
        y = w.targets[target_name]
        is_sand = w.targets["is_sand"]
        valid = np.isfinite(y) & np.isfinite(is_sand)
        sand_vals.append(y[valid][is_sand[valid] == 1])
        shale_vals.append(y[valid][is_sand[valid] == 0])
    sand_concat = np.concatenate(sand_vals) if sand_vals else np.array([])
    shale_concat = np.concatenate(shale_vals) if shale_vals else np.array([])
    mean_sand = float(np.mean(sand_concat)) if len(sand_concat) else float("nan")
    mean_shale = float(np.mean(shale_concat)) if len(shale_concat) else float("nan")
    return mean_sand, mean_shale


def train_stage1(training_wells: list[_SweetSpotWellSamples], feature_names: list[str]) -> dict[str, _FittedTarget]:
    """AI, DT, PHIT independently -- the feasible subset of the doc's
    AI/DT/MURHO/PHIT/POIS/VPVS (no DTS curve available in this survey's
    LAS files, see sweet_spot_training_data's module docstring)."""
    if len(training_wells) < MIN_TRAINING_WELLS:
        return {t: _FittedTarget(status="insufficient_data") for t in STAGE1_TARGETS}

    templates = _make_base_templates()
    feature_scales = per_well_zscore_features(training_wells, feature_names)
    X_scaled = {w.well_id: feature_scales[w.well_id].X_scaled for w in training_wells}

    results: dict[str, _FittedTarget] = {}
    for target_name in STAGE1_TARGETS:
        target_scales = per_well_winsorize_and_zscore_target(training_wells, target_name)
        selection = select_best_model(templates, training_wells, X_scaled, target_scales, target_name)
        if selection.status != "selected":
            results[target_name] = _FittedTarget(status="insufficient_data")
            continue
        final_model, calib = _fit_final(templates[selection.model_name], training_wells, X_scaled, target_scales, target_name)
        results[target_name] = _FittedTarget(
            status="trained", model_name=selection.model_name, cv_r2=selection.cv_r2,
            final_model=final_model, calibration=calib, oof_predictions_physical=selection.oof_predictions_physical,
        )
    return results


def train_stage2(
    training_wells: list[_SweetSpotWellSamples], feature_names: list[str], stage1_results: dict[str, _FittedTarget]
) -> dict[str, _FittedTarget]:
    """GR, RHOB, VSH, PHIE, SWE (feasible subset, no LMRHO) using
    feature_names + Stage 1's OOF cascade columns (oof_ai/oof_dt/oof_phit).
    GR/VSH/PHIE/SWE additionally sweep facies-modulation alphas on top of
    each candidate's own OOF predictions (see module docstring)."""
    if len(training_wells) < MIN_TRAINING_WELLS:
        return {t: _FittedTarget(status="insufficient_data") for t in STAGE2_TARGETS}

    templates = _make_base_templates()
    feature_scales = per_well_zscore_features(training_wells, feature_names)

    cascade_names = [f"oof_{t}" for t in STAGE1_TARGETS]
    X_scaled: dict[str, np.ndarray] = {}
    for w in training_wells:
        base = feature_scales[w.well_id].X_scaled
        extra_cols = []
        for t in STAGE1_TARGETS:
            res = stage1_results.get(t)
            arr = (res.oof_predictions_physical or {}).get(w.well_id) if res and res.status == "trained" else None
            if arr is None:
                arr = np.full(base.shape[0], np.nan)
            finite = np.isfinite(arr)
            mean = float(np.nanmean(arr)) if finite.any() else 0.0
            std = float(np.nanstd(arr)) if finite.any() else 1.0
            std = std if std > 0 and np.isfinite(std) else 1.0
            extra_cols.append((arr - mean) / std)
        extra = np.stack(extra_cols, axis=1) if extra_cols else np.zeros((base.shape[0], 0))
        X_scaled[w.well_id] = np.concatenate([base, extra], axis=1)

    p_sand_oof = _fit_facies_classifier_oof(training_wells, X_scaled)

    results: dict[str, _FittedTarget] = {}
    for target_name in STAGE2_TARGETS:
        target_scales = per_well_winsorize_and_zscore_target(training_wells, target_name)
        use_facies = target_name in FACIES_TARGETS
        selection = select_best_model(
            templates, training_wells, X_scaled, target_scales, target_name,
            p_sand_oof=p_sand_oof if use_facies else None,
        )
        if selection.status != "selected":
            results[target_name] = _FittedTarget(status="insufficient_data")
            continue

        final_model, calib = _fit_final(templates[selection.model_name], training_wells, X_scaled, target_scales, target_name)
        facies_final = None
        if selection.facies_alpha is not None:
            clf = _fit_final_facies_classifier(training_wells, X_scaled)
            mean_sand, mean_shale = _facies_means_all_training(training_wells, target_name)
            facies_final = {"classifier": clf, "mean_sand": mean_sand, "mean_shale": mean_shale}

        results[target_name] = _FittedTarget(
            status="trained", model_name=selection.model_name, cv_r2=selection.cv_r2,
            facies_alpha=selection.facies_alpha, final_model=final_model, calibration=calib,
            oof_predictions_physical=selection.oof_predictions_physical, facies_final=facies_final,
        )
    return results


@dataclass
class SweetSpotCascade:
    stage1: dict[str, _FittedTarget]
    stage2: dict[str, _FittedTarget]
    feature_names: tuple[str, ...]
    training_well_ids: tuple[str, ...]


def train_sweet_spot_cascade(training_wells: list[_SweetSpotWellSamples], feature_names: list[str]) -> SweetSpotCascade:
    """Top-level: Stage 1 -> Stage 2, both selected by max CV R^2 only.
    Callers separately handle blind-well diagnostic evaluation -- this
    function only ever sees training_wells, never the blind well."""
    stage1 = train_stage1(training_wells, feature_names)
    stage2 = train_stage2(training_wells, feature_names, stage1)
    return SweetSpotCascade(
        stage1=stage1, stage2=stage2, feature_names=tuple(feature_names),
        training_well_ids=tuple(w.well_id for w in training_wells),
    )
