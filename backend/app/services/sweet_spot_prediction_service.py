"""
services/sweet_spot_prediction_service.py
----------------------------------------------
Orchestration + region prediction for the Sweet-Spot module: trains the
2-stage cascade on every well except a held-out blind well (genuinely
blind -- never touches training, feature selection, or model selection,
same discipline as blind_well_prediction_service.py), evaluates that
blind well diagnostically, and serves arbitrary inline x crossline "sweet
spot" region predictions for the interactive section UI.

Region-prediction feature scaling (NEW judgment call beyond the source
doc's own scope -- it only ever scores AT well locations): an arbitrary
sweet-spot location has no "well" to normalize against, so features there
are scaled using the MEDIAN of the training wells' own per-feature (mean,
std) -- consistent with sweet_spot_calibration's own "robust median"
philosophy elsewhere in this pipeline, rather than inventing a different,
region-local scaling rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app import petrophysics as pp
from app.repository import MODELS_DIR
from app.services.sweet_spot_calibration import apply_calibration, per_well_zscore_features
from app.services.sweet_spot_feature_engine import CURATED_FEATURES, compute_feature_pool
from app.services.sweet_spot_model_service import MIN_TRAINING_WELLS, train_sweet_spot_cascade
from app.services.sweet_spot_training_data import (
    STAGE1_TARGETS,
    STAGE2_TARGETS,
    _SweetSpotWellSamples,
    build_training_pool,
)

DEFAULT_BLIND_WELL_ID = "Z-02_RAW"
# Guards against an unreasonably large region request (same convention as
# blind_well_prediction_service.MAX_NEIGHBORHOOD_TRACES), sized to
# comfortably cover the frontend's actual usage pattern: a single-inline
# "sweet spot" section padded by a couple of neighboring inlines for the
# feature engine's inline-gradient features (see SweetSpotSectionView.tsx's
# INLINE_PAD) across a realistic survey's full crossline width (this
# module's own source doc's real survey: 252 crosslines) -- roughly
# 5 x 300 with margin, not a genuinely large 2D areal request.
MAX_REGION_TRACES = 2000


class SweetSpotPredictionError(Exception):
    """Base class for sweet-spot-prediction-module errors."""


_trained_cascade_cache: dict[str, dict] = {}


def _cascade_cache_path(blind_well_id: str) -> Path:
    """On-disk path for a trained cascade, gitignored (backend/data/models/
    *.joblib, same convention well_service.get_core_perm_model() already
    uses) -- lets a 'validated' cascade survive a server restart without
    retraining. Never checked into git: the cascade is trained from
    whatever seismic/well data happens to be locally ingested, so a
    committed file would silently be trained against the wrong data on
    anyone else's machine."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in blind_well_id)
    return MODELS_DIR / f"sweet_spot_cascade_{safe_id}.joblib"


def _apply_facies_blend(phys_pred: np.ndarray, X_valid: np.ndarray, result) -> np.ndarray:
    """If `result` (a _FittedTarget) was selected as a facies-modulated
    blend, apply the SAME formula used during selection/scoring
    (verbatim from the doc):
        pred = (1-alpha)*pred_ML + alpha*(P_sand*mean_sand + (1-P_sand)*mean_shale)
    Returns phys_pred unchanged if this target wasn't facies-blended."""
    if result.facies_alpha is None or not result.facies_final:
        return phys_pred
    clf = result.facies_final.get("classifier")
    mean_sand = result.facies_final.get("mean_sand")
    mean_shale = result.facies_final.get("mean_shale")
    if clf is None or not (np.isfinite(mean_sand) and np.isfinite(mean_shale)):
        return phys_pred
    classes = list(clf.classes_)
    if 1 not in classes:
        return phys_pred
    p_sand = clf.predict_proba(X_valid)[:, classes.index(1)]
    facies_baseline = p_sand * mean_sand + (1.0 - p_sand) * mean_shale
    return (1.0 - result.facies_alpha) * phys_pred + result.facies_alpha * facies_baseline


def _score_target(
    result, X_valid: np.ndarray, time_valid: np.ndarray, y_true_valid: np.ndarray,
    depth_valid: np.ndarray | None = None,
) -> dict:
    """Predict + calibrate + (if applicable) facies-blend + score one
    target for the blind well's own center-trace samples."""
    from sklearn.metrics import mean_squared_error, r2_score

    z_pred = result.final_model.predict(X_valid)
    y_pred = apply_calibration(z_pred, time_valid, result.calibration)
    y_pred = _apply_facies_blend(y_pred, X_valid, result)

    valid_true = np.isfinite(y_true_valid)
    if valid_true.sum() < 2:
        return {
            "status": "blind_well_no_valid_samples",
            "message": "Blind well has too few valid samples in its tied interval to score.",
            "model_name": result.model_name, "cv_r2": result.cv_r2, "facies_alpha": result.facies_alpha,
        }

    y_true = y_true_valid[valid_true]
    y_pred_scored = y_pred[valid_true]
    blind_r2 = float(r2_score(y_true, y_pred_scored)) if len(y_true) >= 2 else None
    blind_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_scored)))

    out = {
        "status": "validated", "model_name": result.model_name, "cv_r2": result.cv_r2,
        "facies_alpha": result.facies_alpha, "blind_well_r2": blind_r2, "blind_well_rmse": blind_rmse,
        "n_blind_samples": int(valid_true.sum()),
        "time_ms": time_valid[valid_true].tolist(), "y_true": y_true.tolist(), "y_pred": y_pred_scored.tolist(),
    }
    if depth_valid is not None:
        out["depth_m"] = depth_valid[valid_true].tolist()
    return out


def _evaluate_blind_well(cascade, blind_samples: _SweetSpotWellSamples, feature_names: list[str]) -> dict[str, dict]:
    """Genuinely blind evaluation: ONLY the blind well's own center-trace
    samples are scored (never its 3x3 neighborhood -- there's no ground
    truth at a neighbor trace), and this well never appears in any of
    `cascade`'s training/selection steps."""
    feature_scales = per_well_zscore_features([blind_samples], feature_names)
    X_scaled_full = feature_scales[blind_samples.well_id].X_scaled
    X_center = X_scaled_full[blind_samples.center_row_start : blind_samples.center_row_end]
    time_center = blind_samples.time_ms[blind_samples.center_row_start : blind_samples.center_row_end]
    depth_center = blind_samples.depth_m[blind_samples.center_row_start : blind_samples.center_row_end]

    diagnostics: dict[str, dict] = {}
    stage1_preds_physical: dict[str, np.ndarray] = {}

    for target_name in STAGE1_TARGETS:
        result = cascade.stage1.get(target_name)
        if result is None or result.status != "trained":
            diagnostics[target_name] = {"status": "insufficient_data", "message": f"Stage 1 model for {target_name.upper()} was not trained."}
            continue
        valid = np.all(np.isfinite(X_center), axis=1)
        if not valid.any():
            diagnostics[target_name] = {"status": "blind_well_no_valid_samples", "message": "No valid feature samples for the blind well."}
            stage1_preds_physical[target_name] = np.full(X_center.shape[0], np.nan)
            continue

        y_true_center = blind_samples.targets[target_name][blind_samples.center_row_start : blind_samples.center_row_end]
        diagnostics[target_name] = _score_target(
            result, X_center[valid], time_center[valid], y_true_center[valid], depth_center[valid],
        )
        z_pred_all = np.full(X_center.shape[0], np.nan)
        z_pred_all[valid] = apply_calibration(
            result.final_model.predict(X_center[valid]), time_center[valid], result.calibration
        )
        stage1_preds_physical[target_name] = z_pred_all

    cascade_cols = []
    for t in STAGE1_TARGETS:
        arr = stage1_preds_physical.get(t, np.full(X_center.shape[0], np.nan))
        finite = np.isfinite(arr)
        mean = float(np.nanmean(arr)) if finite.any() else 0.0
        std = float(np.nanstd(arr)) if finite.any() else 1.0
        std = std if std > 0 and np.isfinite(std) else 1.0
        cascade_cols.append((arr - mean) / std)
    extra = np.stack(cascade_cols, axis=1) if cascade_cols else np.zeros((X_center.shape[0], 0))
    X_center_aug = np.concatenate([X_center, extra], axis=1)

    for target_name in STAGE2_TARGETS:
        result = cascade.stage2.get(target_name)
        if result is None or result.status != "trained":
            diagnostics[target_name] = {"status": "insufficient_data", "message": f"Stage 2 model for {target_name.upper()} was not trained."}
            continue
        valid = np.all(np.isfinite(X_center_aug), axis=1)
        if not valid.any():
            diagnostics[target_name] = {"status": "blind_well_no_valid_samples", "message": "No valid feature samples for the blind well."}
            continue
        y_true_center = blind_samples.targets[target_name][blind_samples.center_row_start : blind_samples.center_row_end]
        diagnostics[target_name] = _score_target(
            result, X_center_aug[valid], time_center[valid], y_true_center[valid], depth_center[valid],
        )

    return diagnostics


def get_or_train_cascade(blind_well_id: str = DEFAULT_BLIND_WELL_ID, refresh: bool = False) -> dict:
    """Top-level entry point. Trains entirely on every OTHER well with a
    usable tie, then diagnostically evaluates ONLY blind_well_id -- which
    never participates in any training/selection step. status is always
    one of 'validated', 'blind_well_unusable', or 'insufficient_data'.
    Cached per blind_well_id, in-memory first then on disk (an expensive,
    deliberately-triggered action, same convention as
    blind_well_prediction_service's own result); pass refresh=True to
    force retraining even if a cached/persisted result exists."""
    if not refresh and blind_well_id in _trained_cascade_cache:
        return _trained_cascade_cache[blind_well_id]

    cache_path = _cascade_cache_path(blind_well_id)
    if not refresh and cache_path.exists():
        try:
            result = pp.load_model(cache_path)
        except Exception:
            result = None  # stale/corrupt cache file -- fall through and retrain
        if result is not None:
            _trained_cascade_cache[blind_well_id] = result
            return result

    from app.services import seismic_processor as sp_mod
    from app.services import well_service

    all_ids = [s.well_id for s in well_service.list_well_summaries()]
    if blind_well_id not in all_ids:
        raise well_service.WellNotFoundError(blind_well_id)

    volume = sp_mod.get_segy_volume()
    usable, excluded = build_training_pool(volume, all_ids)

    blind_samples = next((w for w in usable if w.well_id == blind_well_id), None)
    training_wells = [w for w in usable if w.well_id != blind_well_id]
    feature_names = list(CURATED_FEATURES)

    if blind_samples is None:
        own_exclusion = next((e for e in excluded if e["well_id"] == blind_well_id), None)
        result = {
            "status": "blind_well_unusable",
            "message": own_exclusion["reason"] if own_exclusion else f"Blind well '{blind_well_id}' has no usable tie.",
            "blind_well_id": blind_well_id, "training_well_ids": [], "excluded_wells": excluded,
            "feature_names": feature_names, "cascade": None, "training_wells": [], "blind_results": None,
        }
        _trained_cascade_cache[blind_well_id] = result
        return result

    if len(training_wells) < MIN_TRAINING_WELLS:
        result = {
            "status": "insufficient_data",
            "message": (
                f"Only {len(training_wells)} training well(s) have a usable tie (need at least "
                f"{MIN_TRAINING_WELLS}, excluding the blind well)."
            ),
            "blind_well_id": blind_well_id, "training_well_ids": [w.well_id for w in training_wells],
            "excluded_wells": excluded, "feature_names": feature_names, "cascade": None,
            "training_wells": training_wells, "blind_results": None,
        }
        _trained_cascade_cache[blind_well_id] = result
        return result

    cascade = train_sweet_spot_cascade(training_wells, feature_names)
    blind_results = _evaluate_blind_well(cascade, blind_samples, feature_names)

    result = {
        "status": "validated", "message": None, "blind_well_id": blind_well_id,
        "training_well_ids": [w.well_id for w in training_wells], "excluded_wells": excluded,
        "feature_names": feature_names, "cascade": cascade, "training_wells": training_wells,
        "blind_results": blind_results,
    }
    _trained_cascade_cache[blind_well_id] = result
    try:
        pp.save_model(result, cache_path)
    except Exception:
        pass  # disk persistence is a convenience -- don't fail the request over it
    return result


def predict_region(
    inline_range: tuple[int, int],
    crossline_range: tuple[int, int],
    property_names: list[str],
    blind_well_id: str = DEFAULT_BLIND_WELL_ID,
) -> dict:
    """Predicts every requested property across an inline x crossline
    region using the cascade trained by get_or_train_cascade (training
    on-demand if not already cached). Stage 1 predictions for the region
    are always computed first (Stage 2 needs them as cascade features
    regardless of which properties were actually requested)."""
    from app.services import seismic_processor as sp_mod

    trained = get_or_train_cascade(blind_well_id)
    if trained["status"] != "validated":
        raise SweetSpotPredictionError(
            trained["message"] or f"Cascade for blind well '{blind_well_id}' is not trained/validated."
        )
    cascade = trained["cascade"]
    feature_names = trained["feature_names"]
    training_wells = trained["training_wells"]

    unknown = [p for p in property_names if p not in STAGE1_TARGETS and p not in STAGE2_TARGETS]
    if unknown:
        raise SweetSpotPredictionError(
            f"Unknown propert{'y' if len(unknown) == 1 else 'ies'} {unknown} -- expected one of "
            f"{list(STAGE1_TARGETS) + list(STAGE2_TARGETS)}."
        )

    volume = sp_mod.get_segy_volume()
    region = volume.get_region_traces(inline_range, crossline_range)
    traces_3d = region["traces"]
    n_il, n_xl, n_time = traces_3d.shape
    n_pos = n_il * n_xl
    if n_pos > MAX_REGION_TRACES:
        raise SweetSpotPredictionError(
            f"Requested region spans {n_pos} traces, exceeding the {MAX_REGION_TRACES}-trace limit -- "
            "narrow inline_range/crossline_range."
        )

    pool = compute_feature_pool(traces_3d, volume.sample_interval_ms)
    X_curated = np.stack([pool[name] for name in feature_names], axis=-1)  # (n_time, n_pos, n_curated)

    # Region-location feature scaling: median of training wells' own
    # per-feature (mean, std) -- see module docstring.
    feature_scales = per_well_zscore_features(training_wells, feature_names)
    means = np.stack([feature_scales[w.well_id].mean for w in training_wells], axis=0)
    stds = np.stack([feature_scales[w.well_id].std for w in training_wells], axis=0)
    median_mean = np.median(means, axis=0)
    median_std = np.median(stds, axis=0)
    median_std = np.where(median_std > 0, median_std, 1.0)

    X_scaled_3d = (X_curated - median_mean) / median_std
    X_scaled_2d = X_scaled_3d.reshape(n_time * n_pos, len(feature_names))
    valid = np.all(np.isfinite(X_scaled_2d), axis=1)
    time_flat = np.repeat(np.asarray(region["twt_axis_ms"], dtype=float), n_pos)  # matches (t*n_pos+p) row order

    stage1_region_preds: dict[str, np.ndarray] = {}
    for target_name in STAGE1_TARGETS:
        result = cascade.stage1.get(target_name)
        arr = np.full(n_time * n_pos, np.nan)
        if result is not None and result.status == "trained" and valid.any():
            z_pred = result.final_model.predict(X_scaled_2d[valid])
            phys = apply_calibration(z_pred, time_flat[valid], result.calibration)
            arr[valid] = _apply_facies_blend(phys, X_scaled_2d[valid], result)
        stage1_region_preds[target_name] = arr

    cascade_cols = []
    for t in STAGE1_TARGETS:
        arr = stage1_region_preds[t]
        finite = np.isfinite(arr)
        mean = float(np.nanmean(arr)) if finite.any() else 0.0
        std = float(np.nanstd(arr)) if finite.any() else 1.0
        std = std if std > 0 and np.isfinite(std) else 1.0
        cascade_cols.append((arr - mean) / std)
    extra = np.stack(cascade_cols, axis=1) if cascade_cols else np.zeros((n_time * n_pos, 0))
    X_scaled_2d_stage2 = np.concatenate([X_scaled_2d, extra], axis=1)
    valid_stage2 = np.all(np.isfinite(X_scaled_2d_stage2), axis=1)

    predictions: dict[str, np.ndarray] = {}
    for prop in property_names:
        if prop in STAGE1_TARGETS:
            predictions[prop] = stage1_region_preds[prop].reshape(n_time, n_pos)
            continue
        result = cascade.stage2.get(prop)
        pred_flat = np.full(n_time * n_pos, np.nan)
        if result is not None and result.status == "trained" and valid_stage2.any():
            z_pred = result.final_model.predict(X_scaled_2d_stage2[valid_stage2])
            phys = apply_calibration(z_pred, time_flat[valid_stage2], result.calibration)
            pred_flat[valid_stage2] = _apply_facies_blend(phys, X_scaled_2d_stage2[valid_stage2], result)
        predictions[prop] = pred_flat.reshape(n_time, n_pos)

    return {
        "blind_well_id": blind_well_id,
        "inline_axis": region["inline_axis"],
        "crossline_axis": region["crossline_axis"],
        "twt_axis_ms": region["twt_axis_ms"],
        "predictions": {k: v.tolist() for k, v in predictions.items()},
    }
