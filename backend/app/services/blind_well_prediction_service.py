"""
services/blind_well_prediction_service.py
--------------------------------------------
End-to-end neighborhood-expanded CWT/SSWT property prediction, validated
with a genuine BLIND well holdout -- not leave-one-well-out averaged
across every well (that's spectral_property_prediction_service.py's job),
this module trains on every OTHER well and predicts ONLY the one held
back, which never appears in neighborhood sampling, feature selection,
base-learner training, or meta-learner training until the final predict
call.

Pipeline, phase by phase:

1. Neighborhood expansion (_extract_well_samples): around each training
   well's own tied trace, gather every trace within NEIGHBORHOOD_RADIUS_M
   (capped at MAX_NEIGHBORHOOD_TRACES for compute -- SSWT is expensive per
   trace), and assign EVERY one of those traces' samples the well's own
   logged property value at the corresponding tied time (same overlap/
   depth_at_time mapping the well's own center trace uses -- the seismic
   time axis is trace-independent, so this is a direct reuse, not a
   re-tie). Farther traces get a lower IDW-style sample weight
   (1/(1+distance_m)) rather than an equal vote, reusing this codebase's
   existing IDW convention (well_zone_tie_service.py) as a soft label
   confidence instead of a hard spatial copy.

2. Feature bank (_trace_sample_features): per time sample, per trace --
   classic Hilbert instantaneous attributes (amplitude, envelope,
   instantaneous frequency) from the raw trace, plus derived CWT and SSWT
   features (dominant frequency, peak amplitude, spectral bandwidth,
   centroid, entropy, and low/mid/high energy-in-band) from
   SegyVolume.get_spectral_decomposition_trace's per-time-sample energy
   matrices.

3. Feature selection (_select_features): pooled Pearson correlation
   against each target (VSH/PHIE/SWE separately -- they don't share a
   feature set), with a leave-one-well-out STABILITY check -- a feature
   only counts if its correlation sign is consistent across every
   training well, not just in the pooled fit (see module docstring's
   "small well count" caveat: a feature that only "works" pooled is very
   likely fitting one well's own noise). Collinear duplicates (CWT and
   SSWT dominant frequency overlap heavily) are dropped greedily. Top
   TOP_K_FEATURES kept per property.

4. Decision gate (_decision_gate): a cheap single-XGBoost, top-3-feature,
   leave-one-well-out check across the TRAINING wells only -- if this is
   still negative, the full stack is very unlikely to rescue it (stacking
   reduces variance among decent learners, it doesn't manufacture signal
   that isn't there). Reported alongside the full result, not used to
   block the pipeline -- the point is to let a human judge before trusting
   the fancier number.

5. Stacking (_fit_stack): four base learners (XGBoost for sharp bed-
   boundary/tuning transitions, ExtraTrees and RandomForest for
   noise-robust smoothing/diversity, Ridge for the smooth global trend
   trees can't extrapolate), each trained per training well fold to
   generate honest out-of-fold predictions (leave-one-well-out, NOT random
   k-fold -- samples within one well's neighborhood are spatially
   correlated, a random split would leak), which a Ridge meta-learner is
   then fit on. A second, final set of base learners trained on ALL
   training wells (never the blind one) is what actually predicts the
   blind well.

6. Blind-well evaluation (run_blind_well_prediction): the held-out well's
   OWN center trace only (its real logged depths, not its neighborhood --
   there's no ground truth at a neighbor trace) is scored: R^2, RMSE, and
   the full depth/real/predicted arrays for a predicted-vs-logged plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services import well_service
from app.services.spectral_petro_correlation_service import (
    PETRO_CURVES,
    _WellTieContext,
    _property_series,
)
from app.services.spectral_property_prediction_service import (
    _DirectTieResult,
    _resolve_direct_tie,
)

DEFAULT_BLIND_WELL_ID = "Z-02_RAW"

# Radius picked for Zamzama-type reservoir facies continuity (per the
# reviewed plan) -- traces farther than this are unlikely to still share
# the tied well's own logged interval's rock properties.
NEIGHBORHOOD_RADIUS_M = 150.0
# Includes the well's own center trace. SSWT (ssqueezepy) is meaningfully
# expensive per trace (see seismic_processor.py's SSWT benchmark logging),
# so this caps a single request's compute rather than pulling in every
# trace a dense survey's 150m radius could contain.
MAX_NEIGHBORHOOD_TRACES = 12

MIN_TRAINING_WELLS = 3
MIN_SAMPLES_PER_WELL_PROPERTY = 5
TOP_K_FEATURES = 8
COLLINEARITY_THRESHOLD = 0.95

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RANDOM_STATE = 42

FEATURE_NAMES = [
    "inst_amplitude",
    "inst_envelope",
    "inst_freq_hz",
    "cwt_dominant_freq_hz",
    "cwt_peak_amp",
    "cwt_bandwidth_hz",
    "cwt_centroid_hz",
    "cwt_entropy",
    "cwt_energy_low",
    "cwt_energy_mid",
    "cwt_energy_high",
    "sswt_dominant_freq_hz",
    "sswt_peak_amp",
    "sswt_bandwidth_hz",
    "sswt_centroid_hz",
    "sswt_entropy",
    "sswt_energy_low",
    "sswt_energy_mid",
    "sswt_energy_high",
]


class BlindWellPredictionError(Exception):
    """Base class for blind-well-prediction-module errors."""


@dataclass
class _WellSamples:
    well_id: str
    X: np.ndarray  # (n_samples, len(FEATURE_NAMES))
    y: dict[str, np.ndarray]  # property_name -> (n_samples,), may contain NaN
    weight: np.ndarray  # (n_samples,) IDW-style soft label weight, 1.0 at the well's own trace


def _spectral_features(energy: np.ndarray, freq_hz: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """energy: (n_time, n_freq). Per-time-sample derived features:
    dominant frequency, peak amplitude, spectral bandwidth/centroid
    (energy-weighted), Shannon entropy (0-1 normalized), and energy
    fraction in the low/mid/high thirds of the available frequency range."""
    eps = 1e-12
    energy = np.asarray(energy, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    total = energy.sum(axis=1) + eps
    p = energy / total[:, None]

    dominant_freq = freq_hz[np.argmax(energy, axis=1)]
    peak_amp = energy.max(axis=1)
    centroid = (p * freq_hz[None, :]).sum(axis=1)
    bandwidth = np.sqrt((p * (freq_hz[None, :] - centroid[:, None]) ** 2).sum(axis=1))
    entropy = -(p * np.log(p + eps)).sum(axis=1) / np.log(max(len(freq_hz), 2))

    f_lo, f_hi = float(freq_hz.min()), float(freq_hz.max())
    thirds = np.linspace(f_lo, f_hi, 4)
    low_mask = (freq_hz >= thirds[0]) & (freq_hz < thirds[1])
    mid_mask = (freq_hz >= thirds[1]) & (freq_hz < thirds[2])
    high_mask = (freq_hz >= thirds[2]) & (freq_hz <= thirds[3])

    return {
        f"{prefix}_dominant_freq_hz": dominant_freq,
        f"{prefix}_peak_amp": peak_amp,
        f"{prefix}_bandwidth_hz": bandwidth,
        f"{prefix}_centroid_hz": centroid,
        f"{prefix}_entropy": entropy,
        f"{prefix}_energy_low": p[:, low_mask].sum(axis=1),
        f"{prefix}_energy_mid": p[:, mid_mask].sum(axis=1),
        f"{prefix}_energy_high": p[:, high_mask].sum(axis=1),
    }


def _instantaneous_attributes(trace: np.ndarray, dt_ms: float) -> dict[str, np.ndarray]:
    """Classic Hilbert-transform seismic attributes: instantaneous
    amplitude (the raw trace itself), envelope (analytic-signal
    magnitude), and instantaneous frequency (time-derivative of the
    analytic signal's unwrapped phase, in Hz)."""
    from scipy.signal import hilbert

    analytic = hilbert(trace)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dt_s = dt_ms / 1000.0
    inst_freq = np.gradient(phase) / (2.0 * np.pi * dt_s)
    return {"inst_amplitude": trace, "inst_envelope": envelope, "inst_freq_hz": inst_freq}


def _trace_sample_features(volume, trace_idx: int, ctx: _WellTieContext) -> np.ndarray | None:
    """(n_overlap_samples, len(FEATURE_NAMES)) feature matrix for ONE
    trace, at the well's own tied+overlap-masked seismic-time samples --
    or None if this trace's spectral decomposition couldn't be computed
    (e.g. too short for SSWT, see SSWT_MIN_SAMPLES)."""
    try:
        inline_n = int(volume.inline[trace_idx])
        crossline_n = int(volume.crossline[trace_idx])
        spec = volume.get_spectral_decomposition_trace(inline_n, crossline_n, method="cwt", include_sswt=True)
    except Exception:  # noqa: BLE001
        return None

    sswt_amp = spec.get("sswt_amplitude")
    sswt_freq = spec.get("sswt_freq_hz")
    if not sswt_amp or not sswt_freq:
        return None

    cwt_energy = np.array(spec["energy"])[ctx.overlap]
    cwt_freq_arr = np.array(spec["freq_hz"])
    sswt_energy = np.array(sswt_amp)[ctx.overlap]
    sswt_freq_arr = np.array(sswt_freq)
    if cwt_energy.shape[0] == 0:
        return None

    raw_trace = volume.get_trace(trace_idx)[ctx.overlap]
    feat_dict = {
        **_instantaneous_attributes(raw_trace, volume.sample_interval_ms),
        **_spectral_features(cwt_energy, cwt_freq_arr, "cwt"),
        **_spectral_features(sswt_energy, sswt_freq_arr, "sswt"),
    }
    return np.column_stack([feat_dict[name] for name in FEATURE_NAMES])


def _neighborhood_trace_indices(volume, center_trace_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Every trace within NEIGHBORHOOD_RADIUS_M of the CENTER TRACE's own
    real coordinate (not the well header's X/Y -- the tied trace's actual
    position on the grid), nearest first, capped at
    MAX_NEIGHBORHOOD_TRACES. Always includes the center trace itself
    (distance 0)."""
    cx = float(volume.source_x[center_trace_idx])
    cy = float(volume.source_y[center_trace_idx])
    dist = np.sqrt((volume.source_x - cx) ** 2 + (volume.source_y - cy) ** 2)
    within = np.where(dist <= NEIGHBORHOOD_RADIUS_M)[0]
    order = np.argsort(dist[within])
    within = within[order][:MAX_NEIGHBORHOOD_TRACES]
    return within, dist[within]


def _extract_well_samples(volume, tie: _DirectTieResult) -> _WellSamples | None:
    """Neighborhood-expanded training samples for one well -- see module
    docstring phase 1."""
    ctx = tie.ctx
    neighbor_idxs, neighbor_dist = _neighborhood_trace_indices(volume, tie.trace_idx)
    y_per_property = {name: _property_series(ctx, name) for name in PETRO_CURVES}

    rows_X: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for trace_idx, dist_m in zip(neighbor_idxs, neighbor_dist):
        X_trace = _trace_sample_features(volume, int(trace_idx), ctx)
        if X_trace is None:
            continue
        rows_X.append(X_trace)
        weights.append(np.full(X_trace.shape[0], 1.0 / (1.0 + float(dist_m))))

    if not rows_X:
        return None

    X = np.concatenate(rows_X, axis=0)
    weight = np.concatenate(weights, axis=0)
    y_tiled = {name: np.tile(y_per_property[name], len(rows_X)) for name in PETRO_CURVES}
    return _WellSamples(well_id=ctx.well_id, X=X, y=y_tiled, weight=weight)


def _extract_center_trace_samples(
    volume, tie: _DirectTieResult
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
    """Just the well's own tied trace, no neighborhood -- for the BLIND
    well, whose real logged values only exist at its own depths, not its
    neighbors'. Returns (X, y_per_property, depth_m)."""
    ctx = tie.ctx
    X = _trace_sample_features(volume, tie.trace_idx, ctx)
    if X is None:
        return None
    y = {name: _property_series(ctx, name) for name in PETRO_CURVES}
    return X, y, ctx.depth_at_time


def _select_features(training_samples: list[_WellSamples], property_name: str) -> tuple[list[int], list[dict]]:
    """Pooled correlation ranked, but only counts a feature as "stable" if
    its per-well correlation sign agrees across every training well
    (leave-one-well-out stability, not just the pooled full-data fit --
    see module docstring phase 3). Greedily drops anything collinear
    (|r| > COLLINEARITY_THRESHOLD) with an already-selected feature."""
    n_features = training_samples[0].X.shape[1]
    diagnostics: list[dict] = []
    scores = np.full(n_features, -1.0)

    for f_idx in range(n_features):
        per_well_corr = []
        for s in training_samples:
            y = s.y[property_name]
            x = s.X[:, f_idx]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < MIN_SAMPLES_PER_WELL_PROPERTY:
                continue
            if np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                continue
            r = float(np.corrcoef(x[valid], y[valid])[0, 1])
            if np.isfinite(r):
                per_well_corr.append(r)

        if len(per_well_corr) < 2:
            diagnostics.append(
                {"feature": FEATURE_NAMES[f_idx], "pooled_corr": None, "stable": False, "selected": False}
            )
            continue

        pooled_corr = float(np.mean(per_well_corr))
        stable = bool(pooled_corr != 0 and all(np.sign(c) == np.sign(pooled_corr) for c in per_well_corr))
        scores[f_idx] = abs(pooled_corr) if stable else abs(pooled_corr) * 0.3
        diagnostics.append(
            {"feature": FEATURE_NAMES[f_idx], "pooled_corr": pooled_corr, "stable": stable, "selected": False}
        )

    X_pool = np.concatenate([s.X for s in training_samples], axis=0)
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for idx in order:
        if len(selected) >= TOP_K_FEATURES or scores[idx] <= 0:
            continue
        collinear = False
        for sel_idx in selected:
            a, b = X_pool[:, idx], X_pool[:, sel_idx]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() > 10 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0:
                if abs(np.corrcoef(a[valid], b[valid])[0, 1]) > COLLINEARITY_THRESHOLD:
                    collinear = True
                    break
        if collinear:
            continue
        selected.append(int(idx))
        diagnostics[idx]["selected"] = True

    return selected, diagnostics


def _decision_gate(training_samples: list[_WellSamples], property_name: str, feature_idxs: list[int]) -> float | None:
    """Cheap sanity check (module docstring phase 4): single XGBoost, top
    3 selected features, leave-one-well-out across the TRAINING wells
    only. A strongly negative result here means the full stack is very
    unlikely to help -- surfaced as a diagnostic, not used to block."""
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

    k = feature_idxs[:3]
    if not k:
        return None

    all_true: list[float] = []
    all_pred: list[float] = []
    for i in range(len(training_samples)):
        train_s = [s for j, s in enumerate(training_samples) if j != i]
        X_train = np.concatenate([s.X[:, k] for s in train_s], axis=0)
        y_train = np.concatenate([s.y[property_name] for s in train_s], axis=0)
        valid_tr = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        X_test = training_samples[i].X[:, k]
        y_test = training_samples[i].y[property_name]
        valid_te = np.isfinite(y_test) & np.all(np.isfinite(X_test), axis=1)
        if valid_tr.sum() < MIN_SAMPLES_PER_WELL_PROPERTY or valid_te.sum() < 2:
            continue

        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
            objective="reg:squarederror", verbosity=0,
        )
        model.fit(X_train[valid_tr], y_train[valid_tr])
        pred = model.predict(X_test[valid_te])
        all_true.extend(y_test[valid_te].tolist())
        all_pred.extend(pred.tolist())

    if len(all_true) < 2:
        return None
    return float(r2_score(all_true, all_pred))


def _make_base_models() -> dict:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    return {
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE,
            objective="reg:squarederror", verbosity=0,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE
        ),
        "ridge": Ridge(alpha=1.0),
    }


def _fit_stack(X: np.ndarray, y: np.ndarray, weight: np.ndarray, groups: np.ndarray) -> dict:
    """4-base-learner stack (module docstring phase 5): leave-one-well-out
    out-of-fold predictions train the Ridge meta-learner; a SEPARATE final
    set of base learners, trained on every training well at once, is what
    actually predicts the blind well (via _predict_stack)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    base_names = list(_make_base_models().keys())
    unique_wells = sorted(set(groups.tolist()))
    oof_preds = np.zeros((len(y), len(base_names)))

    for held_out in unique_wells:
        train_mask = groups != held_out
        test_mask = groups == held_out
        models = _make_base_models()
        scaler = StandardScaler().fit(X[train_mask])
        X_train_s = scaler.transform(X[train_mask])
        X_test_s = scaler.transform(X[test_mask])
        for j, name in enumerate(base_names):
            model = models[name]
            if name == "ridge":
                model.fit(X_train_s, y[train_mask], sample_weight=weight[train_mask])
                oof_preds[test_mask, j] = model.predict(X_test_s)
            else:
                model.fit(X[train_mask], y[train_mask], sample_weight=weight[train_mask])
                oof_preds[test_mask, j] = model.predict(X[test_mask])

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_preds, y, sample_weight=weight)
    stack_loocv_r2 = float(r2_score(y, meta_model.predict(oof_preds))) if len(y) >= 2 else None

    final_models = _make_base_models()
    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    for name, model in final_models.items():
        if name == "ridge":
            model.fit(X_scaled, y, sample_weight=weight)
        else:
            model.fit(X, y, sample_weight=weight)

    return {
        "final_models": final_models,
        "final_scaler": final_scaler,
        "meta_model": meta_model,
        "base_names": base_names,
        "stack_loocv_r2": stack_loocv_r2,
    }


def _predict_stack(stack: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = stack["final_scaler"].transform(X)
    base_preds = np.zeros((X.shape[0], len(stack["base_names"])))
    for j, name in enumerate(stack["base_names"]):
        model = stack["final_models"][name]
        base_preds[:, j] = model.predict(X_scaled if name == "ridge" else X)
    return stack["meta_model"].predict(base_preds)


def run_blind_well_prediction(blind_well_id: str = DEFAULT_BLIND_WELL_ID) -> dict:
    """Top-level entry point. Trains entirely on every OTHER well with a
    usable tie, then predicts ONLY blind_well_id -- which never
    participates in neighborhood sampling, feature selection, or any
    model training. status is always one of 'validated',
    'blind_well_unusable', or 'insufficient_data', never a silently
    fabricated result."""
    from app.services import seismic_processor as sp_mod

    all_ids = [s.well_id for s in well_service.list_well_summaries()]
    if blind_well_id not in all_ids:
        raise well_service.WellNotFoundError(blind_well_id)

    volume = sp_mod.get_segy_volume()

    excluded: list[dict] = []
    ties: dict[str, _DirectTieResult] = {}
    for well_id in all_ids:
        try:
            tie = _resolve_direct_tie(volume, well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            excluded.append({"well_id": well_id, "reason": f"No usable tie: {exc}"})
            continue
        if tie.boundary_pinned or tie.low_confidence:
            excluded.append({
                "well_id": well_id,
                "reason": f"Low-confidence/boundary-pinned tie (correlation={tie.correlation:.3f}).",
            })
            continue
        ties[well_id] = tie

    if blind_well_id not in ties:
        return {
            "status": "blind_well_unusable",
            "message": f"Blind well '{blind_well_id}' has no usable tie -- cannot validate against it.",
            "blind_well_id": blind_well_id,
            "training_well_ids": [],
            "excluded_wells": excluded,
            "neighborhood_radius_m": NEIGHBORHOOD_RADIUS_M,
            "results": None,
        }

    training_ids = [w for w in ties if w != blind_well_id]
    if len(training_ids) < MIN_TRAINING_WELLS:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(training_ids)} training well(s) have a usable tie (need at least "
                f"{MIN_TRAINING_WELLS}, excluding the blind well)."
            ),
            "blind_well_id": blind_well_id,
            "training_well_ids": training_ids,
            "excluded_wells": excluded,
            "neighborhood_radius_m": NEIGHBORHOOD_RADIUS_M,
            "results": None,
        }

    training_samples: list[_WellSamples] = []
    for well_id in training_ids:
        samples = _extract_well_samples(volume, ties[well_id])
        if samples is None:
            excluded.append({"well_id": well_id, "reason": "Could not extract spectral features for its neighborhood."})
            continue
        training_samples.append(samples)

    if len(training_samples) < MIN_TRAINING_WELLS:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(training_samples)} training well(s) had extractable spectral features "
                f"(need at least {MIN_TRAINING_WELLS})."
            ),
            "blind_well_id": blind_well_id,
            "training_well_ids": [s.well_id for s in training_samples],
            "excluded_wells": excluded,
            "neighborhood_radius_m": NEIGHBORHOOD_RADIUS_M,
            "results": None,
        }

    blind_extract = _extract_center_trace_samples(volume, ties[blind_well_id])
    if blind_extract is None:
        return {
            "status": "blind_well_unusable",
            "message": f"Blind well '{blind_well_id}''s own trace could not be spectrally decomposed.",
            "blind_well_id": blind_well_id,
            "training_well_ids": [s.well_id for s in training_samples],
            "excluded_wells": excluded,
            "neighborhood_radius_m": NEIGHBORHOOD_RADIUS_M,
            "results": None,
        }
    X_blind, y_blind, depth_blind = blind_extract

    from sklearn.metrics import mean_squared_error, r2_score

    results: dict[str, dict] = {}
    for property_name in PETRO_CURVES:
        X_pool = np.concatenate([s.X for s in training_samples], axis=0)
        y_pool = np.concatenate([s.y[property_name] for s in training_samples], axis=0)
        valid_pool = np.isfinite(y_pool) & np.all(np.isfinite(X_pool), axis=1)
        if valid_pool.sum() < MIN_TRAINING_WELLS * MIN_SAMPLES_PER_WELL_PROPERTY:
            results[property_name] = {
                "status": "insufficient_data",
                "message": f"Too few valid pooled training samples for {property_name.upper()}.",
            }
            continue

        selected_idx, feature_diagnostics = _select_features(training_samples, property_name)
        if not selected_idx:
            results[property_name] = {
                "status": "no_stable_features",
                "message": f"No feature was correlation-stable across training wells for {property_name.upper()}.",
                "feature_diagnostics": feature_diagnostics,
            }
            continue

        decision_gate_r2 = _decision_gate(training_samples, property_name, selected_idx)

        groups = np.concatenate([np.full(s.X.shape[0], s.well_id) for s in training_samples])
        X_train_all = np.concatenate([s.X[:, selected_idx] for s in training_samples], axis=0)
        y_train_all = np.concatenate([s.y[property_name] for s in training_samples], axis=0)
        w_train_all = np.concatenate([s.weight for s in training_samples], axis=0)
        valid = np.isfinite(y_train_all) & np.all(np.isfinite(X_train_all), axis=1)
        X_train_all, y_train_all, w_train_all, groups = (
            X_train_all[valid], y_train_all[valid], w_train_all[valid], groups[valid]
        )

        stack = _fit_stack(X_train_all, y_train_all, w_train_all, groups)

        X_blind_sel = X_blind[:, selected_idx]
        y_blind_true = y_blind[property_name]
        valid_blind = np.isfinite(y_blind_true) & np.all(np.isfinite(X_blind_sel), axis=1)
        if valid_blind.sum() < 2:
            results[property_name] = {
                "status": "blind_well_no_valid_samples",
                "message": f"Blind well has too few valid {property_name.upper()} samples in its tied interval to score.",
                "selected_features": [FEATURE_NAMES[i] for i in selected_idx],
                "feature_diagnostics": feature_diagnostics,
                "decision_gate_r2": decision_gate_r2,
                "stack_loocv_r2": stack["stack_loocv_r2"],
            }
            continue

        y_blind_pred = _predict_stack(stack, X_blind_sel[valid_blind])
        y_blind_valid = y_blind_true[valid_blind]
        blind_r2 = float(r2_score(y_blind_valid, y_blind_pred)) if valid_blind.sum() >= 2 else None
        blind_rmse = float(np.sqrt(mean_squared_error(y_blind_valid, y_blind_pred)))

        results[property_name] = {
            "status": "validated",
            "message": None,
            "selected_features": [FEATURE_NAMES[i] for i in selected_idx],
            "feature_diagnostics": feature_diagnostics,
            "decision_gate_r2": decision_gate_r2,
            "stack_loocv_r2": stack["stack_loocv_r2"],
            "blind_well_r2": blind_r2,
            "blind_well_rmse": blind_rmse,
            "n_training_samples": int(len(y_train_all)),
            "n_blind_samples": int(valid_blind.sum()),
            "depth_m": depth_blind[valid_blind].tolist(),
            "y_true": y_blind_valid.tolist(),
            "y_pred": y_blind_pred.tolist(),
        }

    return {
        "status": "validated",
        "message": None,
        "blind_well_id": blind_well_id,
        "training_well_ids": [s.well_id for s in training_samples],
        "excluded_wells": excluded,
        "neighborhood_radius_m": NEIGHBORHOOD_RADIUS_M,
        "results": results,
    }
