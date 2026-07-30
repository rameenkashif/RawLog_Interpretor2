"""
services/prediction_pipeline_service.py
------------------------------------------
Blind-well VSH/PHIE/SWE prediction, ported from a user-supplied reference
pipeline (full_pipeline.py / full_pipeline_2.py), upgraded to the
"improved_pipeline.py" fix set the same author supplied after the first
version's pooled LOOCV R^2 came back catastrophically negative (-2 to
-6): raw 222-bin CWT/SSWT amplitude on ~40 samples/well was badly
overfitting. Reproduced here:

  1. PCA on the CWT/SSWT spectrum before fitting (optional per config --
     see point 4).
  2. Instantaneous attributes (envelope, cos/sin phase, instantaneous
     frequency, all via a Hilbert transform on the well's own trace) as
     extra low-dimensional features (optional per config).
  3. Spatial context: envelope mean/std over a 5x5 neighboring-trace
     window (porosity/lithology often has some lateral continuity).
  4. A per-property model config -- (spectrum, use_instantaneous_attrs,
     n_pca_components, model_name) -- chosen by an ACTUAL grid search run
     by this module (run_grid_search / get_best_config below), not a
     hardcoded value copied from an external run. This matters: after the
     well-tie fix (point 5) changed how much real signal is actually in
     the spectrum, the true best config for a property can change (e.g.
     SWE's winner moved from PCA-3+Ridge to no-PCA raw-spectrum+
     RandomForest once the tie was corrected) -- a hardcoded config from
     a run against the OLD tie would silently stay wrong. Cached
     in-process per target after the first search (see
     _GRID_SEARCH_CACHE) since a full search is expensive -- see
     run_grid_search's docstring for the cost and how to force a rerun.
  5. Well->trace tie: resolved via direct_tie_service.resolve_direct_tie
     (the proven direct nearest-trace + DPTM full-window-search tie
     already used elsewhere in this app) instead of the reference
     scripts' hand-fit, survey-specific coordinate anchors -- the ONLY
     change from the reference pipelines' own steps 3-4. Feature/target
     alignment is still by NEAREST seismic time-sample index with
     duplicate-depth averaging (matching the reference pipeline exactly,
     not this app's usual continuous interpolation). direct_tie_service
     itself now anchors to a real checkshot station when one is available
     for the well (see checkshot_service.py / well_seismic_tie.
     checkshot_anchor_shift) instead of a blind +/-100ms statistical
     search -- this module doesn't need to know which; it just gets a
     more accurate tie automatically. Each well's tie_source
     ("checkshot (N valid pt)" vs. "statistical_fallback") is surfaced in
     the prediction response so this is never silently assumed.

Deliberately a SEPARATE model/page from spectral_property_prediction_
service.py (RandomForest on the full raw spectrum always, no PCA/
instantaneous attrs/config search), not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services import direct_tie_service as dts
from app.services import well_service

TARGET_LAS_NAMES = {"vsh": "VSH", "phie": "PHIE", "swe": "SWE"}
METHOD_ENERGY_KEYS = {"cwt": "energy", "sswt": "sswt_amplitude"}
METHOD_FREQ_KEYS = {"cwt": "freq_hz", "sswt": "sswt_freq_hz"}

# A model config: (spectrum, use_instantaneous_attrs, n_pca_components_or_None, model_name).
Config = tuple[str, bool, int | None, str]

# The grid search space, matching the reference "improved_pipeline"'s own
# description of what it searched: spectrum x PCA component count
# (including "no PCA" = raw spectrum) x with/without instantaneous attrs
# x model family. 2 x 3 x 2 x 4 = 48 candidates per property.
GRID_SPECTRA: tuple[str, ...] = ("cwt", "sswt")
GRID_PCA_OPTIONS: tuple[int | None, ...] = (None, 3, 5)
GRID_USE_INST_OPTIONS: tuple[bool, ...] = (True, False)
GRID_MODEL_NAMES: tuple[str, ...] = ("ridge", "pls", "rf", "gb")

RIDGE_ALPHA = 5.0
SPATIAL_WINDOW = 2  # 5x5 neighboring-trace window (2*2+1)

MIN_VALID_SAMPLES = 5

# In-process cache: target -> {"best_config": Config, "best_r2": float|None,
# "leaderboard": [...]}. Populated lazily by get_best_config the first time
# each target is requested; persists for the life of the running backend
# process (survives across requests, not across a restart) -- see
# run_grid_search's docstring for why this matters and how to force a
# rerun (e.g. after wells change).
_GRID_SEARCH_CACHE: dict[str, dict] = {}


@dataclass
class WellPredictionFeatures:
    well_id: str
    inline_number: int
    crossline_number: int
    correlation: float
    best_freq_hz: float
    polarity: int
    bulk_shift_ms: float
    depths_m: np.ndarray
    twt_ms: np.ndarray
    targets: dict[str, np.ndarray]  # 'vsh'/'phie'/'swe' -> (n_samples,), aligned to depths_m/twt_ms
    features: dict[str, np.ndarray]  # 'cwt'/'sswt' -> (n_samples, n_freq)
    freq_hz: dict[str, np.ndarray]  # 'cwt'/'sswt' -> (n_freq,)
    inst_attrs: np.ndarray  # (n_samples, 6): envelope, cos(phase), sin(phase), inst_freq, env_nbhd_mean, env_nbhd_std
    tie_source: str = "unknown"  # "checkshot (N valid pt)" or "statistical_fallback" -- see direct_tie_service


def _extract_curve(rows: list[dict], name: str) -> np.ndarray:
    arr = np.array(
        [row.get(name) if row.get(name) is not None else np.nan for row in rows], dtype=float
    )
    arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
    return arr


def _spatial_neighbor_trace_indices(volume, inline_number: int, crossline_number: int) -> list[int]:
    """Trace indices for the SPATIAL_WINDOW x SPATIAL_WINDOW neighborhood
    (5x5 by default) of inlines/crosslines around (inline_number,
    crossline_number), via the volume's dense grid lookup -- skips gaps
    (grid_trace_idx == -1) rather than erroring, same table
    well_zone_tie_service's IDW interpolation uses for exactly this kind
    of "nearby traces" query."""
    geometry = volume.get_grid_geometry()
    grid_trace_idx = geometry["grid_trace_idx"]
    inlines_sorted = geometry["inlines_sorted"]
    crosslines_sorted = geometry["crosslines_sorted"]

    il_pos = int(np.searchsorted(inlines_sorted, inline_number))
    xl_pos = int(np.searchsorted(crosslines_sorted, crossline_number))
    il_lo, il_hi = max(0, il_pos - SPATIAL_WINDOW), min(len(inlines_sorted), il_pos + SPATIAL_WINDOW + 1)
    xl_lo, xl_hi = max(0, xl_pos - SPATIAL_WINDOW), min(len(crosslines_sorted), xl_pos + SPATIAL_WINDOW + 1)

    window = grid_trace_idx[il_lo:il_hi, xl_lo:xl_hi]
    return [int(idx) for idx in window.flatten() if idx >= 0]


def _envelope(trace: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert

    return np.abs(hilbert(trace.astype(np.float64)))


def _instantaneous_attrs_full_trace(trace: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """envelope, phase (unwrapped), instantaneous frequency (Hz) for one
    full trace via a Hilbert transform -- same recipe as the reference
    pipeline's compute_instantaneous_attributes, applied per-trace here
    instead of across a pre-built 3D cube."""
    from scipy.signal import hilbert

    analytic = hilbert(trace.astype(np.float64))
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi * (dt_ms / 1000.0))
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  # pad to full length, edge-mode
    return envelope, phase, inst_freq


def _instantaneous_attrs_at_indices(
    volume, trace_idx: int, inline_number: int, crossline_number: int, sample_idx: np.ndarray, dt_ms: float
) -> np.ndarray:
    """(len(sample_idx), 6) feature matrix: envelope, cos(phase),
    sin(phase), instantaneous frequency (all from this well's own trace),
    plus the SPATIAL_WINDOW x SPATIAL_WINDOW neighborhood's mean/std
    envelope -- mirrors the reference "improved_pipeline"'s
    build_rich_features exactly, sourced via this volume's trace/grid
    lookups instead of a pre-built cube."""
    trace = volume.get_trace(trace_idx)
    envelope, phase, inst_freq = _instantaneous_attrs_full_trace(trace, dt_ms)

    neighbor_idxs = _spatial_neighbor_trace_indices(volume, inline_number, crossline_number)
    neighbor_envelopes = [_envelope(volume.get_trace(idx)) for idx in neighbor_idxs] or [envelope]
    neighbor_stack = np.stack(neighbor_envelopes, axis=0)
    env_nbhd_mean = neighbor_stack.mean(axis=0)
    env_nbhd_std = neighbor_stack.std(axis=0)

    return np.stack([
        envelope[sample_idx],
        np.cos(phase[sample_idx]),
        np.sin(phase[sample_idx]),
        inst_freq[sample_idx],
        env_nbhd_mean[sample_idx],
        env_nbhd_std[sample_idx],
    ], axis=1)


def _build_well_features(volume, well_id: str) -> WellPredictionFeatures:
    """Raises TieError/WellNotFoundError/SegyVolumeError -- caller treats
    that as "excluded", never silently proceeding on a well that
    couldn't actually be tied or aligned."""
    tie = dts.resolve_direct_tie(volume, well_id)

    result = volume.get_spectral_decomposition_trace(
        tie.inline_number, tie.crossline_number, method="cwt", include_sswt=True
    )
    time_ms = np.array(result["time_ms"], dtype=float)

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth_all = _extract_curve(rows, "DEPT")
    dptm_all = _extract_curve(rows, "DPTM")
    targets_all = {k: _extract_curve(rows, v) for k, v in TARGET_LAS_NAMES.items()}

    valid = np.isfinite(depth_all) & np.isfinite(dptm_all)
    for arr in targets_all.values():
        valid &= np.isfinite(arr)
    if valid.sum() < MIN_VALID_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}' has too few samples with DEPT, DPTM, and all of VSH/PHIE/SWE "
            f"valid together ({int(valid.sum())}, need >= {MIN_VALID_SAMPLES})."
        )

    depth_v = depth_all[valid]
    t_shifted = dptm_all[valid] + tie.bulk_shift_ms
    targets_v = {k: v[valid] for k, v in targets_all.items()}

    in_range = (t_shifted >= time_ms[0]) & (t_shifted <= time_ms[-1])
    if in_range.sum() < MIN_VALID_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}''s logged interval doesn't overlap the seismic recording window "
            "after the direct tie's bulk shift."
        )
    depth_v = depth_v[in_range]
    t_shifted = t_shifted[in_range]
    targets_v = {k: v[in_range] for k, v in targets_v.items()}

    # Nearest seismic-time-sample index per depth (matching the reference
    # pipeline exactly, not this app's usual continuous interpolation),
    # then average away duplicate depths landing on the same sample --
    # routine given log sampling is far finer than the seismic's ~2-4ms.
    idxs = np.array([int(np.argmin(np.abs(time_ms - t))) for t in t_shifted])
    uniq_idx, inverse = np.unique(idxs, return_inverse=True)

    def _agg(arr: np.ndarray) -> np.ndarray:
        return np.array([arr[inverse == k].mean() for k in range(len(uniq_idx))])

    agg_depth = _agg(depth_v)
    agg_targets = {k: _agg(v) for k, v in targets_v.items()}

    features: dict[str, np.ndarray] = {}
    freq_hz: dict[str, np.ndarray] = {}
    for method, energy_key in METHOD_ENERGY_KEYS.items():
        arr = np.array(result[energy_key], dtype=float)  # (n_time, n_freq)
        features[method] = arr[uniq_idx, :]
        freq_hz[method] = np.array(result[METHOD_FREQ_KEYS[method]], dtype=float)

    inst_attrs = _instantaneous_attrs_at_indices(
        volume, tie.trace_idx, tie.inline_number, tie.crossline_number, uniq_idx, volume.sample_interval_ms
    )

    return WellPredictionFeatures(
        well_id=well_id,
        inline_number=tie.inline_number,
        crossline_number=tie.crossline_number,
        correlation=tie.correlation,
        best_freq_hz=tie.best_freq_hz,
        polarity=tie.polarity,
        bulk_shift_ms=tie.bulk_shift_ms,
        depths_m=agg_depth,
        twt_ms=time_ms[uniq_idx],
        targets=agg_targets,
        features=features,
        freq_hz=freq_hz,
        inst_attrs=inst_attrs,
        tie_source=tie.tie_source,
    )


def _eligible_well_features(volume) -> tuple[dict[str, WellPredictionFeatures], list[dict]]:
    features: dict[str, WellPredictionFeatures] = {}
    excluded: list[dict] = []
    for summary in well_service.list_well_summaries():
        try:
            features[summary.well_id] = _build_well_features(volume, summary.well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            excluded.append({"well_id": summary.well_id, "reason": str(exc)})
    return features, excluded


def _assemble_features(
    well_features: dict[str, WellPredictionFeatures], well_id: str, config: Config
) -> np.ndarray:
    spec_key, use_inst, _n_pca, _model_name = config
    X = well_features[well_id].features[spec_key]
    if use_inst:
        X = np.concatenate([X, well_features[well_id].inst_attrs], axis=1)
    return X


def _fit_model(well_features: dict[str, WellPredictionFeatures], train_ids: list[str], target: str, config: Config):
    """Fit the given config's exact recipe: StandardScaler -> optional PCA
    -> {Ridge, PLS, RandomForest, GradientBoosting}. Returns (scaler,
    pca_or_None, model, n_train_samples) so the SAME fitted pipeline can
    also be applied to a full inline's worth of traces
    (render_property_inline_maps_image) without duplicating the fit
    logic."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    _spec_key, _use_inst, n_pca, model_name = config

    X_train = np.vstack([_assemble_features(well_features, w, config) for w in train_ids])
    y_train = np.concatenate([well_features[w].targets[target] for w in train_ids])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    pca = None
    if n_pca is not None:
        n_components = min(n_pca, X_train_s.shape[0] - 1, X_train_s.shape[1])
        if n_components < 1:
            raise ValueError(f"Not enough training samples ({X_train_s.shape[0]}) for PCA-{n_pca}.")
        pca = PCA(n_components=n_components, random_state=0).fit(X_train_s)
        X_train_s = pca.transform(X_train_s)

    if model_name == "ridge":
        model = Ridge(alpha=RIDGE_ALPHA).fit(X_train_s, y_train)
    elif model_name == "pls":
        n_components = min(5, X_train_s.shape[1], X_train_s.shape[0] - 1)
        model = PLSRegression(n_components=n_components).fit(X_train_s, y_train)
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=0
        ).fit(X_train_s, y_train)
    elif model_name == "gb":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=0
        ).fit(X_train_s, y_train)
    else:
        raise ValueError(f"Unknown model_name {model_name!r} in config {config!r}.")

    return scaler, pca, model, int(len(y_train))


def _apply_model(scaler, pca, model, X: np.ndarray) -> np.ndarray:
    X_s = scaler.transform(X)
    if pca is not None:
        X_s = pca.transform(X_s)
    pred = model.predict(X_s)
    if pred.ndim > 1:
        pred = pred.ravel()  # PLSRegression.predict returns (n, 1)
    return np.clip(pred, 0.0, 1.0)


def _predict_blind(
    well_features: dict[str, WellPredictionFeatures], blind_well_id: str, target: str, config: Config
) -> dict:
    from sklearn.metrics import r2_score

    train_ids = [w for w in well_features if w != blind_well_id]
    scaler, pca, model, n_train_samples = _fit_model(well_features, train_ids, target, config)
    blind = well_features[blind_well_id]
    X_test = _assemble_features(well_features, blind_well_id, config)
    y_test = blind.targets[target]
    y_pred = _apply_model(scaler, pca, model, X_test)

    r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else None
    return {
        "r2": r2,
        "n_train_samples": n_train_samples,
        "n_train_wells": len(train_ids),
        "n_test_samples": int(len(y_test)),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "depths_m": blind.depths_m.tolist(),
        "twt_ms": blind.twt_ms.tolist(),
    }


def _pooled_loocv_r2(well_features: dict[str, WellPredictionFeatures], target: str, config: Config) -> float | None:
    """Pooled leave-one-well-out R^2 for ONE config: every well takes a
    turn held out, true/pred pooled across all of them into a single
    r2_score -- the honest selection metric run_grid_search uses (not a
    single well's score, which is easy to overfit to by cherry-picking a
    config that happens to help one well)."""
    from sklearn.metrics import r2_score

    well_ids = list(well_features.keys())
    all_true: list[float] = []
    all_pred: list[float] = []
    for held_out in well_ids:
        pred = _predict_blind(well_features, held_out, target, config)
        all_true.extend(pred["y_true"])
        all_pred.extend(pred["y_pred"])
    return float(r2_score(all_true, all_pred)) if len(all_true) >= 2 else None


def get_full_loocv_r2(target: str) -> dict:
    """TRUE leave-one-well-out under this target's grid-search-selected
    config (get_best_config): every well takes a turn as the held-out
    well, pooled into a single R^2. Returns {'r2': float|None,
    'n_wells': int, 'per_well': {well_id: r2_or_None}}."""
    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)
    well_ids = list(well_features.keys())
    if len(well_ids) < 2:
        return {"r2": None, "n_wells": len(well_ids), "per_well": {}}

    from sklearn.metrics import r2_score

    config = get_best_config(target)
    per_well: dict[str, float | None] = {}
    all_true: list[float] = []
    all_pred: list[float] = []
    for held_out in well_ids:
        pred = _predict_blind(well_features, held_out, target, config)
        per_well[held_out] = pred["r2"]
        all_true.extend(pred["y_true"])
        all_pred.extend(pred["y_pred"])

    pooled_r2 = float(r2_score(all_true, all_pred)) if len(all_true) >= 2 else None
    return {"r2": pooled_r2, "n_wells": len(well_ids), "per_well": per_well}


_MODEL_LABELS = {"ridge": "Ridge", "pls": "PLS", "rf": "RandomForest", "gb": "GradientBoosting"}


def _config_description(config: Config) -> str:
    spec_key, use_inst, n_pca, model_name = config
    model_label = _MODEL_LABELS.get(model_name, model_name)
    inst_label = "+ instantaneous attrs" if use_inst else ""
    pca_label = f"PCA-{n_pca}" if n_pca is not None else "no PCA (raw spectrum)"
    return f"{spec_key.upper()} {inst_label}, {pca_label}, {model_label}".replace("  ", " ").strip()


def run_grid_search(target: str) -> dict:
    """Search GRID_SPECTRA x GRID_PCA_OPTIONS x GRID_USE_INST_OPTIONS x
    GRID_MODEL_NAMES (48 candidates) for `target`, scoring each by pooled
    leave-one-well-out R^2 (every well held out in turn) -- the SAME
    selection process and metric the reference "improved_pipeline" used,
    run here against THIS app's own (corrected) well ties/features rather
    than trusting a config chosen against a different tie. A candidate
    that raises (e.g. a degenerate PCA/model fit) is recorded in the
    leaderboard with r2=None and skipped as a possible winner, not
    allowed to crash the whole search.

    Cost: up to 48 candidates x n_wells pooled-LOOCV folds = e.g. ~330
    fit/predict cycles for a 7-well field -- on the order of tens of
    seconds to a few minutes depending on how many GradientBoosting/
    RandomForest candidates end up in the mix. Well features
    (tie + CWT/SSWT + instantaneous attrs) are resolved ONCE and reused
    across every candidate, so this cost is all in repeated model
    fitting, not repeated tie/spectral-decomposition work. See
    get_best_config for the cache that avoids paying this more than once
    per target per process lifetime.
    """
    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, excluded = _eligible_well_features(volume)
    if len(well_features) < 2:
        return {
            "target": target,
            "best_config": None,
            "best_r2": None,
            "n_wells": len(well_features),
            "excluded_wells": excluded,
            "leaderboard": [],
        }

    leaderboard: list[dict] = []
    for spec_key in GRID_SPECTRA:
        for n_pca in GRID_PCA_OPTIONS:
            for use_inst in GRID_USE_INST_OPTIONS:
                for model_name in GRID_MODEL_NAMES:
                    config: Config = (spec_key, use_inst, n_pca, model_name)
                    try:
                        r2 = _pooled_loocv_r2(well_features, target, config)
                        error = None
                    except Exception as exc:  # noqa: BLE001 -- one bad candidate shouldn't kill the search
                        r2 = None
                        error = str(exc)
                    leaderboard.append({
                        "config": config,
                        "description": _config_description(config),
                        "r2": r2,
                        "error": error,
                    })

    scored = [entry for entry in leaderboard if entry["r2"] is not None]
    scored.sort(key=lambda entry: entry["r2"], reverse=True)
    best = scored[0] if scored else None

    leaderboard.sort(key=lambda entry: (entry["r2"] is None, -(entry["r2"] or 0.0)))
    return {
        "target": target,
        "best_config": best["config"] if best else None,
        "best_r2": best["r2"] if best else None,
        "n_wells": len(well_features),
        "excluded_wells": excluded,
        "leaderboard": leaderboard,
    }


# Fallback used only if a grid search finds literally no working
# candidate (e.g. too few wells for PCA/model fitting to succeed at all)
# -- the simplest, least failure-prone recipe, not a claim it's good.
_FALLBACK_CONFIG: Config = ("sswt", False, None, "ridge")


def get_best_config(target: str, force: bool = False) -> Config:
    """This target's grid-search-selected config, cached in-process after
    the first call (see _GRID_SEARCH_CACHE) -- a real search is expensive
    (run_grid_search), so it isn't re-run on every prediction request.
    Pass force=True to invalidate the cache and re-search (e.g. after
    wells/ties changed enough that the old winner might not hold)."""
    if force or target not in _GRID_SEARCH_CACHE:
        _GRID_SEARCH_CACHE[target] = run_grid_search(target)
    result = _GRID_SEARCH_CACHE[target]
    return result["best_config"] or _FALLBACK_CONFIG


def get_grid_search_result(target: str, force: bool = False) -> dict:
    """JSON-friendly grid search result for the /prediction-grid-search
    endpoint: triggers/reuses the same cache get_best_config does (so
    calling this also "warms" get_best_config's cache for `target`, and
    vice versa -- one shared cache entry, not two)."""
    if target not in TARGET_LAS_NAMES:
        raise ValueError(f"target must be one of {list(TARGET_LAS_NAMES)}, got {target!r}.")

    if force or target not in _GRID_SEARCH_CACHE:
        _GRID_SEARCH_CACHE[target] = run_grid_search(target)
    result = _GRID_SEARCH_CACHE[target]

    best_config = result["best_config"]
    return {
        "target": target,
        "best_config_description": _config_description(best_config) if best_config else None,
        "best_r2": result["best_r2"],
        "n_wells": result["n_wells"],
        "excluded_wells": result["excluded_wells"],
        "leaderboard": [
            {"description": entry["description"], "r2": entry["r2"], "error": entry["error"]}
            for entry in result["leaderboard"]
        ],
    }


def get_prediction_result(blind_well_id: str, target: str) -> dict:
    if target not in TARGET_LAS_NAMES:
        raise ValueError(f"target must be one of {list(TARGET_LAS_NAMES)}, got {target!r}.")

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, excluded = _eligible_well_features(volume)
    config = get_best_config(target)

    base = {
        "blind_well_id": blind_well_id,
        "target": target,
        "model_config_description": _config_description(config),
        "excluded_wells": excluded,
    }

    empty_tie_fields = {
        "inline_number": None,
        "crossline_number": None,
        "tie_correlation": None,
        "tie_best_freq_hz": None,
        "tie_polarity": None,
        "tie_bulk_shift_ms": None,
        "tie_source": None,
    }
    if blind_well_id not in well_features:
        return {
            **base,
            **empty_tie_fields,
            "status": "insufficient_data",
            "message": f"'{blind_well_id}' does not currently have a usable direct tie -- see excluded_wells for why.",
            "n_train_wells": 0,
            "result": None,
        }
    if len(well_features) < 2:
        return {
            **base,
            **empty_tie_fields,
            "status": "insufficient_data",
            "message": (
                f"Only {len(well_features)} well(s) have a usable tie -- need at least 2 (the "
                "blind well plus at least one other to train on)."
            ),
            "n_train_wells": 0,
            "result": None,
        }

    pred = _predict_blind(well_features, blind_well_id, target, config)
    blind = well_features[blind_well_id]
    return {
        **base,
        "status": "validated",
        "message": None,
        "inline_number": blind.inline_number,
        "crossline_number": blind.crossline_number,
        "tie_correlation": blind.correlation,
        "tie_best_freq_hz": blind.best_freq_hz,
        "tie_polarity": blind.polarity,
        "tie_bulk_shift_ms": blind.bulk_shift_ms,
        "tie_source": blind.tie_source,
        "n_train_wells": pred["n_train_wells"],
        "result": pred,
    }


def render_full_loocv_heatmap_image() -> bytes:
    """PNG: TRUE pooled leave-one-well-out R^2 (every well takes a turn
    held out, see get_full_loocv_r2) for each of VSH/PHIE/SWE under its
    own grid-search-selected config (get_best_config) -- the reference
    "improved_pipeline"'s own honest selection metric, computed here
    rather than trusted from an external run. Independent of which well
    is currently selected as "blind" elsewhere on the page."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)

    targets = list(TARGET_LAS_NAMES.keys())
    values = np.full(len(targets), np.nan)
    n_wells = len(well_features)

    if n_wells >= 2:
        for i, target in enumerate(targets):
            loocv = get_full_loocv_r2(target)
            if loocv["r2"] is not None:
                values[i] = loocv["r2"]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=140)
    grid = values.reshape(1, -1)
    finite = grid[np.isfinite(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(len(targets)), [t.upper() for t in targets])
    for j, target in enumerate(targets):
        val = grid[0, j]
        text = f"{val:.3f}" if np.isfinite(val) else "n/a"
        ax.text(
            j, 0, f"{text}\n{_config_description(get_best_config(target))}",
            ha="center", va="center", color="black", fontsize=8,
        )
    ax.set_title(f"Full LOOCV pooled R² -- improved pipeline (n={n_wells} wells)")
    fig.colorbar(im, ax=ax, label="R² (can be negative -- worse than the mean)", fraction=0.046)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_frequency_map_image(blind_well_id: str) -> bytes:
    """PNG: amplitude-spectrum frequency map (frequency x crossline,
    averaged over time) for the inline through the given well -- a plain
    FFT (much cheaper than CWT/SSWT), giving a spatial view of which
    frequencies carry energy along that inline, e.g. for spotting a
    low-frequency shadow. Reference pipeline step 2 (amplitude_spectrum_
    cube) computed this for the whole volume but never actually plotted
    it -- this renders it for one inline, which is what's actually useful
    to look at."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    tie = dts.resolve_direct_tie(volume, blind_well_id)
    section = volume.get_inline_section(tie.inline_number)
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=int)

    dt_s = volume.sample_interval_ms / 1000.0
    spectrum = np.abs(np.fft.rfft(amplitude, axis=0))  # (n_freq, n_traces)
    freqs = np.fft.rfftfreq(amplitude.shape[0], d=dt_s)
    nyquist = freqs[-1]
    band = freqs <= min(nyquist, 100.0)  # seismic energy is rarely useful above ~100 Hz

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(freqs[band], spectrum[band].mean(axis=1), color="#2563EB")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean amplitude")
    axes[0].set_title(f"Average spectrum -- Inline {tie.inline_number}")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(
        spectrum[band], aspect="auto", cmap="viridis",
        extent=[position_axis[0], position_axis[-1], freqs[band][-1], freqs[band][0]],
    )
    axes[1].axvline(tie.crossline_number, color="white", lw=1, ls="--", alpha=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Crossline")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"Frequency map -- through {blind_well_id}")
    fig.colorbar(im, ax=axes[1], label="Amplitude", fraction=0.046)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_property_inline_maps_image(blind_well_id: str) -> bytes:
    """PNG: for VSH, PHIE, and SWE, a real-vs-predicted pair along the
    blind well's own inline, each using ITS OWN grid-search-selected
    config (get_best_config: PCA + optional instantaneous attrs + the
    per-property best model family) -- LEFT is the actual (unmodified)
    seismic amplitude section, RIGHT is the SAME inline with the
    blind-well-trained model's predicted property value painted across
    EVERY trace along it (not just the well's own location), each
    property in its own colormap. A real property value doesn't exist
    away from a well, so "real" here means the actual seismic data, not a
    true property map.

    Can be heavy depending on what the grid search picked: any property
    whose winning config uses SSWT needs every trace along the inline to
    get its own SSWT (the more expensive transform), and any property
    using instantaneous attrs needs a 5x5 spatial-neighborhood envelope
    average at every trace too -- touching several NEIGHBORING inlines'
    traces, not just this one. Per-trace envelope/spectral results are
    cached and reused across all 3 properties and across overlapping
    spatial windows rather than recomputed.

    EXPLORATORY, not a validated spatial map: even this improved
    pipeline's pooled LOOCV R^2 has typically been small on real wells --
    a confidently colored map does not mean the colors are trustworthy.
    That caveat is rendered directly into the image, not just documented
    here.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)
    if blind_well_id not in well_features:
        raise ValueError(f"'{blind_well_id}' does not currently have a usable direct tie.")
    if len(well_features) < 2:
        raise ValueError(f"Only {len(well_features)} well(s) have a usable tie -- need at least 2.")

    targets = list(TARGET_LAS_NAMES.keys())
    configs = {t: get_best_config(t) for t in targets}
    train_ids = [w for w in well_features if w != blind_well_id]
    fitted = {t: _fit_model(well_features, train_ids, t, configs[t]) for t in targets}
    blind = well_features[blind_well_id]

    section = volume.get_inline_section(blind.inline_number)
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=int)
    twt_axis_ms = np.array(section["twt_axis_ms"], dtype=float)
    vmax = float(np.percentile(np.abs(amplitude), 98)) or 1e-6
    dt_ms = volume.sample_interval_ms

    envelope_cache: dict[int, np.ndarray] = {}

    def _cached_envelope(idx: int) -> np.ndarray:
        if idx not in envelope_cache:
            envelope_cache[idx] = _envelope(volume.get_trace(idx))
        return envelope_cache[idx]

    n_time = len(twt_axis_ms)
    pred_grids = {t: np.full((n_time, len(position_axis)), np.nan) for t in targets}
    spec_cache: dict[str, np.ndarray] = {}
    need_sswt = any(configs[t][0] == "sswt" for t in targets)
    need_inst = any(configs[t][1] for t in targets)

    geometry = volume.get_grid_geometry()
    il_pos = int(np.searchsorted(geometry["inlines_sorted"], blind.inline_number))

    for j, xl in enumerate(position_axis):
        xl = int(xl)
        try:
            spec = volume.get_spectral_decomposition_trace(
                blind.inline_number, xl, method="cwt", include_sswt=need_sswt
            )
        except Exception:  # noqa: BLE001 -- a gap trace shouldn't abort the whole map
            continue
        spec_cache["cwt"] = np.array(spec["energy"], dtype=float)
        if need_sswt:
            spec_cache["sswt"] = np.array(spec["sswt_amplitude"], dtype=float)

        xl_pos = int(np.searchsorted(geometry["crosslines_sorted"], xl))
        trace_idx = int(geometry["grid_trace_idx"][il_pos, xl_pos])
        if trace_idx < 0:
            continue

        inst_attrs_full = None
        if need_inst:
            envelope, phase, inst_freq = _instantaneous_attrs_full_trace(volume.get_trace(trace_idx), dt_ms)
            neighbor_idxs = _spatial_neighbor_trace_indices(volume, blind.inline_number, xl)
            neighbor_stack = np.stack([_cached_envelope(idx) for idx in neighbor_idxs] or [envelope], axis=0)
            env_nbhd_mean = neighbor_stack.mean(axis=0)
            env_nbhd_std = neighbor_stack.std(axis=0)
            inst_attrs_full = np.stack(
                [envelope, np.cos(phase), np.sin(phase), inst_freq, env_nbhd_mean, env_nbhd_std], axis=1
            )  # (n_time, 6)

        for t in targets:
            spec_key, use_inst, _n_pca, _model_name = configs[t]
            X_line = spec_cache[spec_key]
            if use_inst:
                X_line = np.concatenate([X_line, inst_attrs_full], axis=1)
            scaler, pca, model, _n = fitted[t]
            pred_grids[t][:, j] = _apply_model(scaler, pca, model, X_line)

    cmaps = {"vsh": "YlOrBr", "phie": "Greens", "swe": "Blues"}
    extent = [position_axis[0], position_axis[-1], twt_axis_ms[-1], twt_axis_ms[0]]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True, sharey=True)
    for row, t in enumerate(targets):
        ax_real, ax_pred = axes[row]
        ax_real.imshow(amplitude, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, extent=extent)
        ax_real.set_title(f"Real seismic -- Inline {blind.inline_number}" if row == 0 else "")
        ax_real.set_ylabel("Two-Way Time (ms)")

        im = ax_pred.imshow(pred_grids[t], aspect="auto", cmap=cmaps[t], vmin=0.0, vmax=1.0, extent=extent)
        ax_pred.set_title(f"Predicted {t.upper()} ({_config_description(configs[t])})")
        fig.colorbar(im, ax=ax_pred, label=t.upper(), fraction=0.04)

        for ax in (ax_real, ax_pred):
            ax.axvline(blind.crossline_number, color="black", lw=1, ls="--", alpha=0.7)
        ax_real.text(
            blind.crossline_number, twt_axis_ms[0], f" {blind_well_id}",
            fontsize=8, va="bottom", color="black",
        )

    for ax in axes[-1]:
        ax.set_xlabel("Crossline")

    fig.suptitle(
        f"EXPLORATORY -- blind well: {blind_well_id}, improved pipeline (PCA-3 + instantaneous "
        "attrs + per-property best model). Pooled LOOCV R² has still been small/near zero on real "
        "wells (see LOOCV heatmap): colors show the model's estimate, not a validated spatial result.",
        fontsize=10, color="firebrick", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_prediction_image(blind_well_id: str, target: str) -> bytes:
    """PNG bytes: side-by-side inline-section images (TRUE vs. PREDICTED
    target, painted as a colored strip at the blind well's crossline
    position) -- direct port of the reference pipeline's
    plot_inline_true_vs_pred, using this module's tie/features and
    grid-search-selected config (get_best_config) instead."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from app.services import seismic_processor as sp

    data = get_prediction_result(blind_well_id, target)
    if data["status"] != "validated":
        raise ValueError(data["message"])

    volume = sp.get_segy_volume()
    section = volume.get_inline_section(data["inline_number"])
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=float)
    twt_axis_ms = np.array(section["twt_axis_ms"], dtype=float)
    vmax = float(np.percentile(np.abs(amplitude), 98)) or 1e-6

    result = data["result"]
    true_vals = np.array(result["y_true"], dtype=float)
    pred_vals = np.array(result["y_pred"], dtype=float)
    twt_vals = np.array(result["twt_ms"], dtype=float)
    xl0 = float(data["crossline_number"])

    norm = Normalize(vmin=0.0, vmax=max(float(true_vals.max()), float(pred_vals.max()), 1e-6))
    cmap = plt.cm.YlOrBr
    xl_step = float(position_axis[1] - position_axis[0]) if len(position_axis) > 1 else 1.0
    strip_half_traces = 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    titles = [f"TRUE {target.upper()}", f"PREDICTED {target.upper()} (blind: {blind_well_id})"]
    for ax, title, vals in zip(axes, titles, [true_vals, pred_vals]):
        ax.imshow(
            amplitude, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
            extent=[position_axis[0], position_axis[-1], twt_axis_ms[-1], twt_axis_ms[0]],
        )
        for t, v in zip(twt_vals, vals):
            ax.add_patch(plt.Rectangle(
                (xl0 - strip_half_traces * xl_step, t - 4),
                2 * strip_half_traces * xl_step, 8,
                color=cmap(norm(v)),
            ))
        ax.axvline(xl0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xlim(xl0 - 40 * xl_step, xl0 + 40 * xl_step)
        ax.set_ylim(float(twt_vals.max()) + 20, float(twt_vals.min()) - 20)
        ax.set_xlabel("Crossline")
        ax.set_title(title)
    axes[0].set_ylabel("Two-Way Time (ms)")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, fraction=0.03, label=target.upper())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
