"""
services/spectral_property_prediction_service.py
------------------------------------------------------
Multi-frequency CWT/SSWT amplitude -> VSH/PHIE/SWE property prediction,
validated with leave-one-well-out cross-validation across wells with a
usable tie -- NOT a random depth-sample split, which would badly overstate
performance since adjacent depth samples within one well are strongly
autocorrelated. Deliberately a POINT-SOURCE validation only: this module
answers "does a real, generalizable relationship exist at the wells we
have," not "what does VSH look like across the whole seismic volume" --
extending to a volume-wide map is an explicit follow-up, gated on this
validating with real (not in-sample) skill first. See README.md/
AGENT_BRIEF.md for the full reasoning.

Two things this module gets right that the existing single-frequency
CWT/SWT/SSWT-vs-petrophysics correlation view (spectral_petro_correlation_
service.py) does not, because that view is a lighter-weight first-pass
check, not a model meant to generalize:

1. Well eligibility and time-shift correction are both driven by
   dashboard_upload_service.get_synthetic_summary(well_id) (backed by
   synthetic_seismogram_service.generate), NOT tie_service.get_well_
   seismic_tie -- because THIS module (like spectral_petro_correlation_
   service.py) resolves well->trace via coordinate_calibration_service
   against the single active SEG-Y volume, the same system synthetic_
   seismogram_service uses. tie_service resolves against a separate,
   independently-named dataset with its own DPTM-based time axis --
   using its tie confidence/shift here would be gating on and correcting
   with a tie computed against a different trace and time axis than the
   one actually used below. This means the wells this module considers
   "usable" can legitimately differ from what get_well_seismic_tie/
   get_field_overview report -- that's an architectural fact, not a bug.

2. Depth-time alignment applies the synthetic seismogram's own
   best_shift_ms correction (spectral_petro_correlation_service._resolve_
   well_tie_context's new optional time_shift_ms parameter) before
   extracting spectral features -- the existing correlation view does
   not apply this, using only the raw unshifted sonic-integrated axis.
   Leaving that misalignment in place while training a model would bias
   results toward "no relationship" regardless of whether one exists.

Boundary-pinned/low-confidence wells are excluded from training, per
well_seismic_tie.cross_correlate_and_shift's own documented guidance:
"Boundary-pinned results should be excluded from aggregate statistics
(mean correlation, ML training sets) by default."
"""

from __future__ import annotations

import numpy as np

from app.services.spectral_petro_correlation_service import (
    PETRO_CURVES,
    _property_series,
    _resolve_well_tie_context,
)

# Below this many eligible wells, leave-one-well-out is meaningless -- you
# can't hold one out and still have anything to train on.
MIN_ELIGIBLE_WELLS = 2

# A well can be "eligible" overall (usable tie) but still have too few
# valid samples for one particular property (e.g. a short logged interval
# with mostly-null PHIE) -- guarded per property/well, not just globally.
MIN_SAMPLES_PER_WELL_PROPERTY = 5

# Same hyperparameters as the existing CORE_PERM_PRED model
# (petrophysics.train_core_perm_model) -- consistency with this
# codebase's one other "regression on sparse, proxy-quality data" model,
# not independently tuned.
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_RANDOM_STATE = 42

METHODS = ("sswt", "cwt")


def _eligible_wells() -> tuple[list[str], list[dict]]:
    """Classifies every currently loaded well as eligible (has a usable
    synthetic-seismogram tie -- see module docstring for why this tie
    source, not tie_service's) or excluded (with a human-readable
    reason), never silently dropping a well without saying why."""
    from app.services import dashboard_upload_service as dus
    from app.services import well_service

    eligible: list[str] = []
    excluded: list[dict] = []

    for summary in well_service.list_well_summaries():
        well_id = summary.well_id
        synth = dus.get_synthetic_summary(well_id)
        if "error" in synth:
            excluded.append({"well_id": well_id, "reason": f"No usable tie: {synth['error']}"})
            continue
        if synth.get("boundary_pinned"):
            excluded.append({
                "well_id": well_id,
                "reason": "Shift search pinned to its boundary -- likely a spurious match, not a genuine tie.",
            })
            continue
        if synth.get("low_confidence"):
            corr = synth.get("correlation")
            corr_str = f"{corr:.3f}" if corr is not None else "n/a"
            excluded.append({
                "well_id": well_id,
                "reason": f"Low-confidence tie (correlation={corr_str}, below the 0.3 threshold).",
            })
            continue
        eligible.append(well_id)

    return eligible, excluded


def _extract_well_features(volume, well_id: str, shift_ms: float) -> dict | None:
    """One well's shift-corrected feature matrices (n_samples, n_freq) for
    both spectral methods, plus its VSH/PHIE/SWE target series aligned to
    the same samples. Returns None (caller treats as an exclusion, not a
    crash) if the well's tie/curves can't actually be resolved despite
    passing the eligibility check -- e.g. a curve edge case
    get_synthetic_summary's own resolution didn't hit."""
    try:
        ctx = _resolve_well_tie_context(volume, well_id, time_shift_ms=shift_ms)
        result = volume.get_spectral_decomposition_trace(
            ctx.inline_number, ctx.crossline_number, method="cwt", include_sswt=True
        )
    except Exception:  # noqa: BLE001
        return None

    cwt_freq_hz = np.array(result["freq_hz"])
    cwt_X = np.array(result["energy"])[ctx.overlap]  # (n_samples, n_cwt_freq)
    sswt_freq_hz = np.array(result.get("sswt_freq_hz") or [])
    sswt_amp = result.get("sswt_amplitude")
    sswt_X = np.array(sswt_amp)[ctx.overlap] if sswt_amp else np.empty((cwt_X.shape[0], 0))

    y = {name: _property_series(ctx, name) for name in PETRO_CURVES}

    return {
        "sswt_freq_hz": sswt_freq_hz,
        "sswt_X": sswt_X,
        "cwt_freq_hz": cwt_freq_hz,
        "cwt_X": cwt_X,
        "y": y,
    }


def _train_and_loocv(wells_features: dict[str, dict], method: str, property_name: str) -> dict | None:
    """Leave-one-well-out CV across every well with enough valid samples
    for this property. Returns None (not a fabricated score) if fewer
    than MIN_ELIGIBLE_WELLS have enough samples for this specific
    property, even if more wells were eligible overall."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    per_well_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for well_id, feats in wells_features.items():
        X = feats[f"{method}_X"]
        y = feats["y"][property_name]
        if X.shape[1] == 0:
            continue
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < MIN_SAMPLES_PER_WELL_PROPERTY:
            continue
        per_well_xy[well_id] = (X[valid], y[valid])

    usable_ids = list(per_well_xy.keys())
    if len(usable_ids) < MIN_ELIGIBLE_WELLS:
        return None

    per_well_results: list[dict] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    for held_out in usable_ids:
        train_ids = [w for w in usable_ids if w != held_out]
        X_train = np.concatenate([per_well_xy[w][0] for w in train_ids], axis=0)
        y_train = np.concatenate([per_well_xy[w][1] for w in train_ids], axis=0)
        X_test, y_test = per_well_xy[held_out]

        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_RANDOM_STATE
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else None
        per_well_results.append({"well_id": held_out, "r2": r2, "n_samples": int(len(y_test))})
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    pooled_r2 = float(r2_score(all_true, all_pred)) if len(all_true) >= 2 else None

    # A second model fit on ALL usable wells, for feature importance only
    # -- this is in-sample and must never be reported as (or confused
    # with) the validation score above, which is strictly held-out.
    X_all = np.concatenate([per_well_xy[w][0] for w in usable_ids], axis=0)
    y_all = np.concatenate([per_well_xy[w][1] for w in usable_ids], axis=0)
    final_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_RANDOM_STATE
    )
    final_model.fit(X_all, y_all)
    freq_axis = wells_features[usable_ids[0]][f"{method}_freq_hz"]
    feature_importance = [
        {"frequency_hz": float(f), "importance": float(imp)}
        for f, imp in zip(freq_axis, final_model.feature_importances_)
    ]

    return {
        "loocv_r2": pooled_r2,
        "n_wells_used": len(usable_ids),
        "per_well": per_well_results,
        "feature_importance": feature_importance,
    }


def get_property_models() -> dict:
    """Top-level entry point: eligibility -> feature extraction -> LOOCV
    for every (property, method) combination. status='insufficient_data'
    is a first-class, explicit outcome (never a fabricated score) when
    fewer than MIN_ELIGIBLE_WELLS wells have a usable tie."""
    from app.services import dashboard_upload_service as dus
    from app.services import seismic_processor as sp_mod

    eligible_ids, excluded = _eligible_wells()

    if len(eligible_ids) < MIN_ELIGIBLE_WELLS:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(eligible_ids)} well(s) currently have a usable tie (need at least "
                f"{MIN_ELIGIBLE_WELLS} for leave-one-well-out validation). Improve tie coverage "
                "-- more wells, better coordinate calibration, or a real checkshot -- before a "
                "spatial prediction here would be trustworthy."
            ),
            "eligible_well_ids": eligible_ids,
            "excluded_wells": excluded,
            "n_wells_used": len(eligible_ids),
            "results": None,
        }

    volume = sp_mod.get_segy_volume()

    wells_features: dict[str, dict] = {}
    for well_id in eligible_ids:
        synth = dus.get_synthetic_summary(well_id)
        shift_ms = synth.get("best_shift_ms") or 0.0
        feats = _extract_well_features(volume, well_id, shift_ms)
        if feats is None:
            excluded.append({
                "well_id": well_id,
                "reason": "Passed the tie-confidence check but its trace/curves could not be re-resolved for feature extraction.",
            })
            continue
        wells_features[well_id] = feats

    used_ids = list(wells_features.keys())
    if len(used_ids) < MIN_ELIGIBLE_WELLS:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(used_ids)} well(s) had extractable spectral features after "
                f"eligibility filtering (need at least {MIN_ELIGIBLE_WELLS})."
            ),
            "eligible_well_ids": used_ids,
            "excluded_wells": excluded,
            "n_wells_used": len(used_ids),
            "results": None,
        }

    results: dict[str, dict[str, dict | None]] = {}
    for property_name in PETRO_CURVES:
        results[property_name] = {}
        for method in METHODS:
            results[property_name][method] = _train_and_loocv(wells_features, method, property_name)

    return {
        "status": "validated",
        "message": None,
        "eligible_well_ids": used_ids,
        "excluded_wells": excluded,
        "n_wells_used": len(used_ids),
        "results": results,
    }
