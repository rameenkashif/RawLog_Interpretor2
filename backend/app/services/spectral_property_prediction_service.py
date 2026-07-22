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

Well eligibility and time alignment here are driven by _resolve_direct_tie
below -- the SAME two-part algorithm tie_service.get_well_seismic_tie uses
for the Well-to-Seismic Tie page (a direct nearest-trace spatial search via
well_seismic_tie.find_nearest_trace_index, then a DPTM-based, full-seismic-
window frequency/polarity/bulk-shift correlation search via
well_seismic_tie.search_best_tie_full_window), applied against this
feature's single active SEG-Y volume instead of a separately uploaded
dataset. This module used to resolve well->trace via
coordinate_calibration_service (the same calibrated-fit transform
synthetic_seismogram_service uses) -- that was replaced because the fit
stretches these wells' own coordinate extent to the FULL seismic survey's
extent, which badly over-spreads crossline position for a well cluster
that only covers a small part of the survey. Cross-checking against the
Well-to-Seismic Tie page's own per-well numbers confirmed the calibrated
fit was landing tens of bins away from the position that page's direct
search finds and validates with real waveform correlation (0.6-0.94 across
Z-02..Z-08) -- this module now uses that same proven-good resolution
instead. This means the wells this module considers "usable" can still
legitimately differ from get_field_overview's coordinate-calibration-based
tie -- that remains an architectural fact, not a bug.

Boundary-pinned/low-confidence wells are excluded from training, per
well_seismic_tie.cross_correlate_and_shift's own documented guidance:
"Boundary-pinned results should be excluded from aggregate statistics
(mean correlation, ML training sets) by default."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services.dashboard_upload_service import TIE_LOW_CONFIDENCE_THRESHOLD
from app.services.spectral_petro_correlation_service import (
    PETRO_CURVES,
    _WellTieContext,
    _extract_curve,
    _property_series,
)
from app.services.tie_service import _load_config as _load_tie_config
from app.services import well_service

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

# Minimum valid DEPT/DPTM samples to trust a depth<->time mapping for
# placing spectral features by depth -- matches spectral_petro_correlation_
# service.MIN_DEPTH_TIME_SAMPLES / synthetic_seismogram_service's build_
# synthetic guard.
MIN_DEPTH_TIME_SAMPLES = 10


@dataclass
class _DirectTieResult:
    """A well's direct nearest-trace tie against the single active SEG-Y
    volume, plus everything _extract_well_features needs to pull spectral
    features at the resolved trace without re-resolving anything."""

    ctx: _WellTieContext
    correlation: float
    bulk_shift_ms: float
    best_freq_hz: float
    boundary_pinned: bool
    low_confidence: bool


def _resolve_direct_tie(volume, well_id: str) -> _DirectTieResult:
    """Direct nearest-trace spatial search (well_seismic_tie.
    find_nearest_trace_index on raw well_x/well_y vs the volume's own
    source_x/source_y) plus a DPTM-based, full-seismic-window frequency/
    polarity/bulk-shift correlation search (well_seismic_tie.
    search_best_tie_full_window) -- the exact algorithm tie_service.
    get_well_seismic_tie uses for the Well-to-Seismic Tie page, reused here
    against this feature's single active volume instead of a named
    dataset. Raises TieError/WellNotFoundError/SegyVolumeError on any
    failure -- callers treat that as "excluded", never silently proceed.
    """
    config = _load_tie_config()
    max_radius_m = config.get("max_tie_search_radius_m")
    max_shift_ms = float(config.get("tie_search_max_shift_ms", wst.DEFAULT_TIE_SEARCH_MAX_SHIFT_MS))

    well_summary = well_service.get_well_summary(well_id)  # raises WellNotFoundError if absent
    if well_summary.well_x is None or well_summary.well_y is None:
        raise wst.TieError(
            f"Well '{well_id}' has no surface coordinates in its LAS header -- cannot locate it "
            "on the seismic survey."
        )
    trace_idx, distance_m = wst.find_nearest_trace_index(
        well_summary.well_x, well_summary.well_y, volume.source_x, volume.source_y, max_radius_m=max_radius_m
    )
    inline_number = int(volume.inline[trace_idx])
    crossline_number = int(volume.crossline[trace_idx])

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth = _extract_curve(rows, "DEPT")
    dt_log = _extract_curve(rows, "DT")
    rhob = _extract_curve(rows, "RHOB")
    dptm = _extract_curve(rows, "DPTM")

    t_rc, rc = wst.reflectivity_from_time_axis(dptm, dt_log, rhob, volume.sample_interval_ms)
    real_trace = volume.get_trace(trace_idx)
    tie = wst.search_best_tie_full_window(
        t_rc, rc, volume.twt_axis_ms, volume.sample_interval_ms, real_trace, max_shift_ms=max_shift_ms
    )
    boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - wst.BOUNDARY_PINNED_FRACTION) * max_shift_ms
    low_confidence = tie.correlation < TIE_LOW_CONFIDENCE_THRESHOLD or boundary_pinned

    # DEPT<->DPTM mapping for placing seismic-time samples back at a depth
    # (and from there, at a VSH/PHIE/SWE value) -- independent of the
    # DT/RHOB validity reflectivity_from_time_axis required internally,
    # per this codebase's convention of each curve getting its own null
    # mask (see _property_series).
    valid = np.isfinite(depth) & np.isfinite(dptm)
    depth_v, dptm_v = depth[valid], dptm[valid]
    order = np.argsort(dptm_v)
    depth_v, dptm_v = depth_v[order], dptm_v[order]
    keep = np.concatenate([[True], np.diff(dptm_v) > 1e-6])
    depth_v, dptm_v = depth_v[keep], dptm_v[keep]
    if len(depth_v) < MIN_DEPTH_TIME_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}' has too few valid DEPT/DPTM samples ({len(depth_v)}) to align "
            f"spectral features (need >= {MIN_DEPTH_TIME_SAMPLES})."
        )

    seismic_twt = volume.twt_axis_ms
    overlap = (seismic_twt - tie.bulk_shift_ms >= dptm_v[0]) & (seismic_twt - tie.bulk_shift_ms <= dptm_v[-1])
    if not overlap.any():
        raise wst.TieError(
            f"Well '{well_id}'s logged interval does not overlap the seismic survey's recorded "
            "time window after the direct tie's bulk shift -- no samples to correlate."
        )
    depth_at_time = np.interp(seismic_twt[overlap] - tie.bulk_shift_ms, dptm_v, depth_v)

    ctx = _WellTieContext(
        well_id=well_id,
        inline_number=inline_number,
        crossline_number=crossline_number,
        distance_m=distance_m,
        tie_method="direct_nearest_trace",
        rows=rows,
        depth=depth,
        depth_at_time=depth_at_time,
        overlap=overlap,
    )
    return _DirectTieResult(
        ctx=ctx,
        correlation=tie.correlation,
        bulk_shift_ms=tie.bulk_shift_ms,
        best_freq_hz=tie.best_freq_hz,
        boundary_pinned=boundary_pinned,
        low_confidence=low_confidence,
    )


def _eligible_wells(volume) -> tuple[list[str], list[dict], dict[str, _DirectTieResult]]:
    """Classifies every currently loaded well as eligible (has a usable
    direct nearest-trace tie -- see module docstring) or excluded (with a
    human-readable reason), never silently dropping a well without saying
    why. Returns the resolved tie for each eligible well too, so
    get_property_models doesn't need to re-resolve it."""
    eligible: list[str] = []
    excluded: list[dict] = []
    tie_results: dict[str, _DirectTieResult] = {}

    for summary in well_service.list_well_summaries():
        well_id = summary.well_id
        try:
            result = _resolve_direct_tie(volume, well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            excluded.append({"well_id": well_id, "reason": f"No usable tie: {exc}"})
            continue
        if result.boundary_pinned:
            excluded.append({
                "well_id": well_id,
                "reason": "Shift search pinned to its boundary -- likely a spurious match, not a genuine tie.",
            })
            continue
        if result.low_confidence:
            excluded.append({
                "well_id": well_id,
                "reason": f"Low-confidence tie (correlation={result.correlation:.3f}, below the 0.3 threshold).",
            })
            continue
        eligible.append(well_id)
        tie_results[well_id] = result

    return eligible, excluded, tie_results


def _extract_well_features(volume, ctx: _WellTieContext) -> dict | None:
    """One well's already-resolved-and-shift-corrected feature matrices
    (n_samples, n_freq) for both spectral methods, plus its VSH/PHIE/SWE
    target series aligned to the same samples. Returns None (caller treats
    as an exclusion, not a crash) if the trace/curves can't actually be
    pulled despite passing the eligibility check."""
    try:
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
    from app.services import seismic_processor as sp_mod

    volume = sp_mod.get_segy_volume()
    eligible_ids, excluded, tie_results = _eligible_wells(volume)

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

    wells_features: dict[str, dict] = {}
    for well_id in eligible_ids:
        feats = _extract_well_features(volume, tie_results[well_id].ctx)
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
