"""
services/sweet_spot_training_data.py
----------------------------------------
Training-data construction for the Sweet-Spot prediction module (V11-style
architecture doc, Section 4): for each well, resolve a phase-rotated
full-window tie (well_seismic_tie.search_best_tie_full_window_phase_grid +
refine_tie_warp), pull the nearest 3x3 seismic trace neighborhood
(seismic_processor.get_trace_neighbors_3x3), compute the 58-attribute
feature pool for every trace in that neighborhood
(sweet_spot_feature_engine.compute_feature_pool), and align it to the
well's own logged targets via the SAME depth<->time overlap convention
direct_tie_service/spectral_petro_correlation_service already use.

Every one of the 3x3 neighborhood's traces is assigned the well's OWN
target values at the SAME tied time samples (the seismic time axis is
trace-independent, so this needs no per-neighbor re-tie) -- same
neighborhood-expansion principle blind_well_prediction_service.py already
uses for its radius-based neighborhood, just against this module's fixed
3x3 grid instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.config_loader import get_well_config
from app.services import well_service
from app.services.tie_service import _load_config as _load_tie_config
from app.services.sweet_spot_feature_engine import compute_feature_pool

# Minimum valid DEPT/DPTM samples to trust a depth<->time mapping at all --
# same threshold direct_tie_service.MIN_DEPTH_TIME_SAMPLES uses.
MIN_DEPTH_TIME_SAMPLES = 10

_TARGET_LAS_NAMES: dict[str, str] = {
    "dt": "DT", "rhob": "RHOB", "phit": "PHIT", "gr": "GR", "vsh": "VSH", "phie": "PHIE", "swe": "SWE",
}
# Data-reality scope cut (see plan): none of these wells' LAS files carry a
# shear-sonic (DTS) curve, so MURHO/POIS/VPVS/LMRHO (the doc's
# shear-dependent elastic targets) aren't attempted -- the doc's own
# results already conclude those need pre-stack AVO data this survey
# doesn't have either way.
STAGE1_TARGETS: tuple[str, ...] = ("ai", "dt", "phit")
STAGE2_TARGETS: tuple[str, ...] = ("gr", "rhob", "vsh", "phie", "swe")
ALL_TARGETS: tuple[str, ...] = STAGE1_TARGETS + STAGE2_TARGETS + ("is_sand",)


def _extract_curve(rows: list[dict], name: str) -> np.ndarray:
    arr = np.array([row.get(name) if row.get(name) is not None else np.nan for row in rows], dtype=float)
    arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
    return arr


def _target_series(rows: list[dict], depth_all: np.ndarray, las_name: str, depth_at_time: np.ndarray) -> np.ndarray:
    """Depth -> target-value interpolation against THIS target's OWN null
    mask -- same convention as spectral_petro_correlation_service's
    _property_series (each property may have different valid depths than
    DPTM or any other curve)."""
    values = _extract_curve(rows, las_name)
    valid = np.isfinite(depth_all) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full_like(depth_at_time, np.nan)
    d, v = depth_all[valid], values[valid]
    return np.interp(depth_at_time, d, v, left=np.nan, right=np.nan)


@dataclass
class _SweetSpotWellSamples:
    well_id: str
    X58: np.ndarray                    # (n_samples, 58) full attribute pool
    feature_names: tuple[str, ...]     # column names of X58, in order
    targets: dict[str, np.ndarray]     # target_name -> (n_samples,), may contain NaN
    time_ms: np.ndarray                # seismic TWT (ms) per sample
    depth_m: np.ndarray                # logged depth (m) per sample
    tie_weight: float                  # this well's tie correlation (uniform per-well weight)
    tie_correlation: float
    inline_number: int
    crossline_number: int
    distance_m: float                  # well-to-nearest-trace distance
    center_row_start: int              # X58/targets/time_ms row where the well's OWN center-trace
                                        # block starts (see build_well_samples' tiling order)
    center_row_end: int                # exclusive end of that block


class SweetSpotTrainingDataError(Exception):
    """Base class for sweet-spot-training-data-module errors."""


def build_well_samples(volume, well_id: str) -> _SweetSpotWellSamples | None:
    """Resolves this well's phase-rotated tie, pulls its 3x3 trace
    neighborhood, computes the 58-attribute pool for every neighbor trace,
    and aligns it to the well's own logged targets. Returns None if the
    well's own trace couldn't be spectrally decomposed (e.g. too short);
    raises wst.TieError for a genuine tie failure -- callers should treat
    that as "exclude this well, with a reason," never silently proceed.
    """
    tie_config = _load_tie_config()
    max_radius_m = tie_config.get("max_tie_search_radius_m")

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
    best = wst.search_best_tie_full_window_phase_grid(t_rc, rc, volume.twt_axis_ms, volume.sample_interval_ms, real_trace)
    refined = wst.refine_tie_warp(t_rc, rc, volume.twt_axis_ms, real_trace, best)

    # Depth<->DPTM mapping (independent of the DT/RHOB validity
    # reflectivity_from_time_axis required internally) -- same convention
    # direct_tie_service.resolve_direct_tie uses.
    valid = np.isfinite(depth) & np.isfinite(dptm)
    depth_v, dptm_v = depth[valid], dptm[valid]
    order = np.argsort(dptm_v)
    depth_v, dptm_v = depth_v[order], dptm_v[order]
    keep = np.concatenate([[True], np.diff(dptm_v) > 1e-6])
    depth_v, dptm_v = depth_v[keep], dptm_v[keep]
    if len(depth_v) < MIN_DEPTH_TIME_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}' has too few valid DEPT/DPTM samples ({len(depth_v)}) to build a "
            f"depth<->time mapping (need >= {MIN_DEPTH_TIME_SAMPLES})."
        )

    # Discrete seismic-time <-> depth overlap, via the tie's BULK shift
    # only (the warp refinement's finer +/-8ms local correction is folded
    # into the reported correlation but not into this discrete mapping --
    # a deliberate simplification: at 2ms seismic sampling, the warp's
    # bound is a handful of samples, and re-deriving an exact per-sample
    # inverse-warp depth mapping isn't worth the complexity it would add
    # here). Same overlap/depth_at_time construction as
    # spectral_petro_correlation_service._resolve_well_tie_context.
    total_shift_ms = best.bulk_shift_ms
    seismic_twt = volume.twt_axis_ms
    well_axis_time = seismic_twt - total_shift_ms
    overlap = (well_axis_time >= dptm_v[0]) & (well_axis_time <= dptm_v[-1])
    if not overlap.any():
        raise wst.TieError(
            f"Well '{well_id}'s logged interval does not overlap the seismic survey's recorded "
            "time window -- no samples to correlate."
        )
    depth_at_time = np.interp(seismic_twt[overlap] - total_shift_ms, dptm_v, depth_v)
    time_at_overlap = seismic_twt[overlap]

    neighbors = volume.get_trace_neighbors_3x3(inline_number, crossline_number)
    traces_3d = neighbors["traces"]
    pool58 = compute_feature_pool(traces_3d, volume.sample_interval_ms)  # each (n_time_full, n_pos)

    n_il, n_xl, _ = traces_3d.shape
    n_pos = n_il * n_xl
    feature_names = tuple(sorted(pool58.keys()))
    X_overlap_full = np.stack([pool58[name][overlap, :] for name in feature_names], axis=-1)  # (n_overlap, n_pos, n_feat)
    if X_overlap_full.shape[0] == 0:
        return None

    # Tile: every one of the n_pos neighborhood traces gets the SAME
    # well's own target values at the SAME tied time/depth samples --
    # the seismic time axis is trace-independent, so this needs no
    # per-neighbor re-tie (same principle
    # blind_well_prediction_service._extract_well_samples already uses).
    n_overlap = X_overlap_full.shape[0]
    X58 = X_overlap_full.transpose(1, 0, 2).reshape(n_pos * n_overlap, len(feature_names))
    depth_tiled = np.tile(depth_at_time, n_pos)
    time_tiled = np.tile(time_at_overlap, n_pos)

    # The well's OWN center trace's block within the n_pos-tiled arrays
    # above -- position order matches compute_feature_pool's C-order
    # (il_idx*n_xl + xl_idx) flatten, which get_trace_neighbors_3x3's
    # center_il_pos/center_xl_pos already index into directly.
    center_position_index = neighbors["center_il_pos"] * n_xl + neighbors["center_xl_pos"]
    center_row_start = center_position_index * n_overlap
    center_row_end = center_row_start + n_overlap

    config = get_well_config(well_id)
    vsh_max = config.get("zones", {}).get("vsh_max", 0.4)

    targets: dict[str, np.ndarray] = {}
    for target_name, las_name in _TARGET_LAS_NAMES.items():
        targets[target_name] = np.tile(_target_series(rows, depth, las_name, depth_at_time), n_pos)

    dt_at_depth = _target_series(rows, depth, "DT", depth_at_time)
    rhob_at_depth = _target_series(rows, depth, "RHOB", depth_at_time)
    with np.errstate(divide="ignore", invalid="ignore"):
        ai_at_depth = wst.acoustic_impedance(dt_at_depth, rhob_at_depth, dt_unit="us_per_ft")
    targets["ai"] = np.tile(ai_at_depth, n_pos)

    vsh_at_depth = _target_series(rows, depth, "VSH", depth_at_time)
    is_sand_at_depth = np.where(np.isfinite(vsh_at_depth), (vsh_at_depth < vsh_max).astype(float), np.nan)
    targets["is_sand"] = np.tile(is_sand_at_depth, n_pos)

    return _SweetSpotWellSamples(
        well_id=well_id,
        X58=X58,
        feature_names=feature_names,
        targets=targets,
        time_ms=time_tiled,
        depth_m=depth_tiled,
        tie_weight=refined.correlation,
        tie_correlation=refined.correlation,
        inline_number=inline_number,
        crossline_number=crossline_number,
        distance_m=distance_m,
        center_row_start=center_row_start,
        center_row_end=center_row_end,
    )


def build_training_pool(volume, well_ids: list[str]) -> tuple[list[_SweetSpotWellSamples], list[dict]]:
    """Returns (usable_wells, excluded_wells_with_reasons) -- a well is
    never silently dropped."""
    usable: list[_SweetSpotWellSamples] = []
    excluded: list[dict] = []
    for well_id in well_ids:
        try:
            samples = build_well_samples(volume, well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            excluded.append({"well_id": well_id, "reason": f"No usable tie: {exc}"})
            continue
        if samples is None:
            excluded.append({"well_id": well_id, "reason": "Could not extract spectral features for its neighborhood."})
            continue
        usable.append(samples)
    return usable, excluded
