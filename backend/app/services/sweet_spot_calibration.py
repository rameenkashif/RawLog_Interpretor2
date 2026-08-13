"""
services/sweet_spot_calibration.py
--------------------------------------
Per-well normalization + "dynamic calibration" for the Sweet-Spot
prediction module -- the source doc's own "core V11 innovation" (Section
10), built here to fix the exact two failure modes it documents:

1. Pooled standard deviation scaling: a single outlier well's corrupted
   log can inflate a POOLED std across every training well, scaling every
   prediction too wide (the doc's own Z-07 cycle-skip DT case, 28.99 GPa
   vs ~0.45 GPa typical). Fixed here via the MEDIAN of each individual
   well's own (winsorized) std -- immune to one outlier well, since a
   median doesn't shift much when only one of several values is extreme.

2. Flat population-mean baseline: a single pooled mean ignores that
   different wells sit at different structural depths (e.g. compaction
   raises impedance/density with depth). Fixed here via a degree-2
   polynomial trend of target-vs-seismic-time, fit on the training wells
   and evaluated at whatever time samples are being predicted.

Combined back-conversion: y_hat_physical = z_hat * robust_std_median +
f_trend(seismic_time_ms) -- see apply_calibration.

Every well is z-scored (features AND targets) using its OWN mean/std,
never a pooled one -- this is what "per-well" means throughout this
module, and is what stops cross-well statistical leakage through scaling
(LOGO-CV's own point would otherwise be undermined by the scaler itself
having seen the held-out well's stats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.sweet_spot_training_data import _SweetSpotWellSamples

DEFAULT_WINSORIZE_IQR_SCALE = 3.0


@dataclass
class _WellFeatureScale:
    well_id: str
    X_scaled: np.ndarray  # (n_samples, n_features), z-scored using THIS well's own mean/std
    mean: np.ndarray
    std: np.ndarray


def per_well_zscore_features(
    wells: list[_SweetSpotWellSamples], feature_names: list[str]
) -> dict[str, _WellFeatureScale]:
    """Every well's OWN (mean, std) per curated feature -- never pooled.
    Applies uniformly whether a well is currently playing the training or
    held-out role in a LOGO-CV fold, since z-scoring a well against its
    own distribution needs no label information from any other well."""
    out: dict[str, _WellFeatureScale] = {}
    for well in wells:
        col_idx = [well.feature_names.index(name) for name in feature_names]
        X = well.X58[:, col_idx].astype(float)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where((std > 0) & np.isfinite(std), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        X_scaled = (X - mean) / std
        out[well.well_id] = _WellFeatureScale(well_id=well.well_id, X_scaled=X_scaled, mean=mean, std=std)
    return out


@dataclass
class _WellTargetScale:
    well_id: str
    y_scaled: np.ndarray       # (n_samples,), z-scored winsorized target -- may contain NaN
    y_winsorized: np.ndarray   # (n_samples,), winsorized but still in PHYSICAL units -- may contain NaN
    median: float
    winsorized_std: float


def per_well_winsorize_and_zscore_target(
    wells: list[_SweetSpotWellSamples], target_name: str, iqr_scale: float = DEFAULT_WINSORIZE_IQR_SCALE
) -> dict[str, _WellTargetScale]:
    """Per well: clip the target at ITS OWN median +/- iqr_scale*IQR (guards
    against one bad sample dominating that well's own std), THEN z-score
    using the WINSORIZED value's own std -- so apply_calibration's
    back-conversion inverts against the exact std the model trained
    against, not the pre-winsorized one."""
    out: dict[str, _WellTargetScale] = {}
    for well in wells:
        y = np.asarray(well.targets[target_name], dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            out[well.well_id] = _WellTargetScale(
                well_id=well.well_id, y_scaled=np.full_like(y, np.nan),
                y_winsorized=np.full_like(y, np.nan), median=float("nan"), winsorized_std=float("nan"),
            )
            continue
        yv = y[valid]
        median = float(np.median(yv))
        q1, q3 = np.percentile(yv, [25, 75])
        iqr = q3 - q1
        lo, hi = median - iqr_scale * iqr, median + iqr_scale * iqr
        y_wins = np.clip(y, lo, hi)  # NaN inputs stay NaN under np.clip
        std = float(np.nanstd(y_wins[valid]))
        std = std if std > 0 else 1.0
        y_scaled = (y_wins - median) / std
        out[well.well_id] = _WellTargetScale(
            well_id=well.well_id, y_scaled=y_scaled, y_winsorized=y_wins, median=median, winsorized_std=std,
        )
    return out


@dataclass
class DynamicCalibration:
    robust_std_median: float
    trend_coeffs: np.ndarray  # degree-2 polynomial coefficients (np.polyfit order)


def fit_dynamic_calibration(
    training_wells: list[_SweetSpotWellSamples],
    target_name: str,
    target_scales: dict[str, _WellTargetScale],
) -> DynamicCalibration | None:
    """robust_std_median: median of each TRAINING well's own winsorized std
    (immune to a single outlier well). trend_coeffs: degree-2 polynomial
    of the winsorized target (physical units) vs. seismic time, POOLED
    across training wells -- compaction trend is basin-scale, unlike std
    which must stay per-well-robust to isolate one bad well. Returns None
    if there's not enough valid data across the training wells to fit
    either piece."""
    stds: list[float] = []
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for well in training_wells:
        scale = target_scales.get(well.well_id)
        if scale is None or not np.isfinite(scale.winsorized_std):
            continue
        stds.append(scale.winsorized_std)
        valid = np.isfinite(scale.y_winsorized)
        if valid.any():
            times.append(well.time_ms[valid])
            values.append(scale.y_winsorized[valid])

    if not stds or not times:
        return None

    robust_std_median = float(np.median(stds))
    all_times = np.concatenate(times)
    all_values = np.concatenate(values)
    if len(all_times) < 3:
        return None

    trend_coeffs = np.polyfit(all_times, all_values, deg=2)
    return DynamicCalibration(robust_std_median=robust_std_median, trend_coeffs=trend_coeffs)


def apply_calibration(z_pred: np.ndarray, time_ms: np.ndarray, calib: DynamicCalibration) -> np.ndarray:
    """y_hat_physical = z_hat * robust_std_median + f_trend(seismic_time_ms)."""
    return z_pred * calib.robust_std_median + np.polyval(calib.trend_coeffs, time_ms)
