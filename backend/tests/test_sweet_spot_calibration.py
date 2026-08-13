"""
test_sweet_spot_calibration.py
----------------------------------
Tests for services/sweet_spot_calibration.py -- the per-well normalization
and "dynamic calibration" (robust median std + degree-2 compaction trend)
that fixes the source doc's own documented pooled-statistics bug (Section
10: one outlier well's corrupted std inflating every prediction's scale).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import sweet_spot_calibration as cal
from app.services.sweet_spot_training_data import _SweetSpotWellSamples


def _make_well(well_id, n, feature_mean, feature_std, target_mean, target_std, time_start, seed=0):
    rng = np.random.default_rng(seed)
    X58 = rng.normal(feature_mean, feature_std, size=(n, 3))
    feature_names = ("f0", "f1", "f2")
    time_ms = np.linspace(time_start, time_start + 100.0, n)
    targets = {"vsh": rng.normal(target_mean, target_std, n)}
    return _SweetSpotWellSamples(
        well_id=well_id, X58=X58, feature_names=feature_names, targets=targets,
        time_ms=time_ms, depth_m=np.zeros(n), tie_weight=0.8, tie_correlation=0.8,
        inline_number=1, crossline_number=1, distance_m=10.0,
        center_row_start=0, center_row_end=n,
    )


class TestPerWellZscoreFeatures:
    def test_scaled_features_have_zero_mean_unit_std_per_well(self):
        wells = [
            _make_well("A", 60, feature_mean=10.0, feature_std=2.0, target_mean=0.3, target_std=0.05, time_start=2000, seed=1),
            _make_well("B", 60, feature_mean=-5.0, feature_std=0.5, target_mean=0.3, target_std=0.05, time_start=2100, seed=2),
        ]
        scales = cal.per_well_zscore_features(wells, ["f0", "f1", "f2"])
        for well_id in ("A", "B"):
            X = scales[well_id].X_scaled
            np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-8)
            np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-8)

    def test_each_well_scaled_independently_different_raw_distributions(self):
        # Well B's raw features are wildly different in scale from A's --
        # if scaling were pooled instead of per-well, B's z-scores would
        # NOT come out ~N(0,1) on their own.
        wells = [
            _make_well("A", 60, feature_mean=0.0, feature_std=1.0, target_mean=0.3, target_std=0.05, time_start=2000, seed=1),
            _make_well("B", 60, feature_mean=1000.0, feature_std=200.0, target_mean=0.3, target_std=0.05, time_start=2100, seed=2),
        ]
        scales = cal.per_well_zscore_features(wells, ["f0", "f1", "f2"])
        assert scales["A"].mean[0] == pytest.approx(0.0, abs=0.5)
        assert scales["B"].mean[0] == pytest.approx(1000.0, abs=100.0)

    def test_zero_variance_feature_does_not_divide_by_zero(self):
        well = _make_well("A", 20, feature_mean=0.0, feature_std=1.0, target_mean=0.3, target_std=0.05, time_start=2000)
        well.X58[:, 0] = 5.0  # constant feature -> zero std
        scales = cal.per_well_zscore_features([well], ["f0", "f1", "f2"])
        assert np.isfinite(scales["A"].X_scaled).all()
        np.testing.assert_allclose(scales["A"].X_scaled[:, 0], 0.0)


class TestPerWellWinsorizeAndZscoreTarget:
    def test_outlier_samples_are_clipped(self):
        well = _make_well("A", 50, feature_mean=0, feature_std=1, target_mean=0.3, target_std=0.02, time_start=2000, seed=3)
        well.targets["vsh"][0] = 100.0  # extreme outlier
        scales = cal.per_well_winsorize_and_zscore_target([well], "vsh")
        assert scales["A"].y_winsorized[0] < 10.0  # clipped, not 100

    def test_scaled_output_uses_winsorized_std(self):
        well = _make_well("A", 50, feature_mean=0, feature_std=1, target_mean=0.3, target_std=0.02, time_start=2000, seed=3)
        scales = cal.per_well_winsorize_and_zscore_target([well], "vsh")
        s = scales["A"]
        expected = (s.y_winsorized - s.median) / s.winsorized_std
        np.testing.assert_allclose(s.y_scaled, expected, equal_nan=True)

    def test_too_few_valid_samples_returns_all_nan(self):
        well = _make_well("A", 5, feature_mean=0, feature_std=1, target_mean=0.3, target_std=0.02, time_start=2000)
        well.targets["vsh"] = np.array([0.3, np.nan, np.nan, np.nan, np.nan])
        scales = cal.per_well_winsorize_and_zscore_target([well], "vsh")
        assert np.isnan(scales["A"].y_scaled).all()
        assert np.isnan(scales["A"].median)


class TestFitDynamicCalibration:
    def test_robust_std_median_immune_to_one_outlier_well(self):
        # Doc's own documented bug: Z-07's corrupted LMRHO std was ~28.99
        # GPa vs ~0.45 GPa typical -- a pooled std would be inflated ~14x.
        # The MEDIAN of per-well stds should stay near the typical value.
        normal_std = 0.45
        wells = [
            _make_well(f"NORMAL_{i}", 60, feature_mean=0, feature_std=1, target_mean=5.0, target_std=normal_std, time_start=2000 + i * 50, seed=i)
            for i in range(4)
        ]
        outlier = _make_well("OUTLIER", 60, feature_mean=0, feature_std=1, target_mean=5.0, target_std=28.99, time_start=2200, seed=99)
        all_wells = wells + [outlier]

        target_scales = cal.per_well_winsorize_and_zscore_target(all_wells, "vsh")
        calib = cal.fit_dynamic_calibration(all_wells, "vsh", target_scales)
        assert calib is not None
        # Median of 5 stds (4 near 0.45, 1 near ~29 post-winsorization,
        # itself reduced by winsorization) should land close to ~0.45,
        # nowhere near a pooled/mean estimate that the outlier would drag
        # toward ~6+.
        assert calib.robust_std_median < 1.0

    def test_returns_none_with_no_valid_data(self):
        well = _make_well("A", 5, feature_mean=0, feature_std=1, target_mean=0.3, target_std=0.02, time_start=2000)
        well.targets["vsh"] = np.full(5, np.nan)
        target_scales = cal.per_well_winsorize_and_zscore_target([well], "vsh")
        calib = cal.fit_dynamic_calibration([well], "vsh", target_scales)
        assert calib is None

    def test_trend_fit_recovers_known_linear_relationship(self):
        # A well-behaved case: target increases linearly with time across
        # training wells (a compaction-like trend) -- polyfit degree 2
        # should recover it closely even though it's really degree 1.
        rng = np.random.default_rng(7)
        wells = []
        for i, t0 in enumerate((2000.0, 2100.0, 2200.0)):
            n = 80
            time_ms = np.linspace(t0, t0 + 90.0, n)
            target = 2.0 + 0.01 * time_ms + rng.normal(0, 0.02, n)
            well = _SweetSpotWellSamples(
                well_id=f"W{i}", X58=rng.normal(size=(n, 3)), feature_names=("f0", "f1", "f2"),
                targets={"ai": target}, time_ms=time_ms, depth_m=np.zeros(n), tie_weight=0.8,
                tie_correlation=0.8, inline_number=1, crossline_number=1, distance_m=10.0,
                center_row_start=0, center_row_end=n,
            )
            wells.append(well)

        target_scales = cal.per_well_winsorize_and_zscore_target(wells, "ai")
        calib = cal.fit_dynamic_calibration(wells, "ai", target_scales)
        predicted_at_2150 = np.polyval(calib.trend_coeffs, 2150.0)
        assert predicted_at_2150 == pytest.approx(2.0 + 0.01 * 2150.0, abs=0.1)


class TestApplyCalibration:
    def test_zero_z_pred_returns_trend_only(self):
        calib = cal.DynamicCalibration(robust_std_median=2.0, trend_coeffs=np.array([0.0, 0.0, 5.0]))
        out = cal.apply_calibration(np.array([0.0, 0.0]), np.array([2000.0, 2100.0]), calib)
        np.testing.assert_allclose(out, [5.0, 5.0])

    def test_scales_by_robust_std(self):
        calib = cal.DynamicCalibration(robust_std_median=3.0, trend_coeffs=np.array([0.0, 0.0, 0.0]))
        out = cal.apply_calibration(np.array([1.0, -1.0, 2.0]), np.array([2000.0, 2000.0, 2000.0]), calib)
        np.testing.assert_allclose(out, [3.0, -3.0, 6.0])
