"""
test_coordinate_calibration.py
----------------------------------
Tests for app/coordinate_calibration.py: the per-axis linear well<->seismic
coordinate fit, its validation against real trace positions (bin-spacing
tolerance), and the extrapolation flag for wells outside the fit's known
good zone (see fixes #4/#5 in the calibration audit).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.coordinate_calibration import (
    CoordinateCalibrationError,
    apply_calibration,
    estimate_bin_spacing_m,
    fit_per_axis_calibration,
    validate_and_flag_wells,
)


class TestFitPerAxisCalibration:
    def test_fits_exact_linear_mapping(self):
        cal = fit_per_axis_calibration(
            well_x=np.array([0.0, 100.0, 200.0]),
            well_y=np.array([0.0, 50.0, 100.0]),
            seismic_x_range=(1000.0, 1200.0),
            seismic_y_range=(500.0, 600.0),
        )
        assert cal.a == pytest.approx(1.0)
        assert cal.b == pytest.approx(1000.0)
        assert cal.c == pytest.approx(1.0)
        assert cal.d == pytest.approx(500.0)
        assert cal.n_wells_used == 3

    def test_apply_calibration_maps_midpoint_correctly(self):
        cal = fit_per_axis_calibration(
            well_x=np.array([0.0, 200.0]),
            well_y=np.array([0.0, 100.0]),
            seismic_x_range=(1000.0, 1200.0),
            seismic_y_range=(500.0, 600.0),
        )
        tx, ty = apply_calibration(cal, 100.0, 50.0)
        assert tx == pytest.approx(1100.0)
        assert ty == pytest.approx(550.0)

    def test_different_scale_per_axis(self):
        # X ratio and Y ratio deliberately different (mirrors the real
        # field's finding: ~1.22x on X vs ~0.44x on Y, ruling out a
        # single uniform conversion factor).
        cal = fit_per_axis_calibration(
            well_x=np.array([0.0, 100.0]),
            well_y=np.array([0.0, 100.0]),
            seismic_x_range=(0.0, 122.0),
            seismic_y_range=(0.0, 44.0),
        )
        assert cal.a == pytest.approx(1.22)
        assert cal.c == pytest.approx(0.44)

    def test_fewer_than_two_wells_raises(self):
        with pytest.raises(CoordinateCalibrationError):
            fit_per_axis_calibration(
                well_x=np.array([0.0]), well_y=np.array([0.0]),
                seismic_x_range=(0.0, 1.0), seismic_y_range=(0.0, 1.0),
            )

    def test_ignores_nan_wells(self):
        cal = fit_per_axis_calibration(
            well_x=np.array([0.0, np.nan, 200.0]),
            well_y=np.array([0.0, np.nan, 100.0]),
            seismic_x_range=(1000.0, 1200.0),
            seismic_y_range=(500.0, 600.0),
        )
        assert cal.n_wells_used == 2

    def test_degenerate_range_raises(self):
        with pytest.raises(CoordinateCalibrationError):
            fit_per_axis_calibration(
                well_x=np.array([50.0, 50.0, 50.0]),
                well_y=np.array([0.0, 50.0, 100.0]),
                seismic_x_range=(1000.0, 1200.0),
                seismic_y_range=(500.0, 600.0),
            )


class TestEstimateBinSpacing:
    def test_regular_grid_spacing(self):
        xs, ys = np.meshgrid(np.arange(0, 100, 10.0), np.arange(0, 100, 10.0))
        spacing = estimate_bin_spacing_m(xs.ravel(), ys.ravel())
        assert spacing == pytest.approx(10.0, abs=0.5)

    def test_single_trace_returns_default(self):
        assert estimate_bin_spacing_m(np.array([0.0]), np.array([0.0])) == 1.0


class TestValidateAndFlagWells:
    @pytest.fixture
    def calibration_setup(self):
        cal = fit_per_axis_calibration(
            well_x=np.array([0.0, 200.0]),
            well_y=np.array([0.0, 100.0]),
            seismic_x_range=(1000.0, 1200.0),
            seismic_y_range=(500.0, 600.0),
        )
        gx, gy = np.meshgrid(np.arange(1000.0, 1201.0, 10.0), np.arange(500.0, 601.0, 10.0))
        trace_x, trace_y = gx.ravel(), gy.ravel()
        bin_spacing = estimate_bin_spacing_m(trace_x, trace_y)
        return cal, trace_x, trace_y, bin_spacing

    def test_well_inside_range_is_trustworthy(self, calibration_setup):
        cal, trace_x, trace_y, bin_spacing = calibration_setup
        results = validate_and_flag_wells(
            cal, ["W1"], np.array([100.0]), np.array([50.0]), trace_x, trace_y, bin_spacing,
        )
        r = results[0]
        assert r.transformed_x == pytest.approx(1100.0)
        assert r.transformed_y == pytest.approx(550.0)
        assert r.within_bin_tolerance is True
        assert r.is_extrapolated is False
        assert r.trustworthy is True

    def test_well_far_outside_range_is_flagged_extrapolated(self, calibration_setup):
        # Mirrors the real 8th-well failure: a well whose raw coordinate is
        # way outside the calibration wells' own range extrapolates the fit
        # far outside the survey footprint.
        cal, trace_x, trace_y, bin_spacing = calibration_setup
        results = validate_and_flag_wells(
            cal, ["W8"], np.array([5000.0]), np.array([50.0]), trace_x, trace_y, bin_spacing,
        )
        r = results[0]
        assert r.is_extrapolated is True
        assert r.within_bin_tolerance is False
        assert r.trustworthy is False
        assert r.nearest_trace_distance_m > bin_spacing * 2

    def test_multiple_wells_independent_flags(self, calibration_setup):
        cal, trace_x, trace_y, bin_spacing = calibration_setup
        results = validate_and_flag_wells(
            cal,
            ["GOOD", "BAD"],
            np.array([100.0, -9000.0]),
            np.array([50.0, 50.0]),
            trace_x,
            trace_y,
            bin_spacing,
        )
        by_id = {r.well_id: r for r in results}
        assert by_id["GOOD"].trustworthy is True
        assert by_id["BAD"].trustworthy is False
