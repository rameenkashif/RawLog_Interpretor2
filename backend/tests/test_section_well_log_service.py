"""
test_section_well_log_service.py
---------------------------------
Tests for services/section_well_log_service.py. _curve_at_depth is tested
as a pure function (no real LAS/SEG-Y needed) -- the property-vs-DPTM
alignment logic that matters is exactly the same pattern already covered
end-to-end for spectral features by test_spectral_petro_correlation_
service.py's TestTimeShiftCorrection and test_spectral_property_
prediction_service.py; direct_tie_service.resolve_direct_tie itself
(shared by both) isn't re-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import section_well_log_service as swl


def _rows(depth: list[float], vsh: list[float | None]) -> list[dict]:
    return [{"DEPT": d, "VSH": v} for d, v in zip(depth, vsh)]


class TestCurveAtDepth:
    def test_interpolates_onto_requested_depths(self):
        rows = _rows([100.0, 110.0, 120.0, 130.0], [0.1, 0.3, 0.5, 0.7])
        depth_all = np.array([100.0, 110.0, 120.0, 130.0])
        depth_m = np.array([105.0, 115.0, 125.0])

        result = swl._curve_at_depth(rows, "VSH", depth_all, depth_m)

        assert result == pytest.approx([0.2, 0.4, 0.6])

    def test_returns_none_for_out_of_range_depths(self):
        rows = _rows([100.0, 110.0], [0.1, 0.3])
        depth_all = np.array([100.0, 110.0])
        depth_m = np.array([90.0, 105.0, 120.0])

        result = swl._curve_at_depth(rows, "VSH", depth_all, depth_m)

        assert result[0] is None  # before the logged interval
        assert result[1] == pytest.approx(0.2)
        assert result[2] is None  # after the logged interval

    def test_curve_own_null_mask_independent_of_other_curves(self):
        # VSH has a hole at 110 that DEPT/DPTM validity wouldn't catch --
        # _curve_at_depth must skip it via its own isfinite check, not
        # assume every row's VSH is present just because DEPT is.
        rows = [
            {"DEPT": 100.0, "VSH": 0.1},
            {"DEPT": 110.0, "VSH": None},
            {"DEPT": 120.0, "VSH": 0.5},
        ]
        depth_all = np.array([100.0, 110.0, 120.0])
        depth_m = np.array([110.0])

        result = swl._curve_at_depth(rows, "VSH", depth_all, depth_m)

        # Interpolated straight across the gap from 100->120, not from a
        # (nonexistent) valid sample at 110.
        assert result == pytest.approx([0.3])

    def test_too_few_valid_samples_returns_all_none(self):
        rows = _rows([100.0], [0.1])
        depth_all = np.array([100.0])
        depth_m = np.array([100.0, 105.0])

        result = swl._curve_at_depth(rows, "VSH", depth_all, depth_m)

        assert result == [None, None]


class TestGetSectionWellLogsValidation:
    def test_rejects_invalid_orientation(self):
        with pytest.raises(ValueError, match="orientation"):
            swl.get_section_well_logs("diagonal", 100)
