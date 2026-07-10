"""
test_las_loader.py
--------------------
Unit tests for app/las_loader.py, focused on the surface coordinate
(~Well section) parsing added alongside the well-to-seismic tie feature --
existing required-curve/null-handling behaviour is already exercised
end-to-end via the real Z-02..Z-08 LAS files, so these focus on the new
XWELL/YWELL (and alias) coordinate extraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from app.las_loader import load_las_file

MINIMAL_CURVES = """~Curve Information -----------------------------------------
DEPT.m  :
GR      .   :
RESISTIVITY.   :
RHOB    .   :
NPHI    .   :
DT      .   :
~ASCII -----------------------------------------------------
 100.0   50.0   10.0   2.4   0.2   90.0
 100.5   51.0   10.5   2.4   0.2   90.5
 101.0   52.0   11.0   2.4   0.2   91.0
"""


def _write_las(well_section: str) -> Path:
    text = (
        "~Version ---------------------------------------------------\n"
        "VERS.   2.0 :\n"
        "WRAP.    NO :\n"
        "~Well ------------------------------------------------------\n"
        "STRT.m 100.0 : START DEPTH\n"
        "STOP.m 101.0 : STOP DEPTH\n"
        "STEP.m   0.5 : STEP\n"
        "NULL.    -9999.25 : NULL VALUE\n"
        "WELL.        TEST-1 :\n" + well_section + MINIMAL_CURVES
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".las", delete=False, mode="w")
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


class TestWellCoordinates:
    def test_xwell_ywell_parsed(self):
        path = _write_las("XWELL.m 512340.00 :\nYWELL.m 6543210.00 :\n")
        loaded = load_las_file(path)
        assert loaded.metadata.well_x == pytest.approx(512340.00)
        assert loaded.metadata.well_y == pytest.approx(6543210.00)

    def test_alias_xcoord_ycoord_parsed(self):
        path = _write_las("XCOORD.m 100.0 :\nYCOORD.m 200.0 :\n")
        loaded = load_las_file(path)
        assert loaded.metadata.well_x == pytest.approx(100.0)
        assert loaded.metadata.well_y == pytest.approx(200.0)

    def test_missing_coordinates_are_none(self):
        path = _write_las("")
        loaded = load_las_file(path)
        assert loaded.metadata.well_x is None
        assert loaded.metadata.well_y is None

    def test_blank_coordinate_value_is_none(self):
        path = _write_las("XWELL.m  :\nYWELL.m  :\n")
        loaded = load_las_file(path)
        assert loaded.metadata.well_x is None
        assert loaded.metadata.well_y is None


class TestUnitStandardization:
    """STOP is fixed at 101.0 m by _write_las's minimal well section."""

    def test_feet_detected_and_converted(self):
        # TD=330 ft / STOP=101 m -> ratio 3.267, squarely in the feet range.
        path = _write_las(
            "X   .m 1000000.0 :\nY   .m 2000000.0 :\nKB  .m 150.0 :\nTD  .m 330.0 :\n"
        )
        loaded = load_las_file(path)
        m = loaded.metadata
        assert m.coordinate_unit_detected == "feet"
        assert m.unit_conversion_applied is True
        assert m.well_x == pytest.approx(1000000.0 * 0.3048)
        assert m.well_y == pytest.approx(2000000.0 * 0.3048)
        assert m.kb_m == pytest.approx(150.0 * 0.3048)
        assert m.td_m == pytest.approx(330.0 * 0.3048)
        assert m.td_stop_ratio == pytest.approx(330.0 / 101.0)

    def test_already_meters_not_converted(self):
        # TD=105 m / STOP=101 m -> ratio ~1.04, not feet-like.
        path = _write_las("X   .m 500.0 :\nY   .m 600.0 :\nKB  .m 40.0 :\nTD  .m 105.0 :\n")
        loaded = load_las_file(path)
        m = loaded.metadata
        assert m.coordinate_unit_detected == "meters"
        assert m.unit_conversion_applied is False
        assert m.well_x == pytest.approx(500.0)
        assert m.well_y == pytest.approx(600.0)
        assert m.kb_m == pytest.approx(40.0)
        assert m.td_m == pytest.approx(105.0)

    def test_missing_td_leaves_unvalidated(self):
        path = _write_las("X   .m 500.0 :\nY   .m 600.0 :\n")
        loaded = load_las_file(path)
        m = loaded.metadata
        assert m.coordinate_unit_detected is None
        assert m.unit_conversion_applied is False
        assert m.td_stop_ratio is None
        # No TD/KB to validate against -- X/Y pass through unconverted.
        assert m.well_x == pytest.approx(500.0)

    def test_real_z02_well_detected_as_feet(self):
        """End-to-end check against the actual shipped LAS file."""
        real_path = (
            Path(__file__).resolve().parents[1] / "data" / "raw" / "Z-02_raw.las"
        )
        loaded = load_las_file(real_path)
        m = loaded.metadata
        assert m.coordinate_unit_detected == "feet"
        assert m.unit_conversion_applied is True
        # Converted X/Y must land inside the real SEG-Y survey's known extent
        # (X: 363124-370654, Y: 2949830-2957150) -- this is the whole point
        # of the conversion, so assert it rather than just the raw math.
        assert 363124.0 <= m.well_x <= 370654.0
        assert 2949830.0 <= m.well_y <= 2957150.0


def _write_las_with_dt(dt_values: list[float], dt_unit_declared: str = "") -> Path:
    """Build a minimal LAS file with custom DT values/declared unit, to
    exercise curve unit inference (see _resolve_and_normalize_curve_units)
    independent of the fixed MINIMAL_CURVES sample data above."""
    rows = "\n".join(
        f" {100.0 + i * 0.5:.1f}   50.0   10.0   2.4   0.2   {dt:.2f}" for i, dt in enumerate(dt_values)
    )
    text = (
        "~Version ---------------------------------------------------\n"
        "VERS.   2.0 :\n"
        "WRAP.    NO :\n"
        "~Well ------------------------------------------------------\n"
        f"STRT.m {100.0:.1f} : START DEPTH\n"
        f"STOP.m {100.0 + (len(dt_values) - 1) * 0.5:.1f} : STOP DEPTH\n"
        "STEP.m   0.5 : STEP\n"
        "NULL.    -9999.25 : NULL VALUE\n"
        "WELL.        TEST-1 :\n"
        "~Curve Information -----------------------------------------\n"
        "DEPT.m  :\n"
        "GR      .   :\n"
        "RESISTIVITY.   :\n"
        "RHOB    .   :\n"
        "NPHI    .   :\n"
        f"DT      .{dt_unit_declared}   :\n"
        "~ASCII -----------------------------------------------------\n"
        f"{rows}\n"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".las", delete=False, mode="w")
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


class TestCurveUnitInference:
    def test_dt_in_us_per_ft_range_not_converted(self):
        # Real-field-like values (~48-97), unambiguously us/ft, blank unit.
        path = _write_las_with_dt([48.0, 65.0, 97.0])
        loaded = load_las_file(path)
        dt_info = next(c for c in loaded.metadata.curve_units if c.curve == "DT")
        assert dt_info.resolved_unit == "us_per_ft"
        assert dt_info.inferred is True
        assert dt_info.conversion_applied is False
        assert loaded.df["DT"].tolist() == pytest.approx([48.0, 65.0, 97.0])

    def test_dt_in_us_per_m_range_converted_to_us_per_ft(self):
        # Values >160 fall outside the us/ft range entirely -- unambiguous us/m.
        path = _write_las_with_dt([300.0, 350.0, 400.0])
        loaded = load_las_file(path)
        dt_info = next(c for c in loaded.metadata.curve_units if c.curve == "DT")
        assert dt_info.resolved_unit == "us_per_m"
        assert dt_info.conversion_applied is True
        from app.las_loader import FT_TO_M

        expected = [v * FT_TO_M for v in (300.0, 350.0, 400.0)]
        assert loaded.df["DT"].tolist() == pytest.approx(expected)

    def test_declared_unit_string_is_trusted_over_inference(self):
        # Value (140) sits in the DT us/ft-us/m overlap zone -- an explicit
        # declared unit should be trusted rather than the inference's
        # us/ft-first tiebreak.
        path = _write_las_with_dt([140.0, 141.0, 142.0], dt_unit_declared="US/M")
        loaded = load_las_file(path)
        dt_info = next(c for c in loaded.metadata.curve_units if c.curve == "DT")
        assert dt_info.resolved_unit == "us_per_m"
        assert dt_info.inferred is False
        assert dt_info.conversion_applied is True

    def test_declared_us_per_ft_not_converted(self):
        path = _write_las_with_dt([90.0, 91.0, 92.0], dt_unit_declared="US/F")
        loaded = load_las_file(path)
        dt_info = next(c for c in loaded.metadata.curve_units if c.curve == "DT")
        assert dt_info.resolved_unit == "us_per_ft"
        assert dt_info.inferred is False
        assert dt_info.conversion_applied is False
        assert loaded.df["DT"].tolist() == pytest.approx([90.0, 91.0, 92.0])

    def test_real_z02_dt_inferred_as_us_per_ft_no_conversion(self):
        """End-to-end check: the real field's actual DT values (~48-97)
        must resolve to us_per_ft and NOT get double-converted."""
        real_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Z-02_raw.las"
        loaded = load_las_file(real_path)
        dt_info = next(c for c in loaded.metadata.curve_units if c.curve == "DT")
        assert dt_info.resolved_unit == "us_per_ft"
        assert dt_info.conversion_applied is False
