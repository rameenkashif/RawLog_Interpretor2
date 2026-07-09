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
