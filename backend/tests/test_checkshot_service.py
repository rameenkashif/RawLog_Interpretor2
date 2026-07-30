"""
test_checkshot_service.py
------------------------------
Unit tests for app/services/checkshot_service.py: parsing an uploaded
checkshot workbook (openpyxl, one sheet per well) and storing/reading it
back through checkshot_repository.

The column layout tested here (index, label, TWT(ms), <blank>, Depth(m),
data starting row 2) was verified against the real uploaded
Zamzama-TDS.xlsx, not just assumed from the reference pipeline's comment.
"""

from __future__ import annotations

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from app.checkshot_repository import FileCheckshotRepository
from app.services import checkshot_service


def _workbook_bytes(sheets: dict[str, list[tuple]]) -> bytes:
    """sheets: {sheet_name: [(idx, label, twt_ms, blank, depth_m), ...]}"""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        ws.append((None, None, "TWT(ms)", None, "Depth"))  # header row (row 1)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestParseCheckshotWorkbook:
    def test_parses_one_sheet_per_well(self):
        data = _workbook_bytes({
            "Z-02": [(1, "adjust1", 2045, None, 3354.8), (2, "adjust1", 2101, None, 3409.01)],
            "Z-03": [(1, "adjust2", 2200, None, 3651.5)],
        })
        result = checkshot_service.parse_checkshot_workbook(data)
        assert set(result.keys()) == {"Z-02", "Z-03"}
        assert result["Z-02"] == [(3354.8, 2045.0), (3409.01, 2101.0)]
        assert result["Z-03"] == [(3651.5, 2200.0)]

    def test_sorts_points_by_depth_ascending(self):
        data = _workbook_bytes({
            "Z-02": [(1, "l", 2200, None, 3700.0), (2, "l", 2045, None, 3350.0)],
        })
        result = checkshot_service.parse_checkshot_workbook(data)
        depths = [d for d, _t in result["Z-02"]]
        assert depths == sorted(depths)

    def test_skips_rows_with_missing_twt_or_depth(self):
        data = _workbook_bytes({
            "Z-02": [(1, "l", None, None, 3350.0), (2, "l", 2100, None, None), (3, "l", 2200, None, 3700.0)],
        })
        result = checkshot_service.parse_checkshot_workbook(data)
        assert result["Z-02"] == [(3700.0, 2200.0)]

    def test_sheet_with_no_valid_rows_is_omitted(self):
        data = _workbook_bytes({
            "Z-02": [(1, "l", 2045, None, 3350.0)],
            "Z-08": [(1, "l", None, None, None)],
        })
        result = checkshot_service.parse_checkshot_workbook(data)
        assert set(result.keys()) == {"Z-02"}

    def test_no_valid_points_anywhere_raises(self):
        data = _workbook_bytes({"Z-02": [(1, "l", None, None, None)]})
        with pytest.raises(checkshot_service.CheckshotValidationError):
            checkshot_service.parse_checkshot_workbook(data)

    def test_corrupt_bytes_raise_validation_error(self):
        with pytest.raises(checkshot_service.CheckshotValidationError):
            checkshot_service.parse_checkshot_workbook(b"not a real xlsx file")


class TestStoreAndReadCheckshots:
    @pytest.fixture(autouse=True)
    def _isolated_repo(self, tmp_path, monkeypatch):
        repo = FileCheckshotRepository(base_dir=tmp_path)
        monkeypatch.setattr(checkshot_service, "get_checkshot_repository", lambda: repo)

    def test_store_then_get_round_trip(self):
        data = _workbook_bytes({
            "Z-02": [(1, "l", 2045, None, 3354.8), (2, "l", 2101, None, 3409.01)],
        })
        counts = checkshot_service.store_checkshot_workbook(data)
        assert counts == {"Z-02": 2}

        points = checkshot_service.get_checkshot_points("Z-02")
        assert points == [(3354.8, 2045.0), (3409.01, 2101.0)]

    def test_get_points_for_unknown_well_returns_empty_list(self):
        assert checkshot_service.get_checkshot_points("Z-99") == []

    def test_status_reports_point_counts_per_well(self):
        data = _workbook_bytes({
            "Z-02": [(1, "l", 2045, None, 3354.8)],
            "Z-03": [(1, "l", 2200, None, 3651.5), (2, "l", 2242, None, 3700.77)],
        })
        checkshot_service.store_checkshot_workbook(data)
        assert checkshot_service.get_checkshot_status() == {"Z-02": 1, "Z-03": 2}

    def test_reupload_overwrites_a_wells_points(self):
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"Z-02": [(1, "l", 2045, None, 3354.8)]})
        )
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"Z-02": [(1, "l", 2045, None, 3354.8), (2, "l", 2101, None, 3409.01)]})
        )
        assert len(checkshot_service.get_checkshot_points("Z-02")) == 2

    def test_well_id_with_extra_suffix_matches_by_prefix(self):
        # This app derives well_id from the LAS FILENAME (e.g.
        # 'Z-02_RAW.las' -> 'Z-02_RAW'), which commonly doesn't match a
        # checkshot workbook's sheet name (usually the field's base well
        # name, 'Z-02', matching the LAS header's WELL mnemonic instead).
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"Z-02": [(1, "l", 2045, None, 3354.8), (2, "l", 2101, None, 3409.01)]})
        )
        assert checkshot_service.get_checkshot_points("Z-02_RAW") == [(3354.8, 2045.0), (3409.01, 2101.0)]

    def test_prefix_match_is_case_insensitive(self):
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"z-02": [(1, "l", 2045, None, 3354.8)]})
        )
        assert checkshot_service.get_checkshot_points("Z-02_RAW") == [(3354.8, 2045.0)]

    def test_prefix_match_requires_a_real_boundary(self):
        # 'Z-02' must NOT match a well_id of 'Z-020...' -- '2' immediately
        # after the prefix is alphanumeric, not a real name boundary.
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"Z-02": [(1, "l", 2045, None, 3354.8)]})
        )
        assert checkshot_service.get_checkshot_points("Z-020_RAW") == []

    def test_exact_match_preferred_over_prefix_match(self):
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({
                "Z-02": [(1, "l", 1000, None, 1000.0)],
                "Z-02_RAW": [(1, "l", 2045, None, 3354.8)],
            })
        )
        assert checkshot_service.get_checkshot_points("Z-02_RAW") == [(3354.8, 2045.0)]

    def test_status_still_keyed_by_original_sheet_name(self):
        # get_checkshot_status is a diagnostic listing of what was
        # actually uploaded -- it must NOT be affected by the lookup
        # fallback used for tie resolution.
        checkshot_service.store_checkshot_workbook(
            _workbook_bytes({"Z-02": [(1, "l", 2045, None, 3354.8)]})
        )
        assert checkshot_service.get_checkshot_status() == {"Z-02": 1}
