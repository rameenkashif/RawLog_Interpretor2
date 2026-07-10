"""
test_well_zone_tie_service.py
---------------------------------
Tests for app/services/well_zone_tie_service.py (the "Well-Seismic Tie"
map) and its router endpoint (GET /api/seismic/well-zone-tie-map).

Reuses test_seismic_processor.py's synthetic-SEG-Y helper and coordinate
patching conventions -- same as test_synthetic_seismogram_service.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

segyio = pytest.importorskip("segyio")

from app.repository import FileWellRepository
from app.services import seismic_processor as sp
from app.services import well_service
from app.services import well_zone_tie_service as wzt
from tests.test_seismic_processor import _set_las_coordinate_meters, _write_synthetic_segy

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Test survey's SourceX/Y extent (see _write_synthetic_segy defaults):
# X: 363000 + il*10 for il in 382..386 -> 366820-366860
# Y: 2949800 + xl*10 for xl in 46..49  -> 2950260-2950290
WELL_A_XY = (366825.0, 2950262.0)  # near the low-inline/low-crossline corner
WELL_B_XY = (366840.0, 2950275.0)  # near the center
WELL_C_XY = (366858.0, 2950288.0)  # near the high-inline/high-crossline corner


@pytest.fixture
def well_repo(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells")


@pytest.fixture
def volume(tmp_path) -> sp.SegyVolume:
    path = tmp_path / "test_survey.sgy"
    _write_synthetic_segy(path)
    return sp.SegyVolume(path)


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch, well_repo, volume):
    monkeypatch.setattr(
        well_service, "list_well_summaries",
        lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
    )
    monkeypatch.setattr(
        well_service, "get_well_df",
        lambda well_id, repo=None, _f=well_service.get_well_df: _f(well_id, repo=well_repo),
    )
    monkeypatch.setattr(sp, "get_segy_volume", lambda refresh=False: volume)


def _load_aligned_well(well_repo, las_filename: str, xy: tuple[float, float]):
    las_text = (RAW_LAS_DIR / las_filename).read_text()
    las_text = _set_las_coordinate_meters(las_text, "X", xy[0])
    las_text = _set_las_coordinate_meters(las_text, "Y", xy[1])
    return well_service.process_and_store_las_bytes(las_text.encode(), las_filename, repo=well_repo)


class TestComputeWellZoneTieMap:
    def test_too_few_wells_raises(self, well_repo):
        _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        with pytest.raises(wzt.WellZoneTieError):
            wzt.compute_well_zone_tie_map()

    def test_three_aligned_wells_produce_full_grid(self, well_repo):
        wa = _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        wb = _load_aligned_well(well_repo, "Z-03_raw.las", WELL_B_XY)
        wc = _load_aligned_well(well_repo, "Z-04_raw.las", WELL_C_XY)

        result = wzt.compute_well_zone_tie_map()

        assert len(result["wells"]) == 3
        assert {w["well_id"] for w in result["wells"]} == {wa.well_id, wb.well_id, wc.well_id}
        assert len(result["predicted_vsh"]) == len(result["inline_axis"])
        assert all(len(row) == len(result["crossline_axis"]) for row in result["predicted_vsh"])
        # No grid gaps in this fixture's full rectangle, so every cell should
        # have an interpolated (non-null) value.
        assert all(v is not None for row in result["predicted_vsh"] for v in row)
        assert "not a seismic inversion" in result["method_note"].lower()
        assert result["warnings"] == []

    def test_idw_favors_nearest_well_at_its_own_cell(self, well_repo):
        wa = _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        wb = _load_aligned_well(well_repo, "Z-03_raw.las", WELL_B_XY)
        wc = _load_aligned_well(well_repo, "Z-04_raw.las", WELL_C_XY)

        result = wzt.compute_well_zone_tie_map(power=2.0)
        by_id = {w["well_id"]: w for w in result["wells"]}
        vsh_values = {w["well_id"]: w["mean_vsh_pay"] for w in result["wells"]}

        for well_id in (wa.well_id, wb.well_id, wc.well_id):
            entry = by_id[well_id]
            il_idx = result["inline_axis"].index(entry["inline"])
            xl_idx = result["crossline_axis"].index(entry["crossline"])
            predicted_here = result["predicted_vsh"][il_idx][xl_idx]
            own_value = vsh_values[well_id]
            other_values = [v for wid, v in vsh_values.items() if wid != well_id]
            # At (or very near) a well's own tied trace, IDW should weight
            # that well's value far more heavily than the others -- closer
            # to its own value than to either other well's value.
            assert abs(predicted_here - own_value) < min(abs(predicted_here - v) for v in other_values)

    def test_well_missing_coordinates_is_skipped_with_warning(self, well_repo):
        wa = _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        wb = _load_aligned_well(well_repo, "Z-03_raw.las", WELL_B_XY)
        metadata, df = well_repo.get_well(wb.well_id)
        metadata.well_x = None
        metadata.well_y = None
        well_repo.save_well(metadata, df)

        # Need at least 2 usable wells overall for the assertions below to
        # be meaningful; add a third aligned well so removing wb's
        # coordinates still leaves 2 usable wells.
        _load_aligned_well(well_repo, "Z-04_raw.las", WELL_C_XY)

        result = wzt.compute_well_zone_tie_map()
        assert len(result["wells"]) == 2
        assert any("no surface coordinates" in w for w in result["warnings"])


class TestRouter:
    def test_endpoint_returns_map(self, well_repo):
        from fastapi.testclient import TestClient

        import main

        _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)
        _load_aligned_well(well_repo, "Z-03_raw.las", WELL_B_XY)
        _load_aligned_well(well_repo, "Z-04_raw.las", WELL_C_XY)

        client = TestClient(main.app)
        response = client.get("/api/seismic/well-zone-tie-map")
        assert response.status_code == 200
        body = response.json()
        assert len(body["wells"]) == 3
        assert "method_note" in body

    def test_endpoint_422_when_too_few_wells(self, well_repo):
        from fastapi.testclient import TestClient

        import main

        _load_aligned_well(well_repo, "Z-02_raw.las", WELL_A_XY)

        client = TestClient(main.app)
        response = client.get("/api/seismic/well-zone-tie-map")
        assert response.status_code == 422
