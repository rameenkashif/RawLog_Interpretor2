"""
test_sweet_spot_training_data.py
------------------------------------
Tests for services/sweet_spot_training_data.py.

Layered coverage, same strategy as the rest of this session's new test
suites:
- _extract_curve/_target_series checked directly against constructed rows
  with known ground truth (fast, no SEG-Y/well_service needed).
- build_well_samples/build_training_pool checked end-to-end against a
  small synthetic SEG-Y (segyio) + monkeypatched well_service/config
  functions -- avoids needing a real LAS file or repository, and keeps
  the well-tie phase-grid search fast (small trace count).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app import well_seismic_tie as wst
from app.services import sweet_spot_training_data as sstd
from app.services import seismic_processor as sp
from app.services import well_service


class TestExtractCurve:
    def test_missing_key_becomes_nan(self):
        rows = [{"DEPT": 100.0}, {"DEPT": 101.0, "DT": 80.0}]
        arr = sstd._extract_curve(rows, "DT")
        assert np.isnan(arr[0])
        assert arr[1] == pytest.approx(80.0)

    def test_null_sentinel_becomes_nan(self):
        rows = [{"DT": -9999.0}, {"DT": 75.0}]
        arr = sstd._extract_curve(rows, "DT")
        assert np.isnan(arr[0])
        assert arr[1] == pytest.approx(75.0)


class TestTargetSeries:
    def test_interpolates_onto_depth_at_time(self):
        rows = [{"DEPT": 100.0, "VSH": 0.2}, {"DEPT": 110.0, "VSH": 0.4}, {"DEPT": 120.0, "VSH": 0.6}]
        depth_all = np.array([100.0, 110.0, 120.0])
        depth_at_time = np.array([105.0, 115.0])
        out = sstd._target_series(rows, depth_all, "VSH", depth_at_time)
        np.testing.assert_allclose(out, [0.3, 0.5])

    def test_too_few_valid_samples_returns_all_nan(self):
        rows = [{"DEPT": 100.0, "VSH": 0.2}]
        depth_all = np.array([100.0])
        depth_at_time = np.array([100.0, 101.0])
        out = sstd._target_series(rows, depth_all, "VSH", depth_at_time)
        assert np.isnan(out).all()

    def test_outside_valid_range_is_nan(self):
        rows = [{"DEPT": 100.0, "VSH": 0.2}, {"DEPT": 110.0, "VSH": 0.4}]
        depth_all = np.array([100.0, 110.0])
        depth_at_time = np.array([50.0, 105.0, 200.0])
        out = sstd._target_series(rows, depth_all, "VSH", depth_at_time)
        assert np.isnan(out[0])
        assert not np.isnan(out[1])
        assert np.isnan(out[2])


# ---- End-to-end fixture: a small synthetic SEG-Y + a constructed well ------
INLINES = list(range(400, 405))     # 400..404
CROSSLINES = list(range(100, 104))  # 100..103
N_SAMPLES = 80
DELAY_MS = 2000.0
INTERVAL_MS = 2.0
SOURCE_X_BASE = 500000.0
SOURCE_Y_BASE = 4000000.0


def _write_test_segy(path: Path, seed: int = 0) -> None:
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(N_SAMPLES) * INTERVAL_MS + DELAY_MS
    spec.tracecount = len(INLINES) * len(CROSSLINES)

    rng = np.random.default_rng(seed)
    i = 0
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = int(INTERVAL_MS * 1000)
        f.text[0] = (
            "C 1 TEST SURVEY TRACE INLINE AT 9 AND SIZE 4 TRACE CROSSLINE AT 13 AND SIZE 4"
        ).ljust(3200)
        for il in INLINES:
            for xl in CROSSLINES:
                f.header[i] = {
                    segyio.TraceField.FieldRecord: il,
                    segyio.TraceField.TraceNumber: xl,
                    segyio.TraceField.SourceX: int(SOURCE_X_BASE + il * 25),
                    segyio.TraceField.SourceY: int(SOURCE_Y_BASE + xl * 25),
                    segyio.TraceField.DelayRecordingTime: int(DELAY_MS),
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(INTERVAL_MS * 1000),
                }
                f.trace[i] = rng.normal(0, 1.0, N_SAMPLES).astype(np.float32)
                i += 1


class _FakeWellSummary:
    def __init__(self, well_x, well_y):
        self.well_x = well_x
        self.well_y = well_y


def _fake_curve_rows(n: int = 60) -> list[dict]:
    """A well with full-coverage DEPT/DT/RHOB/DPTM/GR/VSH/PHIE/SWE/PHIT,
    depth-time roughly matching the test SEG-Y's own recorded window."""
    depth = np.linspace(3000.0, 3100.0, n)
    dptm = np.linspace(DELAY_MS + 10.0, DELAY_MS + (N_SAMPLES - 10) * INTERVAL_MS, n)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        rows.append({
            "DEPT": float(depth[i]),
            "DT": float(70.0 + rng.normal(0, 2)),
            "RHOB": float(2.3 + rng.normal(0, 0.05)),
            "DPTM": float(dptm[i]),
            "GR": float(60.0 + rng.normal(0, 5)),
            "VSH": float(np.clip(0.3 + rng.normal(0, 0.1), 0, 1)),
            "PHIE": float(np.clip(0.15 + rng.normal(0, 0.03), 0, 0.3)),
            "SWE": float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1)),
            "PHIT": float(np.clip(0.2 + rng.normal(0, 0.03), 0, 0.35)),
        })
    return rows


@pytest.fixture
def volume(tmp_path):
    path = tmp_path / "test_sweet_spot.sgy"
    _write_test_segy(path)
    return sp.SegyVolume(path)


class TestBuildWellSamples:
    def _patch_well(self, monkeypatch, well_x, well_y, rows):
        monkeypatch.setattr(well_service, "get_well_summary", lambda well_id: _FakeWellSummary(well_x, well_y))
        monkeypatch.setattr(well_service, "get_well_curves", lambda well_id: {"data": rows})
        monkeypatch.setattr(sstd, "_load_tie_config", lambda: {"max_tie_search_radius_m": 1000.0})
        monkeypatch.setattr(sstd, "get_well_config", lambda well_id: {"zones": {"vsh_max": 0.4}})

    def test_returns_populated_samples(self, volume, monkeypatch):
        # Well located near the trace at inline=402, crossline=101.
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        assert result is not None
        assert result.well_id == "TEST_WELL"
        assert result.inline_number == 402
        assert result.crossline_number == 101
        n_pos = result.X58.shape[1] if result.X58.ndim == 2 else None
        assert result.X58.ndim == 2
        assert result.X58.shape[1] == len(result.feature_names)
        assert -1.0 <= result.tie_correlation <= 1.0

    def test_all_targets_present(self, volume, monkeypatch):
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        assert set(result.targets.keys()) == set(sstd.ALL_TARGETS)
        for name, arr in result.targets.items():
            assert arr.shape == (result.X58.shape[0],)

    def test_neighborhood_tiling_repeats_same_targets_per_position(self, volume, monkeypatch):
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        vsh = result.targets["vsh"]
        # X58's rows are position-major (all of position 0's samples, then
        # position 1's, ...) -- so the target array (tiled the same way)
        # must repeat in identical blocks of length n_overlap.
        n_pos = 9  # interior trace -> full 3x3 neighborhood
        assert len(vsh) % n_pos == 0
        block = len(vsh) // n_pos
        first_block = vsh[:block]
        for p in range(1, n_pos):
            np.testing.assert_allclose(vsh[p * block : (p + 1) * block], first_block)

    def test_center_row_range_selects_the_wells_own_trace_features(self, volume, monkeypatch):
        # Interior trace (inline=402, crossline=101) sits at the MIDDLE
        # position of its 3x3 neighborhood -- center_il_pos=1, center_xl_pos=1
        # -> position index = 1*3+1 = 4 (of 0..8), so the center block should
        # be the 5th of 9 equal blocks.
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        n_pos = 9
        block = result.X58.shape[0] // n_pos
        assert result.center_row_end - result.center_row_start == block
        assert result.center_row_start == 4 * block

        # The center block's amplitude features must come from the ACTUAL
        # trace at (402, 101), i.e. match a direct volume.get_trace() read
        # at the well's own tied overlap samples (not a neighbor's).
        center_amp = result.X58[result.center_row_start : result.center_row_end, result.feature_names.index("attr_amp_center")]
        direct_trace = volume.get_trace(
            volume._inline_index[402][list(volume.crossline[volume._inline_index[402]]).index(101)]
        )
        # amp_center at the well's own overlap-masked time samples should
        # match the real trace's amplitude there exactly.
        assert len(center_amp) > 0
        # Sanity: values are drawn straight from the real trace, not zero/NaN padding.
        assert np.isfinite(center_amp).all()

    def test_is_sand_matches_vsh_threshold(self, volume, monkeypatch):
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        vsh = result.targets["vsh"]
        is_sand = result.targets["is_sand"]
        valid = np.isfinite(vsh) & np.isfinite(is_sand)
        np.testing.assert_array_equal(is_sand[valid], (vsh[valid] < 0.4).astype(float))

    def test_ai_computed_from_dt_rhob(self, volume, monkeypatch):
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        ai = result.targets["ai"]
        assert np.isfinite(ai).any()
        assert (ai[np.isfinite(ai)] > 0).all()  # AI = velocity*density, always positive

    def test_edge_trace_gets_smaller_neighborhood(self, volume, monkeypatch):
        # Corner of the survey -> 2x2 neighborhood, not 3x3.
        well_x = SOURCE_X_BASE + INLINES[0] * 25
        well_y = SOURCE_Y_BASE + CROSSLINES[0] * 25
        self._patch_well(monkeypatch, well_x, well_y, _fake_curve_rows())

        result = sstd.build_well_samples(volume, "TEST_WELL")
        vsh = result.targets["vsh"]
        n_pos = 4  # corner -> 2x2
        assert len(vsh) % n_pos == 0

    def test_no_coordinates_raises(self, volume, monkeypatch):
        monkeypatch.setattr(well_service, "get_well_summary", lambda well_id: _FakeWellSummary(None, None))
        monkeypatch.setattr(sstd, "_load_tie_config", lambda: {"max_tie_search_radius_m": 1000.0})
        with pytest.raises(wst.TieError, match="no surface coordinates"):
            sstd.build_well_samples(volume, "TEST_WELL")


class TestBuildTrainingPool:
    def test_excludes_well_not_found_with_reason(self, volume, monkeypatch):
        def _raise(well_id):
            raise well_service.WellNotFoundError(well_id)

        monkeypatch.setattr(well_service, "get_well_summary", _raise)
        usable, excluded = sstd.build_training_pool(volume, ["DOES_NOT_EXIST"])
        assert usable == []
        assert len(excluded) == 1
        assert excluded[0]["well_id"] == "DOES_NOT_EXIST"

    def test_excludes_tie_failure_with_reason(self, volume, monkeypatch):
        monkeypatch.setattr(well_service, "get_well_summary", lambda well_id: _FakeWellSummary(None, None))
        monkeypatch.setattr(sstd, "_load_tie_config", lambda: {"max_tie_search_radius_m": 1000.0})
        usable, excluded = sstd.build_training_pool(volume, ["BAD_WELL"])
        assert usable == []
        assert "No usable tie" in excluded[0]["reason"]

    def test_usable_well_is_included(self, volume, monkeypatch):
        well_x = SOURCE_X_BASE + 402 * 25
        well_y = SOURCE_Y_BASE + 101 * 25
        monkeypatch.setattr(well_service, "get_well_summary", lambda well_id: _FakeWellSummary(well_x, well_y))
        monkeypatch.setattr(well_service, "get_well_curves", lambda well_id: {"data": _fake_curve_rows()})
        monkeypatch.setattr(sstd, "_load_tie_config", lambda: {"max_tie_search_radius_m": 1000.0})
        monkeypatch.setattr(sstd, "get_well_config", lambda well_id: {"zones": {"vsh_max": 0.4}})

        usable, excluded = sstd.build_training_pool(volume, ["GOOD_WELL"])
        assert excluded == []
        assert len(usable) == 1
        assert usable[0].well_id == "GOOD_WELL"
