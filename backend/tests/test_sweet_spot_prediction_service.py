"""
test_sweet_spot_prediction_service.py
------------------------------------------
Tests for services/sweet_spot_prediction_service.py -- orchestration
(get_or_train_cascade) and region prediction (predict_region).

Uses a small synthetic SEG-Y (segyio) covering several constructed wells'
coordinates + monkeypatched well_service functions (same approach as
test_sweet_spot_training_data.py), with sweet_spot_model_service's
_make_base_templates swapped for a fast 2-template stub (same approach as
test_sweet_spot_model_service.py) -- the real 11-template pool's
doc-faithful hyperparameters take real minutes across even a few wells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

segyio = pytest.importorskip("segyio")

from app.services import seismic_processor as sp
from app.services import sweet_spot_model_service as sms
from app.services import sweet_spot_prediction_service as ssp
from app.services import well_service
from app.services.sweet_spot_training_data import STAGE1_TARGETS, STAGE2_TARGETS
from app.services.well_service import WellNotFoundError

INLINES = list(range(400, 412))     # 400..411 -- wide enough for a real region-prediction test
CROSSLINES = list(range(100, 112))  # 100..111
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
        self.well_id = None


def _fake_curve_rows(n: int = 60, seed: int = 1) -> list[dict]:
    depth = np.linspace(3000.0, 3100.0, n)
    dptm = np.linspace(DELAY_MS + 10.0, DELAY_MS + (N_SAMPLES - 10) * INTERVAL_MS, n)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "DEPT": float(depth[i]), "DT": float(70.0 + rng.normal(0, 2)),
            "RHOB": float(2.3 + rng.normal(0, 0.05)), "DPTM": float(dptm[i]),
            "GR": float(60.0 + rng.normal(0, 5)), "VSH": float(np.clip(0.3 + rng.normal(0, 0.1), 0, 1)),
            "PHIE": float(np.clip(0.15 + rng.normal(0, 0.03), 0, 0.3)),
            "SWE": float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1)),
            "PHIT": float(np.clip(0.2 + rng.normal(0, 0.03), 0, 0.35)),
        })
    return rows


# 7 wells spread across the survey's interior, each with a distinct
# (inline, crossline) so build_training_pool has enough real diversity.
_WELL_POSITIONS = {
    "Z-02_RAW": (403, 103), "Z-03_RAW": (404, 104), "Z-04_RAW": (405, 105),
    "Z-05_RAW": (406, 106), "Z-06_RAW": (407, 107), "Z-07_RAW": (408, 108), "Z-08_RAW": (409, 109),
}


def _patch_wells(monkeypatch):
    summaries = []
    fake_rows_by_well = {}
    for i, (well_id, (il, xl)) in enumerate(_WELL_POSITIONS.items()):
        summary = _FakeWellSummary(SOURCE_X_BASE + il * 25, SOURCE_Y_BASE + xl * 25)
        summary.well_id = well_id
        summaries.append(summary)
        fake_rows_by_well[well_id] = _fake_curve_rows(seed=i)

    def _get_well_summary(well_id):
        for s in summaries:
            if s.well_id == well_id:
                return s
        raise WellNotFoundError(well_id)

    monkeypatch.setattr(well_service, "list_well_summaries", lambda: summaries)
    monkeypatch.setattr(well_service, "get_well_summary", _get_well_summary)
    monkeypatch.setattr(well_service, "get_well_curves", lambda well_id: {"data": fake_rows_by_well[well_id]})


@pytest.fixture
def volume(tmp_path):
    path = tmp_path / "test_sweet_spot_prediction.sgy"
    _write_test_segy(path)
    return sp.SegyVolume(path)


@pytest.fixture(autouse=True)
def _cheap_pool_and_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sms, "_make_base_templates",
        lambda: {"ridge": lambda: Ridge(alpha=1.0), "rf_tiny": lambda: RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)},
    )
    from app.services import sweet_spot_training_data as sstd
    monkeypatch.setattr(sstd, "_load_tie_config", lambda: {"max_tie_search_radius_m": 1000.0})
    monkeypatch.setattr(sstd, "get_well_config", lambda well_id: {"zones": {"vsh_max": 0.4}})
    # Redirect the on-disk cascade cache to a per-test tmp dir -- without
    # this, every test would read/write the real backend/data/models/
    # directory and could load another test's stale cascade (trained
    # against a totally different synthetic volume/well set) instead of
    # actually training.
    monkeypatch.setattr(ssp, "MODELS_DIR", tmp_path / "models")
    ssp._trained_cascade_cache.clear()
    yield
    ssp._trained_cascade_cache.clear()


class TestGetOrTrainCascade:
    def test_validated_end_to_end(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        result = ssp.get_or_train_cascade("Z-02_RAW")
        assert result["status"] == "validated"
        assert result["blind_well_id"] == "Z-02_RAW"
        assert "Z-02_RAW" not in result["training_well_ids"]
        assert set(result["training_well_ids"]) == set(_WELL_POSITIONS) - {"Z-02_RAW"}
        assert set(result["blind_results"].keys()) == set(STAGE1_TARGETS) | set(STAGE2_TARGETS)

    def test_cached_on_second_call(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        first = ssp.get_or_train_cascade("Z-02_RAW")
        second = ssp.get_or_train_cascade("Z-02_RAW")
        assert first is second  # identical cached object, not just equal

    def test_refresh_retrains(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        first = ssp.get_or_train_cascade("Z-02_RAW")
        second = ssp.get_or_train_cascade("Z-02_RAW", refresh=True)
        assert first is not second
        assert second["status"] == "validated"

    def test_unknown_blind_well_raises(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)
        with pytest.raises(WellNotFoundError):
            ssp.get_or_train_cascade("DOES_NOT_EXIST")

    def test_validated_result_persisted_to_disk(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        ssp.get_or_train_cascade("Z-02_RAW")
        assert ssp._cascade_cache_path("Z-02_RAW").exists()

    def test_loads_from_disk_without_retraining_after_memory_cache_cleared(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        first = ssp.get_or_train_cascade("Z-02_RAW")
        ssp._trained_cascade_cache.clear()  # simulate a server restart (memory cache lost, disk file remains)

        calls = []
        real_train = ssp.train_sweet_spot_cascade

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_train(*args, **kwargs)

        monkeypatch.setattr(ssp, "train_sweet_spot_cascade", _spy)

        second = ssp.get_or_train_cascade("Z-02_RAW")
        assert calls == []  # loaded from disk, never retrained
        assert second["status"] == "validated"
        assert second is not first  # a freshly-deserialized object, not the same instance
        assert set(second["training_well_ids"]) == set(first["training_well_ids"])

    def test_refresh_retrains_even_with_disk_cache_present(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        ssp.get_or_train_cascade("Z-02_RAW")
        ssp._trained_cascade_cache.clear()

        calls = []
        real_train = ssp.train_sweet_spot_cascade

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_train(*args, **kwargs)

        monkeypatch.setattr(ssp, "train_sweet_spot_cascade", _spy)

        result = ssp.get_or_train_cascade("Z-02_RAW", refresh=True)
        assert calls == [1]
        assert result["status"] == "validated"

    def test_corrupt_disk_cache_falls_back_to_retraining(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        cache_path = ssp._cascade_cache_path("Z-02_RAW")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"not a valid joblib file")

        result = ssp.get_or_train_cascade("Z-02_RAW")
        assert result["status"] == "validated"


class TestPredictRegion:
    def test_returns_grids_for_requested_properties(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        result = ssp.predict_region((402, 410), (102, 110), ["gr", "ai"], "Z-02_RAW")
        assert set(result["predictions"].keys()) == {"gr", "ai"}
        gr = np.array(result["predictions"]["gr"])
        assert gr.shape == (N_SAMPLES, len(result["inline_axis"]) * len(result["crossline_axis"]))
        # A wide-enough region should have SOME finite predictions (not
        # everything NaN'd out by edge-only gradient features).
        assert np.isfinite(gr).any()

    def test_single_inline_region_has_zero_finite_predictions(self, volume, monkeypatch):
        # Documents a real, non-obvious limitation (caught during manual UI
        # verification, not by an earlier test): the curated feature set
        # includes inline-neighbor gradients (n_il_p1/n_il_m1), which need a
        # real inline+1 AND inline-1 trace WITHIN the fetched region to be
        # non-NaN. A single-inline-wide request (inline_min == inline_max)
        # can never provide either neighbor, so EVERY position's feature
        # vector contains a NaN and gets filtered out entirely -- the
        # frontend must request a padded inline window and slice the
        # desired inline's own columns back out (see
        # SweetSpotSectionView.tsx's INLINE_PAD), never a single inline.
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        result = ssp.predict_region((405, 405), (102, 110), ["gr"], "Z-02_RAW")
        gr = np.array(result["predictions"]["gr"])
        assert not np.isfinite(gr).any()

        # The SAME inline, once padded with real neighbors on both sides,
        # recovers real coverage for that inline's own column block.
        padded = ssp.predict_region((403, 407), (102, 110), ["gr"], "Z-02_RAW")
        gr_padded = np.array(padded["predictions"]["gr"])
        n_xl = len(padded["crossline_axis"])
        il_idx = padded["inline_axis"].index(405)
        middle_slice = gr_padded[:, il_idx * n_xl : (il_idx + 1) * n_xl]
        assert np.isfinite(middle_slice).any()

    def test_unknown_property_raises(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        with pytest.raises(ssp.SweetSpotPredictionError, match="Unknown property"):
            ssp.predict_region((402, 410), (102, 110), ["not_a_real_property"], "Z-02_RAW")

    def test_oversized_region_raises(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)
        original = ssp.MAX_REGION_TRACES
        ssp.MAX_REGION_TRACES = 4  # temporarily shrink the guard for this test
        try:
            with pytest.raises(ssp.SweetSpotPredictionError, match="exceeding"):
                ssp.predict_region((402, 410), (102, 110), ["gr"], "Z-02_RAW")
        finally:
            ssp.MAX_REGION_TRACES = original

    def test_untrained_cascade_raises(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        # Only ONE well available -> insufficient_data, never reaches "validated".
        summaries = [_FakeWellSummary(SOURCE_X_BASE + 403 * 25, SOURCE_Y_BASE + 103 * 25)]
        summaries[0].well_id = "Z-02_RAW"
        monkeypatch.setattr(well_service, "list_well_summaries", lambda: summaries)
        monkeypatch.setattr(well_service, "get_well_summary", lambda well_id: summaries[0])
        monkeypatch.setattr(well_service, "get_well_curves", lambda well_id: {"data": _fake_curve_rows()})

        with pytest.raises(ssp.SweetSpotPredictionError):
            ssp.predict_region((402, 410), (102, 110), ["gr"], "Z-02_RAW")

    def test_stage1_property_returned_directly(self, volume, monkeypatch):
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: volume)
        _patch_wells(monkeypatch)

        result = ssp.predict_region((402, 410), (102, 110), ["ai"], "Z-02_RAW")
        assert "ai" in result["predictions"]
