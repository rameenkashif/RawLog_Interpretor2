"""
test_prediction_pipeline_service.py
--------------------------------------
Tests for services/prediction_pipeline_service.py.

- _predict_blind is tested as a pure function against constructed
  WellPredictionFeatures (no real LAS/SEG-Y needed) -- real well/tie
  resolution (_build_well_raw_cache) needs a real SEG-Y volume and is
  exercised indirectly via test_spectral_property_prediction_service.py's
  coverage of direct_tie_service, which this module's tie resolution is
  identical to.
- run_grid_search/get_best_config are tested against constructed
  _WellRawCache objects, monkeypatching _eligible_well_raw_caches so no
  real volume is needed there either.
- _block_average (the log-target "blocking" primitive) is tested directly
  as a pure function.

Several model paths (raw spectrum + Ridge; instantaneous-attrs + PCA-3 +
RandomForest; instantaneous-attrs + PCA-3 + Ridge) are exercised across
tests, but "recovers a known relationship"/grid-search-picks-signal tests
use a plain (spectrum-only, no PCA, Ridge, tight block) config -- the
simplest path, whose R^2 behavior can be reasoned about confidently
without running the actual model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import prediction_pipeline_service as pps

PLAIN_RIDGE_CONFIG: pps.Config = ("sswt", False, None, "ridge", 1.0)


def _well(well_id: str, n: int, slope: float, seed: int) -> pps.WellPredictionFeatures:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=0.1, size=(n, 4))
    vsh = np.clip(slope * X[:, 0] + 0.5 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    phie = np.clip(slope * X[:, 0] + 0.15 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    # Small-scale, low-variance relative to the spectral features (scale
    # 0.1) so PCA-3 doesn't get dominated by uninformative instantaneous-
    # attribute variance and squeeze out the real signal.
    inst_attrs = rng.normal(scale=0.01, size=(n, 6))
    return pps.WellPredictionFeatures(
        well_id=well_id,
        inline_number=100,
        crossline_number=200,
        correlation=0.9,
        best_freq_hz=25.0,
        polarity=1,
        bulk_shift_ms=0.0,
        depths_m=np.arange(n, dtype=float),
        twt_ms=np.arange(n, dtype=float) * 2.0,
        targets={"vsh": vsh, "phie": phie, "swe": rng.uniform(0, 1, n)},
        features={"cwt": X, "sswt": X.copy()},
        freq_hz={"cwt": np.array([5.0, 10.0, 15.0, 20.0]), "sswt": np.array([5.0, 10.0, 15.0, 20.0])},
        inst_attrs=inst_attrs,
    )


def _well_raw(well_id: str, n: int, slope: float, seed: int) -> pps._WellRawCache:
    """One raw sample per output (seismic) sample, positioned exactly at
    that sample's own time -- _block_average then selects exactly that
    one point regardless of block_half_ms (it's always the sole point
    within any window), so this fixture behaves identically across every
    GRID_BLOCK_HALF_OPTIONS candidate. Good for tests that aren't
    specifically about blocking itself (grid-search plumbing, caching,
    etc.); see TestBlockAverage for fixtures that exercise real
    multi-sample blocking."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=0.1, size=(n, 4))
    vsh = np.clip(slope * X[:, 0] + 0.5 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    phie = np.clip(slope * X[:, 0] + 0.15 + rng.normal(scale=1e-3, size=n), 0.0, 1.0)
    swe = rng.uniform(0, 1, n)
    inst_attrs = rng.normal(scale=0.01, size=(n, 6))
    time_ms = np.arange(n, dtype=float) * 2.0

    return pps._WellRawCache(
        well_id=well_id,
        inline_number=100,
        crossline_number=200,
        correlation=0.9,
        best_freq_hz=25.0,
        polarity=1,
        bulk_shift_ms=0.0,
        tie_source="checkshot (1 valid pt)",
        depths_m=np.arange(n, dtype=float),
        twt_ms=time_ms.copy(),
        features={"cwt": X, "sswt": X.copy()},
        freq_hz={"cwt": np.array([5.0, 10.0, 15.0, 20.0]), "sswt": np.array([5.0, 10.0, 15.0, 20.0])},
        inst_attrs=inst_attrs,
        uniq_idx=np.arange(n),
        time_ms=time_ms,
        t_valid=time_ms.copy(),
        targets_raw={"vsh": vsh, "phie": phie, "swe": swe},
    )


class TestPredictBlind:
    def test_blind_well_never_appears_in_training_set(self):
        wells = {
            "A": _well("A", 20, slope=2.0, seed=1),
            "B": _well("B", 25, slope=2.0, seed=2),
            "C": _well("C", 30, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "phie", PLAIN_RIDGE_CONFIG)

        assert result["n_train_wells"] == 2
        assert result["n_train_samples"] == 20 + 25
        assert result["n_test_samples"] == 30
        assert len(result["y_true"]) == 30
        assert len(result["y_pred"]) == 30

    def test_predictions_are_clipped_to_0_1(self):
        wells = {
            "A": _well("A", 20, slope=50.0, seed=1),  # exaggerated slope to try to escape [0,1]
            "B": _well("B", 20, slope=50.0, seed=2),
        }
        config = ("sswt", True, 3, "ridge", 1.0)  # instantaneous-attrs + PCA-3 + Ridge
        result = pps._predict_blind(wells, "B", "swe", config)

        assert all(0.0 <= v <= 1.0 for v in result["y_pred"])

    def test_recovers_a_known_relationship_reasonably(self):
        wells = {
            "A": _well("A", 40, slope=2.0, seed=1),
            "B": _well("B", 40, slope=2.0, seed=2),
            "C": _well("C", 40, slope=2.0, seed=3),
        }
        result = pps._predict_blind(wells, "C", "phie", PLAIN_RIDGE_CONFIG)
        assert result["r2"] is not None
        assert result["r2"] > 0.3  # same relationship in every well -- should generalize

    def test_depths_and_twt_are_the_blind_wells_own(self):
        wells = {"A": _well("A", 10, slope=1.0, seed=1), "B": _well("B", 12, slope=1.0, seed=2)}
        config = ("sswt", True, 3, "rf", 1.0)  # instantaneous-attrs + PCA-3 + RandomForest
        result = pps._predict_blind(wells, "B", "vsh", config)
        assert result["depths_m"] == wells["B"].depths_m.tolist()
        assert result["twt_ms"] == wells["B"].twt_ms.tolist()


class TestGetPredictionResultValidation:
    def test_rejects_invalid_target(self):
        with pytest.raises(ValueError, match="target"):
            pps.get_prediction_result("Z-04_RAW", "not_a_target")


# A fast wiggle (period MUCH shorter than the tested window widths) --
# a TIGHT window (< half a period) preserves each output sample's own
# local phase (real variance across outputs); a WIDE window (many full
# periods) averages every output toward the same ~0.5 mean instead.
_WIGGLE_PERIOD_MS = 2.0


def _wiggle(t: np.ndarray) -> np.ndarray:
    return 0.5 + 0.4 * np.sin(2 * np.pi * t / _WIGGLE_PERIOD_MS)


class TestBlockAverage:
    """_block_average: averages a target's RAW per-sample values within
    +/-half_ms of each output seismic sample's own time, returning None
    if the result collapses toward a near-constant signal (see
    MIN_BLOCK_VARIANCE_FRACTION)."""

    def test_tight_window_preserves_local_values(self):
        t_valid = np.linspace(0.0, 40.0, 4000)
        raw = _wiggle(t_valid)
        time_ms = np.linspace(0.0, 40.0, 20)
        uniq_idx = np.array([2, 6, 10, 14, 17])
        tight = pps._block_average(raw, t_valid, uniq_idx, time_ms, half_ms=0.1)
        assert tight is not None
        assert tight.std() > 0.2 * raw.std()

    def test_wide_window_averages_toward_the_global_mean(self):
        # min_variance_fraction=0.0 disables the collapse guard here, to
        # inspect the raw averaging arithmetic in isolation from it (see
        # test_returns_none_when_wide_window_collapses_output_variance
        # for the guard itself, at the same half_ms).
        t_valid = np.linspace(0.0, 40.0, 4000)
        raw = _wiggle(t_valid)
        time_ms = np.linspace(0.0, 40.0, 20)
        uniq_idx = np.array([2, 6, 10, 14, 17])
        wide = pps._block_average(raw, t_valid, uniq_idx, time_ms, half_ms=10.0, min_variance_fraction=0.0)
        assert wide is not None
        np.testing.assert_allclose(wide, 0.5, atol=0.02)  # converged near the wiggle's mean

    def test_returns_none_when_wide_window_collapses_output_variance(self):
        t_valid = np.linspace(0.0, 40.0, 4000)
        raw = _wiggle(t_valid)
        time_ms = np.linspace(0.0, 40.0, 20)
        uniq_idx = np.array([2, 6, 10, 14, 17])
        wide = pps._block_average(raw, t_valid, uniq_idx, time_ms, half_ms=10.0)
        assert wide is None  # collapsed

    def test_does_not_collapse_when_raw_signal_itself_is_flat(self):
        # raw.std() == 0 -- a constant signal isn't a "collapse" (there
        # was never any variance to lose); must not divide by zero either.
        t_valid = np.linspace(0.0, 10.0, 50)
        raw = np.full(50, 0.5)
        time_ms = np.linspace(0.0, 10.0, 5)
        uniq_idx = np.array([0, 1, 2, 3, 4])
        out = pps._block_average(raw, t_valid, uniq_idx, time_ms, half_ms=5.0)
        assert out is not None
        assert out == pytest.approx([0.5] * 5)


def _well_raw_with_wiggle_target(well_id: str, target_key: str, seed: int) -> pps._WellRawCache:
    """A raw cache whose `target_key`'s raw values are the fast wiggle
    above -- internally consistent (features/inst_attrs/uniq_idx all the
    same length) so it can be dropped into a multi-well grid-search test
    to exercise _block_average's collapse guard end to end without
    touching an already-built _well_raw's arrays out of sync."""
    rng = np.random.default_rng(seed)
    n_uniq = 5
    time_ms = np.linspace(0.0, 40.0, 20)
    uniq_idx = np.array([2, 6, 10, 14, 17])
    t_valid = np.linspace(0.0, 40.0, 4000)

    X = rng.normal(loc=0.0, scale=0.1, size=(n_uniq, 4))
    inst_attrs = rng.normal(scale=0.01, size=(n_uniq, 6))
    wiggle = _wiggle(t_valid)
    other = rng.uniform(0, 1, len(t_valid))
    targets_raw = {k: (wiggle if k == target_key else other) for k in ("vsh", "phie", "swe")}

    return pps._WellRawCache(
        well_id=well_id,
        inline_number=100,
        crossline_number=200,
        correlation=0.9,
        best_freq_hz=25.0,
        polarity=1,
        bulk_shift_ms=0.0,
        tie_source="checkshot (1 valid pt)",
        depths_m=np.arange(n_uniq, dtype=float),
        twt_ms=time_ms[uniq_idx],
        features={"cwt": X, "sswt": X.copy()},
        freq_hz={"cwt": np.array([5.0, 10.0, 15.0, 20.0]), "sswt": np.array([5.0, 10.0, 15.0, 20.0])},
        inst_attrs=inst_attrs,
        uniq_idx=uniq_idx,
        time_ms=time_ms,
        t_valid=t_valid,
        targets_raw=targets_raw,
    )


class TestWellFeaturesForBlock:
    def test_returns_none_if_any_well_collapses(self):
        good = _well_raw("A", 30, slope=2.0, seed=1)
        collapsing = _well_raw_with_wiggle_target("B", "vsh", seed=2)
        raw_caches = {"A": good, "B": collapsing}

        result = pps._well_features_for_block(raw_caches, "vsh", block_half_ms=10.0)
        assert result is None

    def test_returns_full_dict_when_no_well_collapses(self):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1), "B": _well_raw("B", 30, slope=2.0, seed=2)}
        result = pps._well_features_for_block(wells, "vsh", block_half_ms=1.0)
        assert result is not None
        assert set(result.keys()) == {"A", "B"}
        assert set(result["A"].targets.keys()) == {"vsh"}  # only the requested target is populated


class TestGridSearch:
    """run_grid_search/get_best_config, against constructed _WellRawCache
    objects with _eligible_well_raw_caches monkeypatched so no real
    SEG-Y volume is needed. Uses a small subset of the real grid
    (monkeypatching the GRID_* constants) so these tests run in a
    reasonable time -- the full 144-candidate x n_wells-fold search is
    exercised implicitly by running with the real constants at least once
    (test_full_grid_runs_and_picks_a_scored_winner).
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        pps._GRID_SEARCH_CACHE.clear()
        yield
        pps._GRID_SEARCH_CACHE.clear()

    def _patch_wells(self, monkeypatch, wells):
        monkeypatch.setattr(pps, "_eligible_well_raw_caches", lambda volume: (wells, []))
        monkeypatch.setattr("app.services.seismic_processor.get_segy_volume", lambda: object())

    def test_picks_the_config_with_the_best_pooled_r2(self, monkeypatch):
        wells = {
            "A": _well_raw("A", 40, slope=2.0, seed=1),
            "B": _well_raw("B", 40, slope=2.0, seed=2),
            "C": _well_raw("C", 40, slope=2.0, seed=3),
        }
        self._patch_wells(monkeypatch, wells)
        # Small search space so this test is fast and deterministic to reason about.
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge", "pls"))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (1.0,))

        result = pps.run_grid_search("phie")

        assert result["best_config"] is not None
        assert result["best_r2"] is not None
        scored = [e for e in result["leaderboard"] if e["r2"] is not None]
        assert result["best_r2"] == max(e["r2"] for e in scored)
        assert len(result["leaderboard"]) == 2  # ridge, pls

    def test_block_half_multiplies_the_candidate_count(self, monkeypatch):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1), "B": _well_raw("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (1.0, 2.0, 4.0))

        result = pps.run_grid_search("phie")

        assert len(result["leaderboard"]) == 3  # 1 spectrum x 1 pca x 1 inst x 1 model x 3 block_half
        block_halves = {entry["config"][4] for entry in result["leaderboard"]}
        assert block_halves == {1.0, 2.0, 4.0}

    def test_a_collapsing_block_half_is_marked_failed_not_crashed(self, monkeypatch):
        good = _well_raw("A", 30, slope=2.0, seed=1)
        collapsing = _well_raw_with_wiggle_target("B", "phie", seed=2)
        wells = {"A": good, "B": collapsing}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (0.1, 10.0))

        result = pps.run_grid_search("phie")

        by_block = {entry["config"][4]: entry for entry in result["leaderboard"]}
        assert by_block[0.1]["error"] is None
        assert by_block[10.0]["error"] is not None
        assert "collapsed" in by_block[10.0]["error"]
        assert result["best_config"][4] == 0.1  # the working block_half won, search didn't crash

    def test_get_best_config_caches_across_calls(self, monkeypatch):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1), "B": _well_raw("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (1.0,))

        calls = {"n": 0}
        original = pps.run_grid_search

        def _counting_search(target):
            calls["n"] += 1
            return original(target)

        monkeypatch.setattr(pps, "run_grid_search", _counting_search)

        first = pps.get_best_config("phie")
        second = pps.get_best_config("phie")

        assert first == second
        assert calls["n"] == 1  # second call reused the cache, didn't re-search

    def test_force_true_bypasses_the_cache(self, monkeypatch):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1), "B": _well_raw("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge",))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (1.0,))

        calls = {"n": 0}
        original = pps.run_grid_search

        def _counting_search(target):
            calls["n"] += 1
            return original(target)

        monkeypatch.setattr(pps, "run_grid_search", _counting_search)

        pps.get_best_config("phie")
        pps.get_best_config("phie", force=True)

        assert calls["n"] == 2

    def test_a_failing_candidate_does_not_crash_the_search(self, monkeypatch):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1), "B": _well_raw("B", 30, slope=2.0, seed=2)}
        self._patch_wells(monkeypatch, wells)
        monkeypatch.setattr(pps, "GRID_SPECTRA", ("sswt",))
        monkeypatch.setattr(pps, "GRID_PCA_OPTIONS", (None,))
        monkeypatch.setattr(pps, "GRID_USE_INST_OPTIONS", (False,))
        monkeypatch.setattr(pps, "GRID_MODEL_NAMES", ("ridge", "not_a_real_model"))
        monkeypatch.setattr(pps, "GRID_BLOCK_HALF_OPTIONS", (1.0,))

        result = pps.run_grid_search("phie")

        failed = [e for e in result["leaderboard"] if e["error"] is not None]
        assert len(failed) == 1
        assert result["best_config"] is not None  # ridge still won despite the other candidate failing

    def test_insufficient_wells_returns_no_winner(self, monkeypatch):
        wells = {"A": _well_raw("A", 30, slope=2.0, seed=1)}
        self._patch_wells(monkeypatch, wells)

        result = pps.run_grid_search("phie")

        assert result["best_config"] is None
        assert result["leaderboard"] == []

    def test_full_grid_runs_and_picks_a_scored_winner(self, monkeypatch):
        """Exercises the REAL (un-monkeypatched) 144-candidate grid at
        least once, so a shape/API mismatch in any (spectrum, PCA,
        inst_attrs, model, block_half) combination -- not just the ones
        the faster tests above cover -- would fail here."""
        wells = {
            "A": _well_raw("A", 40, slope=2.0, seed=1),
            "B": _well_raw("B", 40, slope=2.0, seed=2),
            "C": _well_raw("C", 40, slope=2.0, seed=3),
        }
        self._patch_wells(monkeypatch, wells)

        result = pps.run_grid_search("vsh")

        assert len(result["leaderboard"]) == (
            len(pps.GRID_SPECTRA)
            * len(pps.GRID_PCA_OPTIONS)
            * len(pps.GRID_USE_INST_OPTIONS)
            * len(pps.GRID_MODEL_NAMES)
            * len(pps.GRID_BLOCK_HALF_OPTIONS)
        )
        assert result["best_config"] is not None
        assert result["best_r2"] is not None
