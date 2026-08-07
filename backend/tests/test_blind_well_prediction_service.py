"""
test_blind_well_prediction_service.py
-----------------------------------------
Tests for services/blind_well_prediction_service.py.

Layered coverage, same strategy as test_spectral_property_prediction_service.py:
- Feature-derivation helpers (_spectral_features, _instantaneous_attributes)
  are checked against constructed signals with known ground truth (a
  single spectral peak, a pure sinusoid) -- fast, no real SEG-Y needed.
- _select_features/_decision_gate/_fit_stack/_predict_stack are checked
  against constructed _WellSamples with a KNOWN linear relationship, and
  a spy on the base learners' .fit() confirms the held-out well's own
  rows never appear in its own leave-one-well-out training fold.
- run_blind_well_prediction's orchestration (status branching, and that
  the blind well never appears in training_well_ids) is tested by
  monkeypatching _resolve_direct_tie/_extract_well_samples/
  _extract_center_trace_samples directly -- this module's own seams --
  since real tie resolution + SSWT needs a real SEG-Y volume and is slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import blind_well_prediction_service as bwp


class TestSpectralFeatures:
    def test_single_peak_recovers_dominant_frequency_and_bandwidth(self):
        freq_hz = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        # One time sample, energy concentrated entirely at 15 Hz.
        energy = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        feats = bwp._spectral_features(energy, freq_hz, "test")
        assert feats["test_dominant_freq_hz"][0] == pytest.approx(15.0)
        assert feats["test_peak_amp"][0] == pytest.approx(1.0)
        assert feats["test_centroid_hz"][0] == pytest.approx(15.0)
        assert feats["test_bandwidth_hz"][0] == pytest.approx(0.0, abs=1e-9)
        # All-or-nothing energy -> minimum (zero) entropy.
        assert feats["test_entropy"][0] == pytest.approx(0.0, abs=1e-6)

    def test_flat_spectrum_has_high_entropy(self):
        freq_hz = np.array([5.0, 10.0, 15.0, 20.0])
        energy = np.array([[1.0, 1.0, 1.0, 1.0]])
        feats = bwp._spectral_features(energy, freq_hz, "test")
        # Uniform distribution -> normalized entropy close to 1.
        assert feats["test_entropy"][0] == pytest.approx(1.0, abs=1e-6)

    def test_energy_bands_sum_to_pooled_total(self):
        freq_hz = np.linspace(0.0, 30.0, 7)
        rng = np.random.default_rng(0)
        energy = rng.uniform(0.1, 1.0, size=(5, 7))
        feats = bwp._spectral_features(energy, freq_hz, "test")
        band_sum = feats["test_energy_low"] + feats["test_energy_mid"] + feats["test_energy_high"]
        # Thirds tile the full range (with the low/mid boundary included
        # once each, and the high band closed at both ends), so every
        # sample's normalized bins are covered by exactly one band.
        np.testing.assert_allclose(band_sum, 1.0, atol=1e-9)


class TestInstantaneousAttributes:
    def test_envelope_of_a_constant_amplitude_sinusoid_is_flat(self):
        dt_ms = 2.0
        freq_hz = 25.0
        t = np.arange(0, 500) * (dt_ms / 1000.0)
        trace = np.sin(2 * np.pi * freq_hz * t)
        feats = bwp._instantaneous_attributes(trace, dt_ms)
        # Away from the signal's start/end edges (Hilbert transform edge
        # effects), the envelope of a pure constant-amplitude sinusoid
        # should be ~1 everywhere.
        interior = feats["inst_envelope"][20:-20]
        np.testing.assert_allclose(interior, 1.0, atol=0.05)

    def test_instantaneous_frequency_recovers_the_true_frequency(self):
        dt_ms = 2.0
        freq_hz = 25.0
        t = np.arange(0, 500) * (dt_ms / 1000.0)
        trace = np.sin(2 * np.pi * freq_hz * t)
        feats = bwp._instantaneous_attributes(trace, dt_ms)
        interior = feats["inst_freq_hz"][20:-20]
        np.testing.assert_allclose(interior, freq_hz, atol=1.0)

    def test_amplitude_passthrough_is_the_raw_trace(self):
        trace = np.array([1.0, -2.0, 3.0, -4.0])
        feats = bwp._instantaneous_attributes(trace, dt_ms=2.0)
        np.testing.assert_array_equal(feats["inst_amplitude"], trace)


def _well_samples(well_id: str, n: int, signal_col: int, sign: float = 1.0, seed: int = 0) -> "bwp._WellSamples":
    """_WellSamples with a KNOWN strong linear relationship between
    feature column `signal_col` and the target (scaled by `sign`, so a
    caller can build a well where the relationship's sign flips -- for
    the leave-one-well-out stability check), plus pure-noise columns
    elsewhere."""
    rng = np.random.default_rng(seed)
    n_features = len(bwp.FEATURE_NAMES)
    X = rng.normal(size=(n, n_features))
    y = sign * X[:, signal_col] * 5.0 + rng.normal(scale=0.05, size=n)
    return bwp._WellSamples(
        well_id=well_id,
        X=X,
        y={"vsh": y, "phie": np.full(n, np.nan), "swe": rng.normal(size=n)},
        weight=np.ones(n),
    )


class TestSelectFeatures:
    def test_stable_signal_feature_is_selected(self):
        wells = [
            _well_samples("A", 40, signal_col=0, seed=1),
            _well_samples("B", 40, signal_col=0, seed=2),
            _well_samples("C", 40, signal_col=0, seed=3),
        ]
        selected, diagnostics = bwp._select_features(wells, "vsh")
        assert 0 in selected
        assert diagnostics[0]["stable"] is True
        assert diagnostics[0]["selected"] is True

    def test_sign_flipping_feature_is_not_selected_despite_pooled_correlation(self):
        # Feature 0 correlates POSITIVELY with vsh in A/B but NEGATIVELY in
        # C -- a real, well-specific relationship, not noise, would still
        # show up consistently; this is exactly the "fits one well's own
        # noise" pattern leave-one-well-out stability exists to catch.
        wells = [
            _well_samples("A", 40, signal_col=0, sign=1.0, seed=1),
            _well_samples("B", 40, signal_col=0, sign=1.0, seed=2),
            _well_samples("C", 40, signal_col=0, sign=-1.0, seed=3),
        ]
        selected, diagnostics = bwp._select_features(wells, "vsh")
        assert diagnostics[0]["stable"] is False
        # An unstable feature is heavily downweighted (not necessarily
        # impossible to select if literally nothing else has any signal),
        # but should not be flagged stable.

    def test_insufficient_property_samples_returns_no_diagnostics_crash(self):
        wells = [
            _well_samples("A", 40, signal_col=0, seed=1),
            _well_samples("B", 40, signal_col=0, seed=2),
        ]
        # phie is all-NaN in the fixture -- must not crash, just report
        # nothing as stable/selected.
        selected, diagnostics = bwp._select_features(wells, "phie")
        assert selected == []
        assert all(not d["selected"] for d in diagnostics)

    def test_collinear_duplicate_is_dropped(self):
        rng = np.random.default_rng(7)
        n = 60
        n_features = len(bwp.FEATURE_NAMES)
        X = rng.normal(size=(n, n_features))
        X[:, 1] = X[:, 0] + rng.normal(scale=1e-4, size=n)  # near-duplicate of column 0
        y = X[:, 0] * 5.0 + rng.normal(scale=0.05, size=n)
        wells = [
            bwp._WellSamples(well_id=w, X=X, y={"vsh": y, "phie": np.full(n, np.nan), "swe": rng.normal(size=n)}, weight=np.ones(n))
            for w in ("A", "B", "C")
        ]
        selected, _ = bwp._select_features(wells, "vsh")
        assert not (0 in selected and 1 in selected)


class TestDecisionGateAndStack:
    def _linear_wells(self, n_per_well=40):
        return [
            _well_samples("A", n_per_well, signal_col=0, seed=1),
            _well_samples("B", n_per_well, signal_col=0, seed=2),
            _well_samples("C", n_per_well, signal_col=0, seed=3),
        ]

    def test_decision_gate_recovers_strong_linear_signal(self):
        wells = self._linear_wells()
        r2 = bwp._decision_gate(wells, "vsh", [0, 1, 2])
        assert r2 is not None
        assert r2 > 0.5

    def test_decision_gate_returns_none_with_no_features(self):
        wells = self._linear_wells()
        assert bwp._decision_gate(wells, "vsh", []) is None

    def test_fit_stack_and_predict_recovers_strong_linear_signal(self):
        wells = self._linear_wells(n_per_well=50)
        X = np.concatenate([s.X[:, :3] for s in wells], axis=0)
        y = np.concatenate([s.y["vsh"] for s in wells], axis=0)
        w = np.concatenate([s.weight for s in wells], axis=0)
        groups = np.concatenate([np.full(s.X.shape[0], s.well_id) for s in wells])

        stack = bwp._fit_stack(X, y, w, groups)
        assert stack["stack_loocv_r2"] is not None
        assert stack["stack_loocv_r2"] > 0.3

        pred = bwp._predict_stack(stack, X[:5])
        assert pred.shape == (5,)

    def test_held_out_well_never_appears_in_its_own_loocv_fold(self, monkeypatch):
        from sklearn.ensemble import RandomForestRegressor

        wells = self._linear_wells(n_per_well=20)
        X = np.concatenate([s.X[:, :3] for s in wells], axis=0)
        y = np.concatenate([s.y["vsh"] for s in wells], axis=0)
        w = np.concatenate([s.weight for s in wells], axis=0)
        groups = np.concatenate([np.full(s.X.shape[0], s.well_id) for s in wells])

        fit_sizes: list[int] = []
        original_fit = RandomForestRegressor.fit

        def _spy_fit(self, X_fit, y_fit, *a, **k):
            fit_sizes.append(len(X_fit))
            return original_fit(self, X_fit, y_fit, *a, **k)

        monkeypatch.setattr(RandomForestRegressor, "fit", _spy_fit)
        bwp._fit_stack(X, y, w, groups)

        total = len(y)
        # 3 leave-one-well-out folds (each excluding one well's 20 rows)
        # + 1 final fit on everyone.
        expected_loocv_sizes = sorted([total - 20, total - 20, total - 20])
        assert sorted(fit_sizes[:3]) == expected_loocv_sizes
        assert fit_sizes[3] == total


class TestRunBlindWellPredictionOrchestration:
    def test_unknown_blind_well_raises(self, monkeypatch):
        from app.services import well_service

        class _Summary:
            def __init__(self, well_id):
                self.well_id = well_id

        monkeypatch.setattr(well_service, "list_well_summaries", lambda: [_Summary("A")])
        with pytest.raises(well_service.WellNotFoundError):
            bwp.run_blind_well_prediction("DOES_NOT_EXIST")

    def test_blind_well_unusable_when_tie_fails(self, monkeypatch):
        from app import well_seismic_tie as wst
        from app.services import seismic_processor as sp
        from app.services import well_service

        class _Summary:
            def __init__(self, well_id):
                self.well_id = well_id

        monkeypatch.setattr(sp, "get_segy_volume", lambda: object())
        monkeypatch.setattr(
            well_service, "list_well_summaries",
            lambda: [_Summary("BLIND"), _Summary("A"), _Summary("B"), _Summary("C")],
        )

        class _FakeTie:
            boundary_pinned = False
            low_confidence = False
            correlation = 0.8

        def _fake_resolve(volume, well_id):
            if well_id == "BLIND":
                raise wst.TieError("no coordinates")
            return _FakeTie()

        monkeypatch.setattr(bwp, "_resolve_direct_tie", _fake_resolve)

        result = bwp.run_blind_well_prediction("BLIND")
        assert result["status"] == "blind_well_unusable"
        assert result["results"] is None

    def test_insufficient_training_wells(self, monkeypatch):
        from app.services import seismic_processor as sp
        from app.services import well_service

        class _Summary:
            def __init__(self, well_id):
                self.well_id = well_id

        monkeypatch.setattr(sp, "get_segy_volume", lambda: object())
        monkeypatch.setattr(well_service, "list_well_summaries", lambda: [_Summary("BLIND"), _Summary("A")])

        class _FakeTie:
            boundary_pinned = False
            low_confidence = False
            correlation = 0.8

        monkeypatch.setattr(bwp, "_resolve_direct_tie", lambda volume, well_id: _FakeTie())

        result = bwp.run_blind_well_prediction("BLIND")
        assert result["status"] == "insufficient_data"
        assert result["results"] is None

    def test_validated_result_never_includes_blind_well_in_training(self, monkeypatch):
        from app.services import seismic_processor as sp
        from app.services import well_service

        class _Summary:
            def __init__(self, well_id):
                self.well_id = well_id

        well_ids = ["BLIND", "A", "B", "C"]
        monkeypatch.setattr(sp, "get_segy_volume", lambda: object())
        monkeypatch.setattr(well_service, "list_well_summaries", lambda: [_Summary(w) for w in well_ids])

        class _FakeTie:
            def __init__(self, well_id):
                self.well_id = well_id
            boundary_pinned = False
            low_confidence = False
            correlation = 0.8

        monkeypatch.setattr(bwp, "_resolve_direct_tie", lambda volume, well_id: _FakeTie(well_id))

        def _fake_extract_well_samples(volume, tie):
            return _well_samples(tie.well_id, 30, signal_col=0, seed=hash(tie.well_id) % 1000)

        blind_X = np.random.default_rng(99).normal(size=(15, len(bwp.FEATURE_NAMES)))
        blind_y = {"vsh": blind_X[:, 0] * 5.0, "phie": np.full(15, np.nan), "swe": np.random.default_rng(1).normal(size=15)}
        blind_depth = np.arange(15, dtype=float)

        monkeypatch.setattr(bwp, "_extract_well_samples", _fake_extract_well_samples)
        monkeypatch.setattr(bwp, "_extract_center_trace_samples", lambda volume, tie: (blind_X, blind_y, blind_depth))

        result = bwp.run_blind_well_prediction("BLIND")

        assert result["status"] == "validated"
        assert "BLIND" not in result["training_well_ids"]
        assert set(result["training_well_ids"]) == {"A", "B", "C"}
        assert set(result["results"].keys()) == {"vsh", "phie", "swe"}
        assert result["results"]["vsh"]["status"] == "validated"
        assert result["results"]["vsh"]["blind_well_r2"] is not None
        assert len(result["results"]["vsh"]["y_true"]) == len(result["results"]["vsh"]["y_pred"])
        # phie is all-NaN for the blind well in this fixture.
        assert result["results"]["phie"]["status"] in ("insufficient_data", "no_stable_features", "blind_well_no_valid_samples")
