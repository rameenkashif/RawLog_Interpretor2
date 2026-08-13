"""
test_sweet_spot_feature_engine.py
-------------------------------------
Tests for services/sweet_spot_feature_engine.py, the 55(+3 SI)-attribute
seismic feature pool for the Sweet-Spot prediction module.

Layered coverage, same strategy as test_blind_well_prediction_service.py:
- Individual family functions checked against constructed signals with
  known ground truth (a pure sinusoid's instantaneous frequency/envelope,
  a single-frequency trace's CWT-band energy concentration, a hand-built
  spatial grid's known gradient) -- fast, no real SEG-Y needed.
- compute_feature_pool's shape/naming contract and CURATED_FEATURES
  completeness are checked directly against a constructed 3x3 region.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import sweet_spot_feature_engine as fe


class TestShiftHelper:
    def test_shift_zero_returns_copy_of_input(self):
        signal = np.arange(10, dtype=float).reshape(10, 1)
        out = fe._shift(signal, 0)
        np.testing.assert_allclose(out, signal)
        assert out is not signal

    def test_positive_shift_pulls_deeper_value(self):
        signal = np.arange(10, dtype=float).reshape(10, 1)
        out = fe._shift(signal, 2)
        # out[t] should equal signal[t+2] for valid t
        np.testing.assert_allclose(out[:8, 0], signal[2:, 0])
        assert np.isnan(out[8:, 0]).all()

    def test_negative_shift_pulls_shallower_value(self):
        signal = np.arange(10, dtype=float).reshape(10, 1)
        out = fe._shift(signal, -2)
        np.testing.assert_allclose(out[2:, 0], signal[:8, 0])
        assert np.isnan(out[:2, 0]).all()

    def test_shift_key_naming_matches_doc_convention(self):
        assert fe._shift_key("amp", 0) == "amp_center"
        assert fe._shift_key("amp", 2) == "amp_shift_+2"
        assert fe._shift_key("amp", -3) == "amp_shift_-3"
        assert fe._shift_key("sweetness", 0, center_name="sweetness") == "sweetness"


class TestAmplitudeFamily:
    def test_center_matches_input_exactly(self):
        traces = np.random.default_rng(0).normal(size=(50, 3))
        out = fe.amplitude_family(traces)
        np.testing.assert_allclose(out["amp_center"], traces)

    def test_returns_11_keys(self):
        traces = np.zeros((50, 2))
        out = fe.amplitude_family(traces)
        assert len(out) == 11
        assert set(out.keys()) == {
            "amp_center", *[f"amp_shift_+{k}" for k in range(1, 6)], *[f"amp_shift_-{k}" for k in range(1, 6)],
        }


class TestEnvelopeFamily:
    def test_envelope_of_pure_sinusoid_is_flat(self):
        n = 200
        t = np.arange(n)
        freq_cycles_per_sample = 0.05
        traces = np.sin(2 * np.pi * freq_cycles_per_sample * t).reshape(n, 1)
        out = fe.envelope_family(traces)
        interior = out["env_center"][20:-20, 0]
        # A pure sinusoid's Hilbert envelope should be ~constant (~1.0),
        # away from edge transients.
        assert interior.std() < 0.05
        assert interior.mean() == pytest.approx(1.0, abs=0.05)

    def test_deriv_is_zero_for_flat_envelope(self):
        n = 200
        t = np.arange(n)
        traces = np.sin(2 * np.pi * 0.05 * t).reshape(n, 1)
        out = fe.envelope_family(traces)
        interior_deriv = out["env_deriv"][20:-20, 0]
        assert np.abs(interior_deriv).max() < 0.05


class TestInstFreqFamily:
    def test_recovers_known_sinusoid_frequency(self):
        dt_ms = 2.0
        true_freq_hz = 25.0
        n = 300
        t_s = np.arange(n) * (dt_ms / 1000.0)
        traces = np.sin(2 * np.pi * true_freq_hz * t_s).reshape(n, 1)
        out = fe.inst_freq_family(traces, dt_ms)
        interior = out["ifreq_center"][30:-30, 0]
        assert interior.mean() == pytest.approx(true_freq_hz, rel=0.1)


class TestSweetnessFamily:
    def test_matches_manual_formula(self):
        rng = np.random.default_rng(1)
        envelope = np.abs(rng.normal(size=(50, 2))) + 1.0
        inst_freq = rng.normal(20, 5, size=(50, 2))
        out = fe.sweetness_family(envelope, inst_freq)
        expected_center = envelope / np.sqrt(np.abs(inst_freq) + 1e-6)
        np.testing.assert_allclose(out["sweetness"], expected_center, rtol=1e-6)

    def test_center_key_has_no_suffix(self):
        envelope = np.ones((10, 1))
        inst_freq = np.full((10, 1), 10.0)
        out = fe.sweetness_family(envelope, inst_freq)
        assert "sweetness" in out
        assert "sweetness_center" not in out


class TestCwtBandEnergyFamily:
    def test_single_frequency_trace_concentrates_energy_in_matching_band(self):
        dt_ms = 2.0
        n = 300
        t_s = np.arange(n) * (dt_ms / 1000.0)
        # 30 Hz falls inside the 27-33 Hz band and no other band.
        traces = np.sin(2 * np.pi * 30.0 * t_s).reshape(n, 1)
        out = fe.cwt_band_energy_family(traces, dt_ms)
        interior_30 = out["spec_amp_30hz"][30:-30, 0].mean()
        interior_10 = out["spec_amp_10hz"][30:-30, 0].mean()
        assert interior_30 > interior_10 * 3

    def test_returns_5_bands(self):
        traces = np.zeros((300, 2))
        out = fe.cwt_band_energy_family(traces, 2.0)
        assert set(out.keys()) == {"spec_amp_10hz", "spec_amp_15hz", "spec_amp_20hz", "spec_amp_30hz", "spec_amp_40hz"}


class TestRollingWindowFamily:
    def test_matches_manual_window_stats(self):
        signal = np.array([1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0, 8.0]).reshape(8, 1)
        out = fe.rolling_window_family(signal, window=5)
        # Window centered at index 3 (half=2): samples [1,2,3,10,5] -> idx 1..5
        window_at_3 = signal[1:6, 0]
        assert out["win_max"][3, 0] == pytest.approx(window_at_3.max())
        assert out["win_mean"][3, 0] == pytest.approx(window_at_3.mean())
        assert out["win_min"][3, 0] == pytest.approx(window_at_3.min())
        assert out["win_std"][3, 0] == pytest.approx(window_at_3.std())

    def test_edges_are_nan(self):
        signal = np.arange(10, dtype=float).reshape(10, 1)
        out = fe.rolling_window_family(signal, window=5)
        assert np.isnan(out["win_mean"][:2, 0]).all()
        assert np.isnan(out["win_mean"][-2:, 0]).all()
        assert not np.isnan(out["win_mean"][2:-2, 0]).any()


class TestNeighborGridDiff:
    def test_positive_offset_diff_and_edge_nan(self):
        # 3x2 spatial grid, 1 time sample -- values chosen so il-gradient is
        # simple to verify by hand: value = 10*il + xl.
        grid = np.zeros((3, 2, 1))
        for il in range(3):
            for xl in range(2):
                grid[il, xl, 0] = 10 * il + xl
        diff = fe._neighbor_grid_diff(grid, axis=0, offset=1)
        # diff[il,xl] = grid[il+1,xl] - grid[il,xl] = 10 for il=0,1; NaN at il=2 (no il+1)
        assert diff[0, 0, 0] == pytest.approx(10.0)
        assert diff[1, 0, 0] == pytest.approx(10.0)
        assert np.isnan(diff[2, 0, 0])

    def test_negative_offset_diff_and_edge_nan(self):
        grid = np.zeros((3, 2, 1))
        for il in range(3):
            for xl in range(2):
                grid[il, xl, 0] = 10 * il + xl
        diff = fe._neighbor_grid_diff(grid, axis=0, offset=-1)
        # diff[il,xl] = grid[il-1,xl] - grid[il,xl] = -10 for il=1,2; NaN at il=0
        assert np.isnan(diff[0, 0, 0])
        assert diff[1, 0, 0] == pytest.approx(-10.0)
        assert diff[2, 0, 0] == pytest.approx(-10.0)


class TestStructuralFamily:
    def test_polarity_index_bounded(self):
        rng = np.random.default_rng(2)
        traces_3d = rng.normal(0, 1, size=(3, 3, 100))
        out = fe.structural_family(traces_3d, dt_ms=2.0)
        finite = out["polarity_index"][~np.isnan(out["polarity_index"])]
        assert (finite >= -1.0).all() and (finite <= 1.0).all()

    def test_rel_pos_spans_0_to_1_and_is_identical_across_positions(self):
        traces_3d = np.zeros((2, 2, 50))
        out = fe.structural_family(traces_3d, dt_ms=2.0)
        assert out["rel_pos"][0, 0] == pytest.approx(0.0)
        assert out["rel_pos"][-1, 0] == pytest.approx(1.0)
        # Same rel_pos curve at every spatial position.
        np.testing.assert_allclose(out["rel_pos"][:, 0], out["rel_pos"][:, 1])

    def test_acoustic_impedance_is_cumulative_sum(self):
        traces_3d = np.zeros((1, 1, 5))
        traces_3d[0, 0, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = fe.structural_family(traces_3d, dt_ms=2.0)
        np.testing.assert_allclose(out["acoustic_impedance"][:, 0], np.cumsum([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_inline_gradient_matches_manual_diff(self):
        # Constant-per-il-row amplitude -> deterministic AI-diff across inline.
        traces_3d = np.zeros((3, 2, 10))
        for il in range(3):
            traces_3d[il, :, :] = float(il + 1)  # row il is all (il+1)
        out = fe.structural_family(traces_3d, dt_ms=2.0)
        ai = out["acoustic_impedance"]  # (n_time, n_pos), n_pos = 3*2=6, C-order (il,xl)
        # position (il=0,xl=0) -> column 0; (il=1,xl=0) -> column 2
        diff_col0 = out["n_il_p1_acoustic_impedance_diff"][:, 0]
        expected = ai[:, 2] - ai[:, 0]
        np.testing.assert_allclose(diff_col0, expected)


class TestSiFeatures:
    def test_energy_fractions_are_bounded_0_to_1(self):
        rng = np.random.default_rng(3)
        traces_3d = rng.normal(0, 1, size=(2, 2, 300))
        pool = fe.compute_feature_pool(traces_3d, dt_ms=2.0)
        for name in ("si_spec_frac_10", "si_spec_frac_40"):
            finite = pool[name][~np.isnan(pool[name])]
            assert (finite >= -1e-9).all() and (finite <= 1.0 + 1e-9).all()


class TestComputeFeaturePool:
    def test_all_curated_features_present_with_correct_shape(self):
        rng = np.random.default_rng(4)
        n_il, n_xl, n_time = 3, 3, 150
        traces_3d = rng.normal(0, 1, size=(n_il, n_xl, n_time))
        pool = fe.compute_feature_pool(traces_3d, dt_ms=2.0)
        for name in fe.CURATED_FEATURES:
            assert name in pool, f"{name} missing from pool"
            assert pool[name].shape == (n_time, n_il * n_xl)

    def test_curated_features_list_has_22_entries(self):
        assert len(fe.CURATED_FEATURES) == 22

    def test_flatten_order_is_c_order_il_major(self):
        # position index = il_idx * n_xl + xl_idx
        traces_3d = np.zeros((2, 3, 10))
        traces_3d[1, 2, :] = 99.0  # il=1, xl=2 -> flat index 1*3+2=5
        pool = fe.compute_feature_pool(traces_3d, dt_ms=2.0)
        assert pool["attr_amp_center"][0, 5] == pytest.approx(99.0)
        assert pool["attr_amp_center"][0, 0] == pytest.approx(0.0)

    def test_works_on_single_trace_region(self):
        # n_pos=1 -- exercises the whole pipeline for a lone trace (the
        # eventual "just this one location" case), not just multi-trace regions.
        traces_3d = np.random.default_rng(5).normal(size=(1, 1, 100))
        pool = fe.compute_feature_pool(traces_3d, dt_ms=2.0)
        for name in fe.CURATED_FEATURES:
            assert pool[name].shape == (100, 1)
