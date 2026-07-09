import numpy as np
import pytest

from app.well_seismic_tie import (
    TieError,
    acoustic_impedance,
    apply_stretch_squeeze,
    build_synthetic,
    calibrate_gardner_coefficients,
    cross_correlate_and_shift,
    depth_to_twt,
    extract_statistical_wavelet,
    find_nearest_trace_index,
    gardner_density,
    reflectivity_series,
    ricker_wavelet,
    rock_physics_density,
    washout_qc_flag,
    wavelet_spectra,
)


def test_depth_to_twt_increasing():
    depth = np.linspace(3000, 3100, 50)
    dt = np.full(50, 80.0)  # constant slowness, us/ft
    twt = depth_to_twt(depth, dt, dt_unit="us_per_ft")
    assert np.all(np.diff(twt) > 0)
    assert twt[0] == 0.0


def test_depth_to_twt_t0_ms_shifts_whole_curve():
    depth = np.linspace(3000, 3100, 50)
    dt = np.full(50, 80.0)
    baseline = depth_to_twt(depth, dt, dt_unit="us_per_ft")
    shifted = depth_to_twt(depth, dt, dt_unit="us_per_ft", t0_ms=2030.0)
    np.testing.assert_allclose(shifted, baseline + 2030.0)
    assert shifted[0] == pytest.approx(2030.0)


def test_reflectivity_zero_for_constant_impedance():
    ai = np.full(20, 5000.0)
    refl = reflectivity_series(ai)
    assert np.allclose(refl, 0.0)


def test_ricker_wavelet_zero_phase_peak_at_center():
    t, w = ricker_wavelet(30.0, 0.002)
    center_idx = len(w) // 2
    assert np.argmax(w) in range(center_idx - 1, center_idx + 2)


def test_cross_correlate_identical_traces_gives_correlation_1():
    dt_ms = 2.0
    t = np.linspace(0, 100, 200)
    trace = np.sin(2 * np.pi * 0.05 * t)
    result = cross_correlate_and_shift(trace, trace, dt_ms)
    assert result["correlation"] > 0.99
    assert result["best_shift_ms"] == 0.0


def test_cross_correlate_shifted_trace_recovers_shift():
    dt_ms = 2.0
    t = np.linspace(0, 100, 200)
    trace = np.sin(2 * np.pi * 0.05 * t)
    shifted = np.roll(trace, 5)
    result = cross_correlate_and_shift(shifted, trace, dt_ms)
    assert result["correlation"] > 0.9


def test_build_synthetic_end_to_end_smoke():
    depth = np.linspace(3495, 3722, 500)
    dt_log = 87.0 + 5.0 * np.sin(np.linspace(0, 10, 500))
    rhob = 2.5 + 0.05 * np.cos(np.linspace(0, 10, 500))
    # A real (non-zero-delay) seismic time axis, matching the actual
    # production survey's 2030 ms delay -- deliberately NOT zero-based, so
    # this exercises the t0_ms auto-anchoring (see depth_to_twt/
    # build_synthetic docstrings): a well's sonic-integrated TWT curve
    # starts at 0 ms on its own and has zero overlap with a delayed real
    # axis unless anchored, which used to silently produce an all-zero
    # synthetic (isfinite(0.0) is True, so a weaker assertion here
    # wouldn't have caught that regression).
    twt_axis = np.arange(2030.0, 2654.0, 2.0)

    result = build_synthetic(
        depth_m=depth,
        dt_log=dt_log,
        rhob=rhob,
        seismic_dt_ms=2.0,
        seismic_twt_axis_ms=twt_axis,
        wavelet_freq_hz=30.0,
        dt_unit="us_per_ft",
    )
    assert len(result.synthetic) == len(twt_axis)
    assert np.isfinite(result.synthetic).any()
    assert np.max(np.abs(result.synthetic)) > 0.0


def test_build_synthetic_default_anchors_to_seismic_axis_start():
    """Without an explicit t0_ms, the synthetic must be anchored to
    seismic_twt_axis_ms[0], not left at the well's own 0-based integration
    start -- otherwise it has no overlap with a real (delayed) seismic axis
    and silently comes out all zero."""
    depth = np.linspace(3495, 3722, 500)
    dt_log = np.full(500, 87.0)
    rhob = 2.5 + 0.05 * np.cos(np.linspace(0, 10, 500))
    twt_axis = np.arange(2030.0, 2654.0, 2.0)

    result = build_synthetic(
        depth_m=depth, dt_log=dt_log, rhob=rhob, seismic_dt_ms=2.0,
        seismic_twt_axis_ms=twt_axis, dt_unit="us_per_ft",
    )
    assert result.reflectivity_twt_ms[0] == pytest.approx(2030.0, abs=1.0)


def test_build_synthetic_explicit_t0_ms_overrides_default():
    depth = np.linspace(3495, 3722, 500)
    dt_log = np.full(500, 87.0)
    rhob = 2.5 + 0.05 * np.cos(np.linspace(0, 10, 500))
    twt_axis = np.arange(2030.0, 2654.0, 2.0)

    result = build_synthetic(
        depth_m=depth, dt_log=dt_log, rhob=rhob, seismic_dt_ms=2.0,
        seismic_twt_axis_ms=twt_axis, dt_unit="us_per_ft", t0_ms=500.0,
    )
    assert result.reflectivity_twt_ms[0] == pytest.approx(500.0, abs=1.0)


class TestFindNearestTraceIndex:
    def test_picks_closest_trace(self):
        trace_x = np.array([0.0, 100.0, 200.0, 300.0])
        trace_y = np.array([0.0, 0.0, 0.0, 0.0])
        idx, dist = find_nearest_trace_index(210.0, 5.0, trace_x, trace_y)
        assert idx == 2
        assert dist == pytest.approx(np.hypot(10.0, 5.0))

    def test_exact_match_gives_zero_distance(self):
        trace_x = np.array([500340.0, 512340.0, 520000.0])
        trace_y = np.array([6540000.0, 6543210.0, 6550000.0])
        idx, dist = find_nearest_trace_index(512340.0, 6543210.0, trace_x, trace_y)
        assert idx == 1
        assert dist == pytest.approx(0.0)

    def test_raises_when_outside_max_radius(self):
        trace_x = np.array([0.0, 10000.0])
        trace_y = np.array([0.0, 0.0])
        with pytest.raises(TieError):
            find_nearest_trace_index(5000.0, 5000.0, trace_x, trace_y, max_radius_m=100.0)

    def test_within_max_radius_succeeds(self):
        trace_x = np.array([0.0, 50.0])
        trace_y = np.array([0.0, 0.0])
        idx, dist = find_nearest_trace_index(40.0, 0.0, trace_x, trace_y, max_radius_m=100.0)
        assert idx == 1
        assert dist == pytest.approx(10.0)


class TestGardnerDensity:
    def test_calibration_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        velocity = rng.uniform(3000, 6000, 200)
        true_a, true_b = 0.31, 0.25
        rhob = gardner_density(velocity, true_a, true_b)  # noiseless -> exact recovery
        a, b = calibrate_gardner_coefficients(velocity, rhob)
        assert a == pytest.approx(true_a, rel=1e-3)
        assert b == pytest.approx(true_b, rel=1e-3)

    def test_calibration_raises_with_too_few_samples(self):
        velocity = np.array([4000.0, 4100.0, 4200.0])
        rhob = np.array([2.4, 2.41, 2.42])
        with pytest.raises(TieError):
            calibrate_gardner_coefficients(velocity, rhob)

    def test_calibration_ignores_invalid_samples(self):
        rng = np.random.default_rng(1)
        velocity = rng.uniform(3000, 6000, 200)
        rhob = gardner_density(velocity, 0.31, 0.25)
        velocity_with_nans = velocity.copy()
        velocity_with_nans[:50] = np.nan
        a, b = calibrate_gardner_coefficients(velocity_with_nans, rhob)
        assert a == pytest.approx(0.31, rel=1e-2)
        assert b == pytest.approx(0.25, rel=1e-2)


class TestRockPhysicsDensity:
    def test_zero_vsh_zero_phie_gives_matrix_density(self):
        result = rock_physics_density(np.array([0.0]), np.array([0.0]), rho_matrix=2.65)
        assert result[0] == pytest.approx(2.65)

    def test_full_shale_uses_shale_matrix_density(self):
        result = rock_physics_density(np.array([1.0]), np.array([0.0]), rho_shale=2.75)
        assert result[0] == pytest.approx(2.75)

    def test_full_porosity_gives_fluid_density(self):
        result = rock_physics_density(np.array([0.3]), np.array([1.0]), rho_fluid=1.0)
        assert result[0] == pytest.approx(1.0)


class TestWashoutQcFlag:
    def test_dt_spike_is_flagged(self):
        n = 100
        dt_log = np.full(n, 80.0)
        dt_log[50] = 300.0  # sharp, isolated spike
        nphi = np.full(n, 0.2)
        rhob = np.full(n, 2.4)
        flags = washout_qc_flag(nphi, rhob, dt_log)
        assert flags[50]
        assert not flags[:45].any()
        assert not flags[55:].any()

    def test_nphi_rhob_crossover_is_flagged(self):
        n = 100
        nphi = np.full(n, 0.2)
        rhob = np.full(n, 2.4)  # density porosity ~= (2.65-2.4)/(2.65-1.0) = 0.15, close to NPHI
        nphi[30] = 0.9  # gross crossover
        dt_log = np.full(n, 80.0)
        flags = washout_qc_flag(nphi, rhob, dt_log)
        assert flags[30]

    def test_clean_interval_not_flagged(self):
        n = 100
        rng = np.random.default_rng(2)
        dt_log = 80 + rng.normal(0, 0.5, n)
        nphi = np.full(n, 0.15)
        rhob = np.full(n, 2.45)  # density porosity ~0.12, close to NPHI
        flags = washout_qc_flag(nphi, rhob, dt_log)
        assert not flags.any()

    def test_nan_inputs_not_flagged(self):
        n = 30
        nphi = np.full(n, np.nan)
        rhob = np.full(n, 2.4)
        dt_log = np.full(n, 80.0)
        flags = washout_qc_flag(nphi, rhob, dt_log)
        assert not flags.any()


class TestStatisticalWaveletExtraction:
    def test_wavelet_is_centered_and_normalized(self):
        rng = np.random.default_rng(3)
        trace = rng.normal(0, 1, 300)
        t_ms, wavelet = extract_statistical_wavelet(trace, dt_ms=2.0, length_ms=64.0)
        assert len(t_ms) == len(wavelet)
        assert t_ms[len(t_ms) // 2] == pytest.approx(0.0)
        assert np.max(np.abs(wavelet)) == pytest.approx(1.0)

    def test_too_short_trace_raises(self):
        with pytest.raises(TieError):
            extract_statistical_wavelet(np.array([1.0, 2.0]), dt_ms=2.0)


class TestApplyStretchSqueeze:
    def test_no_tie_points_is_noop(self):
        twt = np.array([100.0, 200.0, 300.0])
        depth = np.array([1000.0, 1010.0, 1020.0])
        result = apply_stretch_squeeze(depth, twt, [])
        np.testing.assert_array_equal(result, twt)

    def test_single_tie_point_shifts_uniformly(self):
        twt = np.array([100.0, 200.0, 300.0])
        depth = np.array([1000.0, 1010.0, 1020.0])
        result = apply_stretch_squeeze(depth, twt, [(1010.0, 5.0)])
        np.testing.assert_allclose(result, twt + 5.0)

    def test_interpolates_between_two_points(self):
        twt = np.zeros(5)
        depth = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = apply_stretch_squeeze(depth, twt, [(0.0, 0.0), (100.0, 10.0)])
        np.testing.assert_allclose(result, [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_holds_constant_outside_control_range(self):
        twt = np.zeros(3)
        depth = np.array([0.0, 500.0, 1000.0])
        result = apply_stretch_squeeze(depth, twt, [(200.0, 3.0), (800.0, 3.0)])
        np.testing.assert_allclose(result, [3.0, 3.0, 3.0])


class TestWaveletSpectra:
    def test_ricker_spectrum_peaks_near_dominant_frequency(self):
        _, wavelet = ricker_wavelet(freq_hz=30.0, dt_s=0.002, length_s=0.256)
        spectra = wavelet_spectra(wavelet, dt_ms=2.0)
        peak_idx = int(np.argmax(spectra["amplitude"]))
        peak_freq = spectra["freq_hz"][peak_idx]
        assert peak_freq == pytest.approx(30.0, abs=10.0)  # coarse -- short wavelet, coarse freq bins

    def test_amplitude_and_phase_same_length_as_freq(self):
        _, wavelet = ricker_wavelet(freq_hz=25.0, dt_s=0.002, length_s=0.128)
        spectra = wavelet_spectra(wavelet, dt_ms=2.0)
        assert len(spectra["freq_hz"]) == len(spectra["amplitude"]) == len(spectra["phase_deg"])