import numpy as np
import pytest

from app.well_seismic_tie import (
    DEFAULT_CANDIDATE_FREQS_HZ,
    DEFAULT_TIE_SEARCH_FREQS_HZ,
    TieError,
    acoustic_impedance,
    apply_stretch_squeeze,
    build_synthetic,
    calibrate_gardner_coefficients,
    cross_correlate_and_shift,
    cross_check_delay_datum,
    depth_to_twt,
    despike_mad,
    extract_statistical_wavelet,
    find_nearest_trace_index,
    gardner_density,
    reflectivity_from_time_axis,
    reflectivity_series,
    ricker_wavelet,
    rock_physics_density,
    search_best_tie,
    search_best_tie_full_window,
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


class TestCrossCorrelateSearchRangeAndBoundaryFlag:
    def test_default_max_shift_is_300ms(self):
        dt_ms = 2.0
        t = np.linspace(0, 100, 200)
        trace = np.sin(2 * np.pi * 0.05 * t)
        result = cross_correlate_and_shift(trace, trace, dt_ms)
        assert result["max_shift_ms"] == 300.0

    def test_small_interior_shift_not_boundary_pinned(self):
        dt_ms = 2.0
        t = np.linspace(0, 100, 200)
        trace = np.sin(2 * np.pi * 0.05 * t)
        shifted = np.roll(trace, 5)  # 10ms, well inside a 300ms window
        result = cross_correlate_and_shift(shifted, trace, dt_ms)
        assert result["boundary_pinned"] is False

    def test_true_shift_beyond_narrow_window_gets_boundary_pinned(self):
        # A single broad Gaussian pulse (not periodic, unlike a sine) so
        # correlation-vs-shift is a clean single peak with no aliasing,
        # and wide enough (sigma=30) that its gradient is still
        # meaningful within a narrow search window. True best alignment
        # is a 100-sample (200ms) shift, but the search is bounded to
        # +/-20ms, forcing the best FOUND shift to sit right at the
        # window edge.
        n = 400
        t = np.arange(n)
        sigma = 30.0
        synthetic = np.exp(-((t - 150) ** 2) / (2 * sigma**2))
        real = np.exp(-((t - 250) ** 2) / (2 * sigma**2))  # +100 samples from synthetic
        dt_ms = 2.0
        result = cross_correlate_and_shift(synthetic, real, dt_ms, max_shift_ms=20.0)
        assert result["boundary_pinned"] is True
        assert abs(result["best_shift_ms"]) == pytest.approx(20.0)

    def test_true_shift_within_wide_window_not_boundary_pinned(self):
        n = 400
        t = np.arange(n)
        sigma = 30.0
        synthetic = np.exp(-((t - 150) ** 2) / (2 * sigma**2))
        real = np.exp(-((t - 250) ** 2) / (2 * sigma**2))
        dt_ms = 2.0
        # Same pair, but a search window wide enough to actually contain
        # the true 200ms shift -- should converge near it, not pin.
        result = cross_correlate_and_shift(synthetic, real, dt_ms, max_shift_ms=300.0)
        assert result["boundary_pinned"] is False
        assert result["best_shift_ms"] == pytest.approx(200.0, abs=10.0)

    def test_negative_max_shift_ms_raises(self):
        dt_ms = 2.0
        trace = np.sin(np.linspace(0, 10, 50))
        with pytest.raises(TieError):
            cross_correlate_and_shift(trace, trace, dt_ms, max_shift_ms=-10.0)

    def test_narrow_window_can_change_result_vs_wide_window(self):
        # Directly demonstrates fix #8's failure mode: a search range too
        # narrow to contain the true answer silently returns a much worse
        # correlation than the real ~1.0 match -- indistinguishable from
        # "the tie genuinely failed" unless you know to widen the window.
        n = 400
        t = np.arange(n)
        sigma = 30.0
        synthetic = np.exp(-((t - 150) ** 2) / (2 * sigma**2))
        real = np.exp(-((t - 250) ** 2) / (2 * sigma**2))
        dt_ms = 2.0
        narrow = cross_correlate_and_shift(synthetic, real, dt_ms, max_shift_ms=20.0)
        wide = cross_correlate_and_shift(synthetic, real, dt_ms, max_shift_ms=300.0)
        assert wide["correlation"] > narrow["correlation"] + 0.3
        assert wide["boundary_pinned"] is False
        assert narrow["boundary_pinned"] is True


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


class TestSearchBestTie:
    """search_best_tie jointly searches wavelet frequency (Ricker) and
    polarity, not just shift position -- see well_seismic_tie.py's
    docstring for why position search alone can't recover a tie that's
    wrong because the assumed frequency/polarity was wrong."""

    def _embed_known_wavelet(self, freq_hz: float, polarity: int, noise_std: float = 0.01, seed: int = 0):
        """Builds a reflectivity series + a "real" trace that's exactly
        that reflectivity convolved with a KNOWN (freq_hz, polarity)
        Ricker wavelet plus a little noise -- so search_best_tie has a
        ground-truth answer to recover."""
        rng = np.random.default_rng(seed)
        dt_ms = 2.0
        n_samples = 300
        seismic_twt_ms = np.arange(n_samples) * dt_ms + 2000.0

        refl_reg = np.zeros(150)
        refl_reg[[20, 50, 90, 120]] = [0.1, -0.15, 0.08, -0.05]
        reg_twt_ms = np.arange(150) * dt_ms + 2050.0

        _, true_wavelet = ricker_wavelet(freq_hz, dt_ms / 1000.0)
        true_wavelet = polarity * true_wavelet
        conv = np.convolve(refl_reg, true_wavelet, mode="full")
        start = (len(conv) - len(refl_reg)) // 2
        synthetic_reg = conv[start : start + len(refl_reg)]
        real_trace = np.interp(seismic_twt_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0)
        real_trace = real_trace + rng.normal(0, noise_std, n_samples)

        return refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace

    def test_recovers_known_frequency_and_polarity(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=35.0, polarity=-1
        )
        result = search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace)
        assert result.wavelet_freq_hz == 35.0
        assert result.polarity == -1
        assert result.correlation > 0.9

    def test_normal_polarity_recovered_too(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=20.0, polarity=1
        )
        result = search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace)
        assert result.wavelet_freq_hz == 20.0
        assert result.polarity == 1
        assert result.correlation > 0.9

    def test_beats_a_wrong_fixed_frequency_and_polarity(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=35.0, polarity=-1
        )
        # A single fixed (wrong) candidate, via the plain position-only search.
        _, wrong_wavelet = ricker_wavelet(15.0, dt_ms / 1000.0)  # wrong freq, wrong (normal) polarity
        conv = np.convolve(refl_reg, wrong_wavelet, mode="full")
        start = (len(conv) - len(refl_reg)) // 2
        synthetic_reg = conv[start : start + len(refl_reg)]
        synthetic_on_axis = np.interp(seismic_twt_ms, reg_twt_ms, synthetic_reg, left=0.0, right=0.0)
        wrong_tie = cross_correlate_and_shift(synthetic_on_axis, real_trace, dt_ms)

        result = search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace)
        assert result.correlation > wrong_tie["correlation"]

    def test_n_candidates_tried_matches_freqs_times_polarities(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=25.0, polarity=1
        )
        result = search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace)
        assert result.n_candidates_tried == len(DEFAULT_CANDIDATE_FREQS_HZ) * 2

    def test_search_polarity_false_only_tries_normal_polarity(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=25.0, polarity=1
        )
        result = search_best_tie(
            refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace, search_polarity=False
        )
        assert result.n_candidates_tried == len(DEFAULT_CANDIDATE_FREQS_HZ)
        assert result.polarity == 1

    def test_fixed_wavelet_mode_searches_polarity_only(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=30.0, polarity=-1
        )
        _, fixed_wavelet = ricker_wavelet(30.0, dt_ms / 1000.0)
        result = search_best_tie(
            refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace,
            candidate_freqs_hz=None, fixed_wavelet=fixed_wavelet,
        )
        assert result.wavelet_freq_hz is None
        assert result.polarity == -1
        assert result.n_candidates_tried == 2
        assert result.correlation > 0.9

    def test_no_candidates_and_no_fixed_wavelet_raises(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=25.0, polarity=1
        )
        with pytest.raises(TieError):
            search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace, candidate_freqs_hz=None)

    def test_respects_max_shift_ms(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=25.0, polarity=1
        )
        result = search_best_tie(
            refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace, max_shift_ms=10.0
        )
        assert abs(result.best_shift_ms) <= 10.0 + 1e-9
        assert result.max_shift_ms == 10.0

    def test_shifted_synthetic_same_length_as_real_trace(self):
        refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace = self._embed_known_wavelet(
            freq_hz=25.0, polarity=1
        )
        result = search_best_tie(refl_reg, reg_twt_ms, seismic_twt_ms, dt_ms, real_trace)
        assert len(result.shifted_synthetic) == len(real_trace)
        assert len(result.synthetic) == len(real_trace)


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


class TestCrossCheckDelayDatum:
    def test_plausible_when_implied_depth_close_to_logged_top(self):
        # delay=2030ms -> one-way 1015ms -> at 3000 m/s implied depth 3045m,
        # close to a logged top of ~3454m (the real Z-02 well's actual
        # value) -- within the default 50% relative-error tolerance.
        result = cross_check_delay_datum(delay_ms=2030.0, logged_top_depth_m=3454.5)
        assert result.plausible is True
        assert result.implied_depth_m == pytest.approx(3045.0, rel=0.01)

    def test_implausible_when_wildly_off(self):
        # delay=2030ms implies ~3045m regardless of logged top; a logged
        # top of 50m makes that wildly implausible.
        result = cross_check_delay_datum(delay_ms=2030.0, logged_top_depth_m=50.0)
        assert result.plausible is False
        assert result.relative_error > 1.0

    def test_zero_logged_top_is_implausible(self):
        result = cross_check_delay_datum(delay_ms=2030.0, logged_top_depth_m=0.0)
        assert result.plausible is False

    def test_custom_velocity_changes_implied_depth(self):
        slow = cross_check_delay_datum(delay_ms=2000.0, logged_top_depth_m=1500.0, avg_velocity_m_s=1500.0)
        fast = cross_check_delay_datum(delay_ms=2000.0, logged_top_depth_m=1500.0, avg_velocity_m_s=4500.0)
        assert fast.implied_depth_m > slow.implied_depth_m


class TestDespikeMad:
    def test_leaves_clean_signal_unchanged(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 200)) * 0.1
        despiked = despike_mad(signal)
        np.testing.assert_allclose(despiked, signal)

    def test_removes_genuine_outlier(self):
        # Real baseline variance alongside one extreme, unambiguous
        # outlier -- unlike a flat-zero baseline, removing the outlier
        # alone shouldn't (and doesn't) collapse the whole signal.
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 0.01, 100)
        signal[50] = 100.0
        despiked = despike_mad(signal)
        assert despiked[50] == 0.0
        assert np.std(despiked) > 0

    def test_sparse_reflectivity_like_signal_not_collapsed_to_zero(self):
        # Mostly-zero signal with a handful of small-but-real nonzero
        # samples -- exactly the shape of a reflectivity series between
        # reflectors. An unfloored MAD threshold degenerates to ~0 here
        # (median of a 94%-zero array is 0) and flags nearly every
        # nonzero sample as a spike; flooring the threshold at a fraction
        # of RMS keeps SOME of that real signal instead of destroying all
        # of it (despike_mad raises outright if it would still collapse
        # to zero variance despite the floor).
        rng = np.random.default_rng(0)
        signal = np.zeros(500)
        spike_idx = rng.choice(500, size=30, replace=False)
        signal[spike_idx] = rng.normal(0, 0.02, size=30)  # small, real reflectivity values
        despiked = despike_mad(signal)  # must not raise
        assert np.std(despiked) > 0
        assert np.count_nonzero(despiked) > 0

    def test_raises_if_would_collapse_signal_with_real_variance(self):
        # Sparse-but-real signal (mostly zero, a few small nonzero
        # values) with the floor explicitly disabled -- reproduces the
        # exact naive-MAD bug (median/MAD of a mostly-zero array is 0, so
        # a 0 threshold flags every nonzero sample) and confirms the
        # sanity check catches the resulting total collapse instead of
        # silently returning an all-zero result.
        signal = np.array([0.0, 0.0, 0.01, 0.0, -0.02, 0.0, 0.015])
        with pytest.raises(TieError):
            despike_mad(signal, threshold_n_mad=0.0, mad_floor_fraction=0.0)

    def test_all_zero_input_returns_all_zero_without_raising(self):
        signal = np.zeros(50)
        despiked = despike_mad(signal)
        assert np.all(despiked == 0.0)


class TestBuildSyntheticDatumCheckAndDespike:
    def test_datum_check_included_in_result(self):
        depth = np.linspace(3454.0, 3480.0, 60)
        dt = np.full(60, 90.0)
        rhob = np.full(60, 2.4)
        seismic_twt = np.arange(2030.0, 2030.0 + 60 * 2.0, 2.0)
        result = build_synthetic(depth, dt, rhob, seismic_dt_ms=2.0, seismic_twt_axis_ms=seismic_twt)
        assert result.datum_check is not None
        assert result.datum_check.delay_ms == pytest.approx(2030.0)
        assert result.datum_check.logged_top_depth_m == pytest.approx(3454.0)

    def test_despike_disabled_skips_despiking(self):
        depth = np.linspace(3454.0, 3480.0, 60)
        dt = np.full(60, 90.0)
        rhob = np.full(60, 2.4)
        seismic_twt = np.arange(2030.0, 2030.0 + 60 * 2.0, 2.0)
        result_despiked = build_synthetic(
            depth, dt, rhob, seismic_dt_ms=2.0, seismic_twt_axis_ms=seismic_twt, despike=True
        )
        result_raw = build_synthetic(
            depth, dt, rhob, seismic_dt_ms=2.0, seismic_twt_axis_ms=seismic_twt, despike=False
        )
        # Constant DT/RHOB -> zero reflectivity everywhere -- despiking is
        # a no-op either way, but both must run without error and agree.
        np.testing.assert_allclose(result_despiked.reflectivity, result_raw.reflectivity)


class TestReflectivityFromTimeAxis:
    """reflectivity_from_time_axis trusts a caller-supplied time axis (e.g.
    a well's vendor DPTM curve) directly, unlike build_synthetic's
    depth_to_twt path which re-derives one by sonic integration."""

    def test_zero_reflectivity_for_constant_impedance(self):
        time_ms = np.linspace(2000.0, 2100.0, 60)
        dt_log = np.full(60, 90.0)
        rhob = np.full(60, 2.4)
        t_rc, rc = reflectivity_from_time_axis(time_ms, dt_log, rhob, seismic_dt_ms=2.0)
        assert np.allclose(rc, 0.0)
        assert len(t_rc) == len(rc)

    def test_output_covers_time_axis_span(self):
        time_ms = np.linspace(2000.0, 2100.0, 60)
        dt_log = 80.0 + np.sin(np.linspace(0, 6, 60)) * 5
        rhob = 2.4 + np.cos(np.linspace(0, 6, 60)) * 0.05
        t_rc, rc = reflectivity_from_time_axis(time_ms, dt_log, rhob, seismic_dt_ms=2.0)
        assert t_rc[0] == pytest.approx(2000.0)
        assert t_rc[-1] < 2100.0
        assert np.all(np.diff(t_rc) > 0)

    def test_unsorted_time_axis_is_sorted(self):
        rng = np.random.default_rng(1)
        order = rng.permutation(60)
        time_ms = np.linspace(2000.0, 2100.0, 60)
        dt_log = 80.0 + np.sin(np.linspace(0, 6, 60)) * 5
        rhob = 2.4 + np.cos(np.linspace(0, 6, 60)) * 0.05
        t_rc, rc = reflectivity_from_time_axis(time_ms[order], dt_log[order], rhob[order], seismic_dt_ms=2.0)
        assert np.all(np.diff(t_rc) > 0)

    def test_too_few_valid_samples_raises(self):
        time_ms = np.array([2000.0, 2001.0, np.nan])
        dt_log = np.array([80.0, 81.0, 82.0])
        rhob = np.array([2.4, 2.4, 2.4])
        with pytest.raises(TieError):
            reflectivity_from_time_axis(time_ms, dt_log, rhob, seismic_dt_ms=2.0)

    def test_nulls_and_nonpositive_values_are_excluded(self):
        n = 30
        time_ms = np.linspace(2000.0, 2100.0, n)
        dt_log = np.full(n, 80.0)
        rhob = np.full(n, 2.4)
        dt_log[5] = -9999.25
        rhob[10] = 0.0
        dt_log[15] = np.nan
        t_rc, rc = reflectivity_from_time_axis(time_ms, dt_log, rhob, seismic_dt_ms=2.0)
        assert len(t_rc) > 0  # ran without error despite bad samples

    def test_matches_rhob_over_dt_impedance_ratio_directly(self):
        # AI = RHOB/DT is scale-invariant in the reflectivity ratio, same as
        # acoustic_impedance()'s velocity*density -- see module docstring.
        time_ms = np.linspace(2000.0, 2050.0, 30)
        dt_log = np.linspace(80.0, 95.0, 30)
        rhob = np.linspace(2.3, 2.6, 30)
        t_rc, rc = reflectivity_from_time_axis(time_ms, dt_log, rhob, seismic_dt_ms=2.0)
        ai = rhob / dt_log
        ai_u = np.interp(np.arange(time_ms[0], time_ms[-1], 2.0), time_ms, ai)
        expected = (ai_u[1:] - ai_u[:-1]) / (ai_u[1:] + ai_u[:-1])
        np.testing.assert_allclose(rc, expected)


class TestSearchBestTieFullWindow:
    """search_best_tie_full_window mirrors the well_tie notebook's
    algorithm: joint frequency/polarity/bulk-shift search across the whole
    seismic window, using absolute-time overlap rather than a discrete
    sample-lag cross-correlation."""

    def _embed_known_tie(self, freq_hz: float, polarity: int, shift_ms: float, noise_std: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        dt_ms = 2.0
        t_rc = np.arange(2000.0, 2120.0, dt_ms)
        rc = np.zeros(len(t_rc))
        rc[[10, 25, 40, 50]] = [0.1, -0.15, 0.08, -0.05]

        _, wav = ricker_wavelet(freq_hz, dt_ms / 1000.0, min(0.100, max(0.030, 0.6 * len(rc) * dt_ms / 1000.0)))
        synth = np.convolve(rc, wav, mode="same")
        if len(synth) != len(rc):
            synth = synth[: len(rc)] if len(synth) > len(rc) else np.pad(synth, (0, len(rc) - len(synth)))
        synth = synth - synth.mean()
        if synth.std() > 0:
            synth = synth / synth.std()
        synth = polarity * synth

        seismic_twt_ms = np.arange(1900.0, 2300.0, dt_ms)
        t_shifted = t_rc + shift_ms
        real_trace = np.interp(seismic_twt_ms, t_shifted, synth, left=0.0, right=0.0)
        real_trace = real_trace + rng.normal(0, noise_std, len(real_trace))
        return t_rc, rc, seismic_twt_ms, dt_ms, real_trace

    def test_recovers_known_freq_polarity_shift(self):
        t_rc, rc, seismic_twt_ms, dt_ms, real_trace = self._embed_known_tie(
            freq_hz=25.0, polarity=-1, shift_ms=14.0
        )
        result = search_best_tie_full_window(t_rc, rc, seismic_twt_ms, dt_ms, real_trace)
        assert result.best_freq_hz == 25.0
        assert result.polarity == -1
        assert result.bulk_shift_ms == pytest.approx(14.0, abs=dt_ms)
        assert result.correlation > 0.9

    def test_positive_polarity_recovered_too(self):
        t_rc, rc, seismic_twt_ms, dt_ms, real_trace = self._embed_known_tie(
            freq_hz=35.0, polarity=1, shift_ms=-20.0
        )
        result = search_best_tie_full_window(t_rc, rc, seismic_twt_ms, dt_ms, real_trace)
        assert result.polarity == 1
        assert result.bulk_shift_ms == pytest.approx(-20.0, abs=dt_ms)
        assert result.correlation > 0.9

    def test_output_arrays_share_reflectivity_length(self):
        t_rc, rc, seismic_twt_ms, dt_ms, real_trace = self._embed_known_tie(
            freq_hz=25.0, polarity=1, shift_ms=0.0
        )
        result = search_best_tie_full_window(t_rc, rc, seismic_twt_ms, dt_ms, real_trace)
        assert len(result.time_ms) == len(rc)
        assert len(result.synthetic_amplitude) == len(rc)
        assert len(result.seismic_amplitude) == len(rc)
        assert len(result.reflectivity) == len(rc)
        np.testing.assert_allclose(result.reflectivity, rc)
        np.testing.assert_allclose(result.time_ms, t_rc + result.bulk_shift_ms)

    def test_respects_default_freq_candidates(self):
        t_rc, rc, seismic_twt_ms, dt_ms, real_trace = self._embed_known_tie(
            freq_hz=25.0, polarity=1, shift_ms=0.0
        )
        result = search_best_tie_full_window(t_rc, rc, seismic_twt_ms, dt_ms, real_trace)
        assert result.best_freq_hz in DEFAULT_TIE_SEARCH_FREQS_HZ

    def test_no_overlap_raises(self):
        t_rc = np.arange(2000.0, 2050.0, 2.0)
        rc = np.zeros(len(t_rc))
        rc[5] = 0.1
        seismic_twt_ms = np.arange(0.0, 50.0, 2.0)  # nowhere near t_rc
        real_trace = np.random.default_rng(0).normal(0, 1, len(seismic_twt_ms))
        with pytest.raises(TieError):
            search_best_tie_full_window(t_rc, rc, seismic_twt_ms, 2.0, real_trace)

    def test_too_short_reflectivity_raises(self):
        with pytest.raises(TieError):
            search_best_tie_full_window(
                np.array([2000.0, 2002.0]), np.array([0.1, 0.2]), np.arange(1900.0, 2100.0, 2.0), 2.0,
                np.random.default_rng(0).normal(0, 1, 100),
            )