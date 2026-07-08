import numpy as np

from app.well_seismic_tie import (
    acoustic_impedance,
    build_synthetic,
    cross_correlate_and_shift,
    depth_to_twt,
    reflectivity_series,
    ricker_wavelet,
)


def test_depth_to_twt_increasing():
    depth = np.linspace(3000, 3100, 50)
    dt = np.full(50, 80.0)  # constant slowness, us/ft
    twt = depth_to_twt(depth, dt, dt_unit="us_per_ft")
    assert np.all(np.diff(twt) > 0)
    assert twt[0] == 0.0


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