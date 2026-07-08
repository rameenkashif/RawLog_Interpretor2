"""
test_seismic_attributes.py
----------------------------
Unit tests for app/seismic_attributes.py, using synthetic trace data (no
real SEG-Y file required -- segy_loader.py is exercised separately via the
API/manual testing since segyio needs a real file on disk).

Run with:
    cd backend
    pytest tests/test_seismic_attributes.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from app import seismic_attributes as sa
from app.config_loader import get_seismic_config


@pytest.fixture
def config():
    return get_seismic_config(None)


@pytest.fixture
def synthetic_traces():
    """20 traces x 200 samples. Traces 0-9 are low-amplitude "background"
    traces; traces 10-19 are high-amplitude "bright spot" traces, so tests
    can check that attributes correctly distinguish the two groups.
    """
    rng = np.random.default_rng(7)
    n_samples = 200
    t = np.arange(n_samples)

    background = 0.2 * np.sin(2 * np.pi * 0.05 * t)
    bright = 2.0 * np.sin(2 * np.pi * 0.05 * t)

    traces = np.zeros((20, n_samples))
    for i in range(10):
        traces[i] = background + rng.normal(0, 0.02, n_samples)
    for i in range(10, 20):
        traces[i] = bright + rng.normal(0, 0.02, n_samples)

    return traces


class TestRmsAmplitude:
    def test_bright_traces_have_higher_rms(self, synthetic_traces):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        assert rms[10:].mean() > rms[:10].mean()

    def test_shape_matches_trace_count(self, synthetic_traces):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        assert rms.shape == (20,)


class TestEnvelope:
    def test_envelope_shape_matches_traces(self, synthetic_traces):
        env = sa.compute_envelope(synthetic_traces)
        assert env.shape == synthetic_traces.shape

    def test_envelope_non_negative(self, synthetic_traces):
        env = sa.compute_envelope(synthetic_traces)
        assert (env >= 0).all()

    def test_average_envelope_higher_for_bright_traces(self, synthetic_traces):
        avg_env = sa.compute_average_envelope(synthetic_traces)
        assert avg_env[10:].mean() > avg_env[:10].mean()


class TestDominantFrequency:
    def test_returns_one_value_per_trace(self, synthetic_traces):
        freqs = sa.compute_dominant_frequency(synthetic_traces, sample_interval_ms=2.0)
        assert freqs.shape == (20,)

    def test_frequencies_are_non_negative(self, synthetic_traces):
        freqs = sa.compute_dominant_frequency(synthetic_traces, sample_interval_ms=2.0)
        assert (freqs >= 0).all()


class TestSeismicProxies:
    def test_vsh_proxy_bounded_0_1(self, synthetic_traces, config):
        avg_env = sa.compute_average_envelope(synthetic_traces)
        vsh_proxy = sa.compute_vsh_seismic_proxy(avg_env, config)
        assert np.nanmin(vsh_proxy) >= 0.0
        assert np.nanmax(vsh_proxy) <= 1.0

    def test_vsh_proxy_disabled_returns_nan(self, synthetic_traces, config):
        cfg = {**config, "vsh_proxy": {"enabled": False}}
        avg_env = sa.compute_average_envelope(synthetic_traces)
        vsh_proxy = sa.compute_vsh_seismic_proxy(avg_env, cfg)
        assert np.isnan(vsh_proxy).all()

    def test_phie_proxy_bounded_0_1(self, synthetic_traces, config):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        phie_proxy = sa.compute_phie_seismic_proxy(rms, config)
        assert np.nanmin(phie_proxy) >= 0.0
        assert np.nanmax(phie_proxy) <= 1.0

    def test_phie_proxy_inversely_related_to_amplitude(self, synthetic_traces, config):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        phie_proxy = sa.compute_phie_seismic_proxy(rms, config)
        # Bright (high-amplitude) traces should get a LOWER phie proxy
        # since it's defined as 1 - normalized_amplitude.
        assert phie_proxy[10:].mean() < phie_proxy[:10].mean()

    def test_swe_proxy_bounded_0_1(self, synthetic_traces, config):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        avg_env = sa.compute_average_envelope(synthetic_traces)
        swe_proxy = sa.compute_swe_seismic_proxy(rms, avg_env, config)
        assert np.nanmin(swe_proxy) >= 0.0
        assert np.nanmax(swe_proxy) <= 1.0

    def test_bright_spot_traces_get_lower_swe_proxy(self, synthetic_traces, config):
        rms = sa.compute_rms_amplitude(synthetic_traces)
        avg_env = sa.compute_average_envelope(synthetic_traces)
        swe_proxy = sa.compute_swe_seismic_proxy(rms, avg_env, config)
        # Bright traces (10-19) should be flagged as more hydrocarbon-like
        # (lower "water saturation" proxy) than the background traces.
        assert swe_proxy[10:].mean() < swe_proxy[:10].mean()


class TestFullSeismicInterpretation:
    def test_all_expected_columns_present(self, synthetic_traces, config):
        df = sa.run_seismic_interpretation(
            synthetic_traces, sample_interval_ms=2.0, config=config
        )
        expected_cols = {
            "TRACE_INDEX",
            "RMS_AMPLITUDE",
            "AVG_ENVELOPE",
            "DOMINANT_FREQ_HZ",
            "VSH_SEISMIC_PROXY",
            "PHIE_SEISMIC_PROXY",
            "SWE_SEISMIC_PROXY",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_trace_count(self, synthetic_traces, config):
        df = sa.run_seismic_interpretation(
            synthetic_traces, sample_interval_ms=2.0, config=config
        )
        assert len(df) == synthetic_traces.shape[0]

    def test_no_unexpected_nans(self, synthetic_traces, config):
        df = sa.run_seismic_interpretation(
            synthetic_traces, sample_interval_ms=2.0, config=config
        )
        for col in ["RMS_AMPLITUDE", "AVG_ENVELOPE", "DOMINANT_FREQ_HZ"]:
            assert not df[col].isna().any(), f"{col} has unexpected NaNs"
