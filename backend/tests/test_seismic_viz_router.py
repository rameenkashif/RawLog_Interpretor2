"""
test_seismic_viz_router.py
-----------------------------
HTTP-level tests for routers/seismic_viz.py (/api/seismic/*), using
FastAPI's TestClient against a small synthetic SEG-Y file (see
test_seismic_processor.py for why the file is synthetic and how its
header layout mirrors the real one) so the routing/response-model/
error-mapping layer is exercised, not just the SegyVolume class directly.
"""

from __future__ import annotations

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from fastapi.testclient import TestClient

from app.services import seismic_processor as sp
from tests.test_seismic_processor import _write_synthetic_segy


@pytest.fixture
def client(tmp_path, monkeypatch):
    _write_synthetic_segy(tmp_path / "survey.sgy")
    monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
    sp._volume_cache.clear()

    import main  # imported here so the monkeypatched RAW_SEISMIC_DIR is in effect before any request

    with TestClient(main.app) as c:
        yield c

    sp._volume_cache.clear()


@pytest.fixture
def wide_client(tmp_path, monkeypatch):
    """80 samples instead of the base `client` fixture's 6, so STFT's
    32-sample window has something meaningful to slide across."""
    _write_synthetic_segy(tmp_path / "wide_survey.sgy", n_samples=80, interval_ms=2)
    monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
    sp._volume_cache.clear()

    import main

    with TestClient(main.app) as c:
        yield c

    sp._volume_cache.clear()


class TestSurveyInfo:
    def test_returns_geometry(self, client):
        resp = client.get("/api/seismic/survey-info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_traces"] == 20
        assert body["inline_min"] == 382
        assert body["crossline_max"] == 49


class TestSections:
    def test_inline_section_ok(self, client):
        resp = client.get("/api/seismic/inline/384")
        assert resp.status_code == 200
        body = resp.json()
        assert body["inline_number"] == 384
        assert len(body["crossline_axis"]) == 4

    def test_inline_section_out_of_range_is_422(self, client):
        resp = client.get("/api/seismic/inline/9999")
        assert resp.status_code == 422

    def test_crossline_section_ok(self, client):
        resp = client.get("/api/seismic/crossline/47")
        assert resp.status_code == 200
        assert resp.json()["crossline_number"] == 47


class TestTimeSlice:
    def test_boundary_clamp_via_http(self, client):
        resp = client.get("/api/seismic/timeslice", params={"time_ms": 2029})
        assert resp.status_code == 200
        body = resp.json()
        assert body["time_ms"] == 2030.0
        assert body["requested_time_ms"] == 2029.0


class TestSpectrum:
    def test_whole_volume(self, client):
        resp = client.get("/api/seismic/spectrum")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_traces_sampled"] == 20

    def test_bad_inline_is_422(self, client):
        resp = client.get("/api/seismic/spectrum", params={"inline_number": 9999})
        assert resp.status_code == 422


class TestSpectralDecompEndpoints:
    def test_full_inline_decomposition_stft(self, wide_client):
        resp = wide_client.get("/api/seismic/spectral-decomp/inline/384", params={"method": "stft"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["method"] == "stft"
        assert max(body["freq_hz"]) <= body["nyquist_hz"] + 1e-9
        assert len(body["energy"]) == len(body["time_ms"])
        assert len(body["energy"][0]) == len(body["freq_hz"])
        assert len(body["energy"][0][0]) == len(body["crossline_axis"])

    def test_full_inline_decomposition_cwt(self, wide_client):
        resp = wide_client.get("/api/seismic/spectral-decomp/inline/384", params={"method": "cwt"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["method"] == "cwt"
        assert max(body["freq_hz"]) <= body["nyquist_hz"] + 1e-9

    def test_frequency_slice_matches_inline_section_shape(self, wide_client):
        section = wide_client.get("/api/seismic/inline/384").json()
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/inline/384",
            params={"method": "stft", "frequency_hz": 50},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "amplitude" in body and "energy" not in body  # slice shape, not the full volume
        assert body["crossline_axis"] == section["crossline_axis"]
        n_pos = len(section["crossline_axis"])
        assert all(len(row) == n_pos for row in body["amplitude"])

    def test_trace_decomposition(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/trace",
            params={"inline_number": 384, "crossline_number": 47, "method": "cwt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert max(body["freq_hz"]) <= body["nyquist_hz"] + 1e-9
        assert len(body["energy"]) == len(body["time_ms"])
        assert len(body["energy"][0]) == len(body["freq_hz"])

    def test_bad_inline_is_422(self, wide_client):
        resp = wide_client.get("/api/seismic/spectral-decomp/inline/9999", params={"method": "stft"})
        assert resp.status_code == 422

    def test_bad_method_is_422(self, wide_client):
        resp = wide_client.get("/api/seismic/spectral-decomp/inline/384", params={"method": "bogus"})
        assert resp.status_code == 422

    def test_unknown_trace_is_422(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/trace",
            params={"inline_number": 384, "crossline_number": 9999, "method": "stft"},
        )
        assert resp.status_code == 422

    def test_swt_inline_default_level_and_wavelet(self, wide_client):
        resp = wide_client.get("/api/seismic/spectral-decomp/inline/384", params={"method": "swt"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["method"] == "swt"
        assert body["level"] == 3
        assert body["wavelet"] == "sym8"
        assert "amplitude" in body and "energy" not in body
        assert len(body["band_hz"]) == 2

    def test_swt_inline_custom_level_and_wavelet(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/inline/384",
            params={"method": "swt", "level": 5, "wavelet": "coif3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["level"] == 5
        assert body["wavelet"] == "coif3"

    def test_swt_amplitude_matches_inline_section_shape(self, wide_client):
        section = wide_client.get("/api/seismic/inline/384").json()
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/inline/384", params={"method": "swt", "level": 2}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["crossline_axis"] == section["crossline_axis"]
        n_pos = len(section["crossline_axis"])
        assert all(len(row) == n_pos for row in body["amplitude"])

    def test_swt_bad_level_is_422(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/inline/384", params={"method": "swt", "level": 9}
        )
        assert resp.status_code == 422

    def test_swt_bad_wavelet_is_422(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/inline/384", params={"method": "swt", "wavelet": "bogus"}
        )
        assert resp.status_code == 422

    def test_swt_trace_decomposition_returns_all_levels(self, wide_client):
        resp = wide_client.get(
            "/api/seismic/spectral-decomp/trace",
            params={"inline_number": 384, "crossline_number": 47, "method": "swt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["levels"] == [1, 2, 3, 4, 5, 6]
        assert len(body["bands_hz"]) == 6
        assert len(body["energy"]) == len(body["time_ms"])
        assert all(len(row) == 6 for row in body["energy"])


class TestWellTieEndpoint:
    def test_unknown_well_is_404(self, client):
        resp = client.get("/api/seismic/well-tie/DOES_NOT_EXIST")
        assert resp.status_code == 404

    def test_no_segy_file_is_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp, "RAW_SEISMIC_DIR", tmp_path)
        sp._volume_cache.clear()
        import main

        with TestClient(main.app) as c:
            resp = c.get("/api/seismic/survey-info")
        assert resp.status_code == 404
        sp._volume_cache.clear()
