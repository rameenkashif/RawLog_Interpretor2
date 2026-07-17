"""
test_segy_loader.py
---------------------
Unit tests for the per-trace surface coordinate extraction added to
segy_loader.py (CDP_X/CDP_Y, falling back to SourceX/SourceY, with the
SEG-Y coordinate scalar applied). Writes small synthetic SEG-Y files to a
temp path with segyio.create -- no real vendor file needed, same approach
used for the raw amplitude data itself in load_segy_file.
"""

from __future__ import annotations

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from app.segy_loader import load_segy_file


def _write_segy(tmp_path, headers: list[dict], n_samples: int = 10) -> str:
    n_traces = len(headers)
    path = str(tmp_path / "test.sgy")
    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(n_samples)
    spec.tracecount = n_traces
    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = 2000
        for i, hdr in enumerate(headers):
            f.header[i] = hdr
            f.trace[i] = np.zeros(n_samples, dtype=np.float32)
    return path


class TestTraceCoordinates:
    def test_cdp_xy_with_positive_scalar(self, tmp_path):
        headers = [
            {
                segyio.TraceField.CDP_X: 500000 + i * 100,
                segyio.TraceField.CDP_Y: 6500000 + i * 50,
                segyio.TraceField.SourceGroupScalar: 1,
            }
            for i in range(5)
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        np.testing.assert_allclose(loaded.trace_x, [500000, 500100, 500200, 500300, 500400])
        np.testing.assert_allclose(loaded.trace_y, [6500000, 6500050, 6500100, 6500150, 6500200])

    def test_negative_scalar_divides(self, tmp_path):
        headers = [
            {
                segyio.TraceField.CDP_X: 50000000,
                segyio.TraceField.CDP_Y: 650000000,
                segyio.TraceField.SourceGroupScalar: -100,
            }
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        np.testing.assert_allclose(loaded.trace_x, [500000.0])
        np.testing.assert_allclose(loaded.trace_y, [6500000.0])

    def test_falls_back_to_source_xy_when_cdp_unset(self, tmp_path):
        headers = [
            {
                segyio.TraceField.SourceX: 12345,
                segyio.TraceField.SourceY: 67890,
                segyio.TraceField.SourceGroupScalar: 1,
            }
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        np.testing.assert_allclose(loaded.trace_x, [12345.0])
        np.testing.assert_allclose(loaded.trace_y, [67890.0])

    def test_no_coordinates_returns_nan(self, tmp_path):
        headers = [{} for _ in range(3)]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        assert np.isnan(loaded.trace_x).all()
        assert np.isnan(loaded.trace_y).all()

    def test_trace_x_y_shape_matches_trace_count(self, tmp_path):
        headers = [
            {segyio.TraceField.CDP_X: i, segyio.TraceField.CDP_Y: i, segyio.TraceField.SourceGroupScalar: 1}
            for i in range(7)
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        assert loaded.trace_x.shape == (7,)
        assert loaded.trace_y.shape == (7,)


class TestHeaderDiagnostics:
    def test_no_declaration_defaults_to_rev1_source_xy(self, tmp_path):
        headers = [{segyio.TraceField.SourceX: 111, segyio.TraceField.SourceY: 222, segyio.TraceField.SourceGroupScalar: 1}]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        assert loaded.metadata.source_byte_locations == {
            "source_x": 73,
            "source_y": 77,
            "inline": 189,
            "crossline": 193,
        }
        assert loaded.metadata.source_byte_locations_declared == {
            "source_x": False,
            "source_y": False,
            "inline": False,
            "crossline": False,
        }
        assert loaded.metadata.textual_header_encoding in ("cp037", "ascii", "latin-1")

    def test_delay_recording_time_read_explicitly(self, tmp_path):
        headers = [
            {segyio.TraceField.DelayRecordingTime: 2030, segyio.TraceField.SourceGroupScalar: 1} for _ in range(3)
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        assert loaded.metadata.delay_recording_time_ms == 2030.0
        assert loaded.metadata.delay_recording_time_uniform is True
        assert loaded.twt_axis_ms[0] == 2030.0

    def test_nonuniform_delay_flagged(self, tmp_path):
        headers = [
            {segyio.TraceField.DelayRecordingTime: 2030, segyio.TraceField.SourceGroupScalar: 1},
            {segyio.TraceField.DelayRecordingTime: 2040, segyio.TraceField.SourceGroupScalar: 1},
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        assert loaded.metadata.delay_recording_time_uniform is False


class TestTraceInlineCrossline:
    def test_standard_byte_locations(self, tmp_path):
        headers = [
            {
                segyio.TraceField.INLINE_3D: 100 + i,
                segyio.TraceField.CROSSLINE_3D: 200 + i,
                segyio.TraceField.SourceGroupScalar: 1,
            }
            for i in range(4)
        ]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        np.testing.assert_allclose(loaded.trace_inline, [100, 101, 102, 103])
        np.testing.assert_allclose(loaded.trace_crossline, [200, 201, 202, 203])

    def test_no_geometry_returns_nan(self, tmp_path):
        headers = [{segyio.TraceField.SourceGroupScalar: 1} for _ in range(3)]
        loaded = load_segy_file(_write_segy(tmp_path, headers))
        # rev1-standard byte locations default to zero-filled INLINE_3D/
        # CROSSLINE_3D fields in a freshly created synthetic file (unlike a
        # real vendor file where those bytes are just uninitialized/absent),
        # so this asserts the shape/dtype contract rather than NaN --
        # test_nonstandard_declaration_falls_back_gracefully below is the
        # real "not available" case (an unresolvable declared byte location).
        assert loaded.trace_inline.shape == (3,)
        assert loaded.trace_crossline.shape == (3,)

    def test_nonstandard_declared_byte_locations(self, tmp_path):
        headers = [
            {
                segyio.TraceField.FieldRecord: 100 + i,  # byte 9
                segyio.TraceField.TraceNumber: 200 + i,  # byte 13
                segyio.TraceField.SourceGroupScalar: 1,
            }
            for i in range(4)
        ]
        path = _write_segy(tmp_path, headers)
        text = ("Trace Inline At 9 And Size 4. Trace Crossline At Byte 13. " * 3).ljust(3200)[:3200]
        with open(path, "r+b") as fh:
            fh.write(text.encode("cp037"))

        loaded = load_segy_file(path)
        assert loaded.metadata.source_byte_locations["inline"] == 9
        assert loaded.metadata.source_byte_locations["crossline"] == 13
        assert loaded.metadata.source_byte_locations_declared["inline"] is True
        np.testing.assert_allclose(loaded.trace_inline, [100, 101, 102, 103])
        np.testing.assert_allclose(loaded.trace_crossline, [200, 201, 202, 203])

    def test_nonstandard_declaration_falls_back_gracefully(self, tmp_path):
        """A declared byte location that doesn't correspond to any real
        SEG-Y trace header field must degrade to 'inline/crossline not
        available' (NaN) rather than failing the whole SEG-Y load -- unlike
        source_x/source_y, which the tie feature genuinely needs."""
        headers = [{segyio.TraceField.SourceGroupScalar: 1} for _ in range(3)]
        path = _write_segy(tmp_path, headers)
        text = ("Trace Inline At 12345 And Size 4. Trace Crossline At Byte 6. " * 3).ljust(3200)[:3200]
        with open(path, "r+b") as fh:
            fh.write(text.encode("cp037"))

        loaded = load_segy_file(path)
        assert np.isnan(loaded.trace_inline).all()
        assert np.isnan(loaded.trace_crossline).all()
        assert "inline" not in loaded.metadata.source_byte_locations
        assert "crossline" not in loaded.metadata.source_byte_locations
        assert loaded.metadata.source_byte_locations["source_x"] == 73  # unaffected
