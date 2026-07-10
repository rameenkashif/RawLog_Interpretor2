"""
test_segy_header_parser.py
------------------------------
Tests for app/segy_header_parser.py: textual header encoding detection
(ASCII vs EBCDIC/cp037) and vendor-declared trace header byte-location
parsing (see fixes #1/#2 in the calibration audit).
"""

from __future__ import annotations

import pytest

segyio = pytest.importorskip("segyio")

from app.segy_header_parser import (
    STANDARD_BYTE_LOCATIONS,
    decode_textual_header,
    parse_byte_locations,
    resolve_trace_fields,
)


def _padded(text: str) -> bytes:
    return text.encode("ascii").ljust(3200, b" ")[:3200]


class TestDecodeTextualHeader:
    def test_plain_ascii_header_decodes_as_ascii(self):
        raw = _padded("C 1 CLIENT LMKR SURVEY ZAMZAMA TRACE INLINE AT 9 AND SIZE 4")
        result = decode_textual_header(raw)
        assert result.encoding == "ascii"
        assert result.printable_fraction == pytest.approx(1.0)
        assert "ZAMZAMA" in result.text

    def test_ebcdic_header_decodes_as_cp037(self):
        ascii_text = "C 1 CLIENT LMKR SURVEY ZAMZAMA".ljust(3200)
        raw = ascii_text.encode("cp037")
        result = decode_textual_header(raw)
        assert result.encoding == "cp037"
        assert "ZAMZAMA" in result.text

    def test_ebcdic_bytes_decoded_as_ascii_would_be_mostly_unprintable(self):
        # Sanity check for the bug this fix addresses: blindly assuming
        # ascii/cp037 the wrong way round garbles the header. Confirm the
        # "wrong" decode of real EBCDIC bytes scores much lower than the
        # correct one, which is exactly why picking by printable fraction
        # matters instead of hardcoding one encoding.
        raw = "C 1 CLIENT LMKR SURVEY ZAMZAMA".ljust(3200).encode("cp037")
        wrong = raw.decode("ascii", errors="replace")
        correct = raw.decode("cp037", errors="replace")
        from app.segy_header_parser import _printable_fraction

        assert _printable_fraction(correct) > _printable_fraction(wrong)

    def test_never_raises_on_garbage_bytes(self):
        raw = bytes(range(256)) * 12 + bytes(128)  # 3200 arbitrary bytes
        result = decode_textual_header(raw[:3200])
        assert result.encoding in ("cp037", "ascii", "latin-1")


class TestParseByteLocations:
    def test_declared_inline_crossline_override_defaults(self):
        text = "C 1 TRACE INLINE AT 9 AND SIZE 4  TRACE CROSSLINE AT 13 AND SIZE 4"
        result = parse_byte_locations(text)
        assert result.byte_locations["inline"] == 9
        assert result.byte_locations["crossline"] == 13
        assert result.declared["inline"] is True
        assert result.declared["crossline"] is True
        # source_x/source_y not mentioned -> defaulted to rev1 standard.
        assert result.byte_locations["source_x"] == STANDARD_BYTE_LOCATIONS["source_x"]
        assert result.byte_locations["source_y"] == STANDARD_BYTE_LOCATIONS["source_y"]
        assert result.declared["source_x"] is False
        assert result.declared["source_y"] is False

    def test_no_declarations_falls_back_to_rev1_standard(self):
        text = "C 1 THIS HEADER SAYS NOTHING ABOUT BYTE LOCATIONS"
        result = parse_byte_locations(text)
        assert result.byte_locations == STANDARD_BYTE_LOCATIONS
        assert all(v is False for v in result.declared.values())

    def test_source_x_y_declarations_parsed(self):
        text = "SOURCE X AT BYTE 73 AND SOURCE Y AT BYTE 77"
        result = parse_byte_locations(text)
        assert result.byte_locations["source_x"] == 73
        assert result.byte_locations["source_y"] == 77
        assert result.declared["source_x"] is True
        assert result.declared["source_y"] is True


class TestResolveTraceFields:
    def test_resolves_known_byte_offsets(self):
        resolved = resolve_trace_fields({"inline": 9, "crossline": 13, "source_x": 73, "source_y": 77})
        assert resolved["inline"] == segyio.TraceField.FieldRecord
        assert resolved["crossline"] == segyio.TraceField.TraceNumber
        assert resolved["source_x"] == segyio.TraceField.SourceX
        assert resolved["source_y"] == segyio.TraceField.SourceY

    def test_resolves_rev1_standard_offsets(self):
        resolved = resolve_trace_fields(STANDARD_BYTE_LOCATIONS)
        assert resolved["inline"] == segyio.TraceField.INLINE_3D
        assert resolved["crossline"] == segyio.TraceField.CROSSLINE_3D

    def test_unknown_byte_offset_raises(self):
        with pytest.raises(ValueError):
            resolve_trace_fields({"inline": 999999})
