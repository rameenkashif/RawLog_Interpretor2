"""
segy_header_parser.py
------------------------
Vendor SEG-Y exports frequently deviate from the SEG-Y rev1 standard in
ways that silently corrupt downstream parsing if assumed away:

1. TEXTUAL HEADER ENCODING. Rev1 mandates EBCDIC (cp037) for the 3200-byte
   textual header, and segyio's own f.text[] property always decodes
   assuming EBCDIC regardless of what's actually in the file -- confirmed
   here by round-tripping a plain-ASCII header through segyio, which
   re-encodes it as EBCDIC on write/read and gives no way to ask for
   ASCII. Some vendors (LMKR exports seen in this dataset) write plain
   ASCII despite the standard; decoding that as cp037 produces line noise.
   decode_textual_header() reads the RAW 3200 header bytes directly
   (bypassing segyio's text property) and tries both cp037 and
   ascii/latin-1, picking whichever decodes to a higher fraction of
   printable characters -- never hardcode the encoding.

2. TRACE HEADER BYTE LOCATIONS. Rev1 defines standard byte offsets for
   Inline/Crossline/SourceX/SourceY (189/193/73/77), but vendors routinely
   declare their own non-standard locations in the textual header itself
   (e.g. "Trace Inline At 9 And Size 4" -- this dataset's actual layout).
   parse_byte_locations() regex-scans the (correctly decoded) textual
   header for such declarations and falls back to the rev1 standard only
   for whichever fields aren't declared. resolve_trace_fields() then maps
   each byte offset to a segyio.TraceField member -- segyio.TraceField's
   enum VALUES are themselves the 1-indexed byte offsets (e.g.
   segyio.TraceField(9) == TraceField.FieldRecord), so this is a direct,
   data-driven lookup, never a hardcoded byte-offset-to-field table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import segyio

TEXTUAL_HEADER_SIZE = 3200

# SEG-Y rev1 standard byte locations (1-indexed), used only for whichever
# fields aren't explicitly declared in the textual header.
STANDARD_BYTE_LOCATIONS: dict[str, int] = {
    "inline": 189,
    "crossline": 193,
    "source_x": 73,
    "source_y": 77,
}

# Candidate encodings tried for the textual header, in no particular
# preference order -- decode_textual_header() picks whichever scores
# highest on printable-character fraction, not the first in this list.
_CANDIDATE_ENCODINGS = ("cp037", "ascii", "latin-1")

_PRINTABLE_ORDS = set(range(0x20, 0x7F)) | {0x0A, 0x0D, 0x09}


@dataclass
class TextualHeaderResult:
    text: str
    encoding: str
    printable_fraction: float


def _printable_fraction(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ord(ch) in _PRINTABLE_ORDS)
    return printable / len(text)


def decode_textual_header(raw: bytes) -> TextualHeaderResult:
    """Decode a 3200-byte SEG-Y textual header, trying both the rev1-
    mandated EBCDIC (cp037) and plain ASCII/latin-1, and returning
    whichever decodes to a higher fraction of printable characters.
    latin-1 never raises (it maps every byte 1:1), so this always returns
    a result even for a garbled/corrupt header -- callers should check
    printable_fraction if they need to detect that case.
    """
    best: TextualHeaderResult | None = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:  # pragma: no cover - all three are always available
            continue
        fraction = _printable_fraction(text)
        if best is None or fraction > best.printable_fraction:
            best = TextualHeaderResult(text=text, encoding=encoding, printable_fraction=fraction)
    assert best is not None  # latin-1 always succeeds, so this can't happen
    return best


# Vendor byte-location declarations look like "Trace Inline At 9 And Size
# 4", "Crossline Byte Location 13", "Source X At Byte 73" -- deliberately
# permissive about the words between the field name and the number (vendor
# wording varies) but anchored on "at" plus a number, and capped in how far
# it'll scan so it doesn't cross into an unrelated sentence.
_BYTE_LOCATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "inline": re.compile(r"\binline\b[^0-9]{0,40}?\bat\b[^0-9]{0,10}(\d{1,4})", re.IGNORECASE),
    "crossline": re.compile(r"\bcrossline\b[^0-9]{0,40}?\bat\b[^0-9]{0,10}(\d{1,4})", re.IGNORECASE),
    "source_x": re.compile(r"\bsource\s*x\b[^0-9]{0,40}?\bat\b[^0-9]{0,10}(\d{1,4})", re.IGNORECASE),
    "source_y": re.compile(r"\bsource\s*y\b[^0-9]{0,40}?\bat\b[^0-9]{0,10}(\d{1,4})", re.IGNORECASE),
}


@dataclass
class ByteLocationResult:
    byte_locations: dict[str, int]
    declared: dict[str, bool]  # True if regex-matched in the header, False if defaulted to rev1 standard


def parse_byte_locations(text_header: str) -> ByteLocationResult:
    """Regex-scan a decoded textual header for vendor-declared trace-header
    byte locations for inline/crossline/source_x/source_y. Falls back to
    the SEG-Y rev1 standard location for any field not found declared, and
    reports which fields were actually found vs. defaulted so callers can
    surface that provenance rather than silently trusting a guess."""
    byte_locations = dict(STANDARD_BYTE_LOCATIONS)
    declared = {field: False for field in STANDARD_BYTE_LOCATIONS}
    for field, pattern in _BYTE_LOCATION_PATTERNS.items():
        match = pattern.search(text_header)
        if match:
            byte_locations[field] = int(match.group(1))
            declared[field] = True
    return ByteLocationResult(byte_locations=byte_locations, declared=declared)


_VALID_TRACE_FIELD_OFFSETS = {int(member) for member in segyio.TraceField.enums()}


def resolve_trace_fields(byte_locations: dict[str, int]) -> dict[str, int]:
    """Resolve byte locations to segyio.TraceField values, i.e. validate
    that each declared byte corresponds to a known SEG-Y trace header
    field. segyio's TraceField enum values ARE the 1-indexed byte offsets,
    so this is a direct data-driven lookup -- never a hardcoded
    byte-offset table. Returns plain ints (matching how segyio's own named
    members like TraceField.FieldRecord behave -- calling TraceField(n)
    directly returns a different wrapper type that f.attributes() doesn't
    accept). Raises ValueError if a declared byte doesn't correspond to
    any known field (e.g. a mis-parsed number) -- segyio.TraceField itself
    doesn't raise for an unknown value (it returns a placeholder "Unknown
    Enum" object), so membership is checked explicitly first."""
    resolved: dict[str, int] = {}
    for field, byte in byte_locations.items():
        if byte not in _VALID_TRACE_FIELD_OFFSETS:
            raise ValueError(
                f"Byte location {byte} declared for '{field}' does not correspond to any known "
                "SEG-Y trace header field."
            )
        resolved[field] = int(byte)
    return resolved


def read_raw_textual_header(path: str) -> bytes:
    """Read the raw 3200-byte textual header directly from disk, bypassing
    segyio's f.text[] property entirely (see module docstring for why:
    segyio always EBCDIC-decodes/encodes through that property, which is
    exactly the assumption this module exists to not make)."""
    with open(path, "rb") as fh:
        return fh.read(TEXTUAL_HEADER_SIZE)


def detect_geometry(path: str) -> tuple[TextualHeaderResult, ByteLocationResult]:
    """Convenience wrapper: read the raw textual header off disk, decode it
    robustly, and parse byte locations from it. This is the entry point
    seismic_processor.py / segy_loader.py should use instead of assuming
    cp037 + rev1 byte locations."""
    raw = read_raw_textual_header(path)
    header_result = decode_textual_header(raw)
    byte_result = parse_byte_locations(header_result.text)
    return header_result, byte_result
