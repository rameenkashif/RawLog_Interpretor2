"""
las_loader.py
--------------
Reads raw LAS well log files with `lasio`, validates that the five required
curves are present, flags missing/null values (LAS null sentinel is
typically -9999.25), and returns a clean `pandas.DataFrame` per well plus
metadata (well name, start/stop depth, step).

This module is intentionally decoupled from the petrophysics calculations
(petrophysics.py) -- its only job is "raw LAS file in, clean DataFrame +
metadata out".
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import lasio
import numpy as np
import pandas as pd

REQUIRED_CURVES = ["DEPT", "GR", "RESISTIVITY", "RHOB", "NPHI", "DT"]
# Optional curves: loaded (and null-cleaned) exactly like the required ones
# when present, but their absence doesn't fail validation. DPTM is a
# vendor-precomputed two-way-time depth/time track some LAS exports carry
# directly -- when present and plausible, well_seismic_tie/petrophysics
# prefer it over re-deriving one by sonic integration (see
# petrophysics.compute_dptm), since it's a real calibrated curve rather
# than an approximation.
OPTIONAL_CURVES = ["DPTM"]
NULL_VALUE = -9999.25
NULL_TOLERANCE = 0.01  # treat anything within this of -9999.25 as null


class LasValidationError(ValueError):
    """Raised when a LAS file is missing required curves or is unparseable."""


@dataclass
class CurveUnitInfo:
    """Per-curve unit provenance: what the LAS header declared (if
    anything) vs. what was inferred from the curve's own value range when
    the header was blank, and whether that inference was actually used.
    See _resolve_curve_unit() -- surfaced so a user can see why DT was (or
    wasn't) unit-converted, not just the resulting numbers."""

    curve: str
    declared_unit_raw: str | None
    resolved_unit: str | None
    inferred: bool
    value_range_used: tuple[float, float] | None = None
    conversion_applied: bool = False


@dataclass
class WellMetadata:
    """Summary metadata describing one loaded well."""

    well_id: str
    well_name: str
    source_filename: str
    start_depth: float
    stop_depth: float
    step: float
    n_samples: int
    null_counts: dict[str, int] = field(default_factory=dict)
    missing_curves: list[str] = field(default_factory=list)
    well_x: float | None = None
    well_y: float | None = None
    kb_m: float | None = None
    td_m: float | None = None
    coordinate_unit_detected: str | None = None  # "feet", "meters", or None if unvalidated
    unit_conversion_applied: bool = False
    td_stop_ratio: float | None = None
    curve_units: list[CurveUnitInfo] = field(default_factory=list)


@dataclass
class LoadedWell:
    """A fully loaded, validated well: clean curve data + metadata."""

    metadata: WellMetadata
    df: pd.DataFrame


def _resolve_curve_name(las: lasio.LASFile, canonical: str) -> str | None:
    """LAS curve mnemonics vary in casing/aliasing across vendors
    (e.g. "RESISTIVITY" might be logged as "RT", "RES_DEEP", "RDEP"...).
    This resolves a canonical name to whatever mnemonic actually exists in
    the file, trying a small set of common aliases first.
    """
    aliases = {
        "DEPT": ["DEPT", "DEPTH", "MD"],
        "GR": ["GR", "GRAY", "GAMMA"],
        "RESISTIVITY": ["RESISTIVITY", "RT", "RDEP", "RES_DEEP", "ILD", "LLD"],
        "RHOB": ["RHOB", "DEN", "RHOZ"],
        "NPHI": ["NPHI", "NEUT", "TNPH"],
        "DT": ["DT", "DTC", "AC"],
    }
    available = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    for alias in aliases.get(canonical, [canonical]):
        if alias.upper() in available:
            return available[alias.upper()]
    return None


def _well_id_from_filename(path: Path) -> str:
    """Derive a well ID from the filename, e.g. 'Z-02.las' -> 'Z-02'."""
    return path.stem.upper()


# Surface coordinate / header mnemonics vary by vendor just like curve
# mnemonics do. These are read from the ~Well section (single-value header
# items), not the ~Curve section. Coordinates are assumed to be in a
# consistent, Euclidean (e.g. UTM easting/northing) CRS shared with any
# seismic data they'll be compared against -- see
# well_seismic_tie.find_nearest_trace_index -- but see
# _standardize_well_header() below: some vendor LAS exports label these
# fields ".m" while actually storing feet, so the unit is validated, not
# assumed, before that comparison is trusted.
_X_COORD_ALIASES = ["XWELL", "XCOORD", "SURFACE_X", "SURX", "X"]
_Y_COORD_ALIASES = ["YWELL", "YCOORD", "SURFACE_Y", "SURY", "Y"]
_KB_ALIASES = ["KB", "KBELEV", "KELLY_BUSHING", "EKB"]
_TD_ALIASES = ["TD", "TOTAL_DEPTH", "TDD", "TDL"]

FT_TO_M = 0.3048
# A well genuinely logged in feet will show a TD/STOP ratio close to the
# feet-to-meters factor (3.28084), since TD (total driller's depth, usually
# slightly deeper than the logged interval) divided by STOP (always in
# meters -- the curve data itself) collapses to roughly that factor times
# TD_m/STOP_m (typically ~1.0-1.15 for a real well). This range is wide
# enough to catch that real-world spread without also matching a well that
# is genuinely already in meters (which would show a ratio near 1).
_FEET_RATIO_RANGE = (2.8, 4.2)


def _resolve_well_numeric(las: lasio.LASFile, aliases: list[str]) -> float | None:
    """Look up a numeric ~Well section header item by mnemonic, trying
    common aliases in order. Returns None if absent, blank, or unparseable
    -- these are optional metadata, not required curves.
    """
    for alias in aliases:
        item = las.well.get(alias)
        if item is None or item.value in (None, ""):
            continue
        try:
            return float(item.value)
        except (TypeError, ValueError):
            continue
    return None


def _standardize_well_header(
    well_x: float | None, well_y: float | None, kb: float | None, td: float | None, stop_depth: float
) -> dict:
    """Detect and correct feet-labeled-as-meters header fields (X, Y, KB,
    TD), confirmed to occur in some vendor LAS exports for this field:
    those four fields are stored in feet despite the LAS header declaring
    unit "m", while STRT/STOP/DEPT (the curve data) are genuinely in
    meters. Detected per-well via the TD/STOP ratio -- NOT hardcoded --
    since a future well's export may already be unit-consistent.

    Returns a dict with (possibly converted) well_x/well_y/kb_m/td_m plus
    QC fields (coordinate_unit_detected, unit_conversion_applied,
    td_stop_ratio) so callers can report what was done, not just the
    resulting numbers.
    """
    result = {
        "well_x": well_x,
        "well_y": well_y,
        "kb_m": kb,
        "td_m": td,
        "coordinate_unit_detected": None,
        "unit_conversion_applied": False,
        "td_stop_ratio": None,
    }

    if td is None or not stop_depth:
        # Can't validate without both TD and a non-zero STOP depth to
        # compare against -- leave values as-is rather than guessing.
        return result

    ratio = td / stop_depth
    result["td_stop_ratio"] = ratio

    if _FEET_RATIO_RANGE[0] <= ratio <= _FEET_RATIO_RANGE[1]:
        result["coordinate_unit_detected"] = "feet"
        result["unit_conversion_applied"] = True
        result["well_x"] = well_x * FT_TO_M if well_x is not None else None
        result["well_y"] = well_y * FT_TO_M if well_y is not None else None
        result["kb_m"] = kb * FT_TO_M if kb is not None else None
        result["td_m"] = td * FT_TO_M
    else:
        result["coordinate_unit_detected"] = "meters"
        # Already consistent with STOP's units -- no conversion needed.

    return result


# DT unit ambiguity is the one that actually matters for downstream
# physics (time-depth conversion, velocity, acoustic impedance -- see
# well_seismic_tie.py): a blank LAS unit field is common, and the raw
# values alone (~48-97 for a real DT curve seen in this field) look
# unremarkable but are meaningless without knowing us/ft vs us/m. Ranges
# below are deliberately ordered with the far more common industry
# convention (us/ft) checked first, so a value in the ambiguous overlap
# (100-160) resolves to us/ft rather than us/m absent other evidence.
# The other curves' ranges exist mainly for QC/display (this app doesn't
# multi-unit-handle RHOB/GR/NPHI elsewhere), but are inferred the same way
# for consistency and so a genuinely wrong-range value is visible.
_VALUE_RANGE_UNITS: dict[str, list[tuple[str, float, float]]] = {
    "DT": [("us_per_ft", 20.0, 160.0), ("us_per_m", 100.0, 550.0)],
    "RHOB": [("g_cc", 1.0, 3.2)],
    "GR": [("api", 0.0, 300.0)],
    "NPHI": [("v_v", 0.0, 1.0)],
}

_DECLARED_UNIT_ALIASES: dict[str, str] = {
    "us/f": "us_per_ft", "usec/f": "us_per_ft", "us/ft": "us_per_ft", "usec/ft": "us_per_ft",
    "us/m": "us_per_m", "usec/m": "us_per_m",
    "g/c3": "g_cc", "g/cc": "g_cc", "gm/cc": "g_cc", "gcc": "g_cc", "k/m3": "kg_m3",
    "api": "api", "gapi": "api",
    "v/v": "v_v", "frac": "v_v", "fraction": "v_v", "dec": "v_v", "vol/vol": "v_v",
}

FT_TO_M_TIME_FACTOR = FT_TO_M  # us_per_ft = us_per_m * FT_TO_M (see acoustic_impedance in well_seismic_tie.py)


def _infer_unit_from_values(canonical: str, values: np.ndarray) -> tuple[str | None, tuple[float, float] | None]:
    """Infer a curve's unit from its own median value falling inside a
    known plausible range. Ranges are checked in the order given in
    _VALUE_RANGE_UNITS (first match wins), so overlapping ranges (DT's
    us_per_ft/us_per_m) resolve deterministically rather than ambiguously."""
    ranges = _VALUE_RANGE_UNITS.get(canonical)
    if not ranges:
        return None, None
    valid = values[np.isfinite(values)] if len(values) else values
    if valid.size == 0:
        return None, None
    median = float(np.nanmedian(valid))
    for unit, lo, hi in ranges:
        if lo <= median <= hi:
            return unit, (lo, hi)
    return None, None


def _resolve_curve_unit(canonical: str, declared_unit_raw: str | None, values: np.ndarray) -> CurveUnitInfo:
    """Resolve a curve's unit: trust a recognized declared LAS unit field
    first, and only fall back to value-range inference when the header is
    blank or the declared string isn't one of the recognized aliases
    (garbage/unfamiliar unit strings are common in vendor exports)."""
    declared_norm = _DECLARED_UNIT_ALIASES.get((declared_unit_raw or "").strip().lower())
    if declared_norm:
        return CurveUnitInfo(
            curve=canonical, declared_unit_raw=declared_unit_raw, resolved_unit=declared_norm, inferred=False
        )
    inferred_unit, value_range = _infer_unit_from_values(canonical, values)
    return CurveUnitInfo(
        curve=canonical,
        declared_unit_raw=declared_unit_raw,
        resolved_unit=inferred_unit,
        inferred=True,
        value_range_used=value_range,
    )


def _resolve_and_normalize_curve_units(
    las: lasio.LASFile, df: pd.DataFrame, resolved_mnemonics: dict[str, str]
) -> list[CurveUnitInfo]:
    """Resolve units for every curve with a known value-range profile, and
    normalize DT to us_per_ft IN PLACE on df if it resolved to us_per_m --
    everything downstream (well_seismic_tie.py's depth_to_twt/
    acoustic_impedance) assumes DT is already us_per_ft. Other curves are
    resolved for QC/display only (this app doesn't unit-convert
    RHOB/GR/NPHI elsewhere, so there's nothing further to normalize)."""
    infos: list[CurveUnitInfo] = []
    for canonical in _VALUE_RANGE_UNITS:
        if canonical not in df.columns:
            continue
        mnemonic = resolved_mnemonics.get(canonical, canonical)
        curve_item = las.curves.get(mnemonic) if mnemonic in las.curves else None
        declared_unit_raw = (curve_item.unit if curve_item is not None else None) or None
        info = _resolve_curve_unit(canonical, declared_unit_raw, df[canonical].to_numpy(dtype=float))

        if canonical == "DT" and info.resolved_unit == "us_per_m":
            df["DT"] = df["DT"] * FT_TO_M_TIME_FACTOR
            info.conversion_applied = True

        infos.append(info)
    return infos


def load_las_file(
    source: str | Path | BinaryIO, filename: str | None = None
) -> LoadedWell:
    """Load and validate a single LAS file.

    Parameters
    ----------
    source : path-like or file-like object (e.g. an UploadFile.file stream)
    filename : original filename, required when `source` is a file-like
        object (used to derive the well ID)

    Raises
    ------
    LasValidationError if required curves are missing or the file can't be
    parsed at all.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        try:
            las = lasio.read(str(path))
        except Exception as exc:  # pragma: no cover - lasio raises many types
            raise LasValidationError(
                f"Failed to parse LAS file '{filename}': {exc}"
            ) from exc
    else:
        # `source` is an in-memory upload stream (e.g. FastAPI's UploadFile.file,
        # wrapped in io.BytesIO by the caller). LAS files are plain text, but
        # `lasio.read()` does its own encoding detection and line-based text
        # parsing when given a *path* -- feeding it a raw binary stream directly
        # bypasses that and breaks internal str operations (bytes vs str
        # mismatches). To keep upload behavior identical to the path-based
        # CLI/script path (which works correctly), write the uploaded bytes to
        # a temporary .las file on disk and let lasio read/decode it exactly
        # the same way it does for backend/data/raw/*.las files.
        if filename is None:
            raise LasValidationError("filename is required when loading from a stream")

        raw_bytes = source.read() if hasattr(source, "read") else source

        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            las = lasio.read(tmp_path)
        except Exception as exc:  # pragma: no cover
            raise LasValidationError(
                f"Failed to parse LAS file '{filename}': {exc}"
            ) from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        path = Path(filename)

    # Resolve each required curve to whatever mnemonic is actually present.
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical in REQUIRED_CURVES:
        actual = _resolve_curve_name(las, canonical)
        if actual is None:
            missing.append(canonical)
        else:
            resolved[canonical] = actual

    if missing:
        raise LasValidationError(
            f"LAS file '{filename}' is missing required curve(s): {', '.join(missing)}. "
            f"Available curves: {[c.mnemonic for c in las.curves]}"
        )

    for canonical in OPTIONAL_CURVES:
        actual = _resolve_curve_name(las, canonical)
        if actual is not None:
            resolved[canonical] = actual

    df = pd.DataFrame(
        {canonical: las[actual] for canonical, actual in resolved.items()}
    )

    # Flag null sentinel values (-9999.25, or whatever the file header declares)
    file_null_value = las.well.get("NULL")
    null_value = (
        float(file_null_value.value)
        if file_null_value and file_null_value.value
        else NULL_VALUE
    )

    null_counts: dict[str, int] = {}
    for col in df.columns:
        is_null = np.isclose(df[col], null_value, atol=NULL_TOLERANCE)
        null_counts[col] = int(is_null.sum())
        df.loc[is_null, col] = np.nan

    # Drop rows where depth itself is null -- can't do anything without it.
    df = df.dropna(subset=["DEPT"]).reset_index(drop=True)

    if df.empty:
        raise LasValidationError(
            f"LAS file '{filename}' has no valid depth samples after cleaning."
        )

    # Resolve (and, for DT, normalize) curve units AFTER null cleaning --
    # a raw -9999.25 sentinel would badly skew the median used for
    # value-range inference.
    curve_units = _resolve_and_normalize_curve_units(las, df, resolved)

    depths = df["DEPT"].to_numpy()
    start_depth = float(depths[0])
    stop_depth = float(depths[-1])
    step = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.0

    well_id = _well_id_from_filename(path)
    well_name = las.well.get("WELL")
    well_name = well_name.value if well_name and well_name.value else well_id

    well_x = _resolve_well_numeric(las, _X_COORD_ALIASES)
    well_y = _resolve_well_numeric(las, _Y_COORD_ALIASES)
    kb = _resolve_well_numeric(las, _KB_ALIASES)
    td = _resolve_well_numeric(las, _TD_ALIASES)
    standardized = _standardize_well_header(well_x, well_y, kb, td, stop_depth)

    metadata = WellMetadata(
        well_id=well_id,
        well_name=str(well_name),
        source_filename=filename,
        start_depth=start_depth,
        stop_depth=stop_depth,
        step=step,
        n_samples=len(df),
        null_counts=null_counts,
        missing_curves=[],
        well_x=standardized["well_x"],
        well_y=standardized["well_y"],
        kb_m=standardized["kb_m"],
        td_m=standardized["td_m"],
        coordinate_unit_detected=standardized["coordinate_unit_detected"],
        unit_conversion_applied=standardized["unit_conversion_applied"],
        td_stop_ratio=standardized["td_stop_ratio"],
        curve_units=curve_units,
    )

    return LoadedWell(metadata=metadata, df=df)


def load_las_folder(folder: str | Path) -> list[LoadedWell]:
    """Load every .las file found in a folder (non-recursive).

    Files that fail validation are skipped with a printed warning rather
    than aborting the whole batch, so one bad file doesn't block loading
    the rest of the field's wells.
    """
    folder = Path(folder)
    wells: list[LoadedWell] = []
    for las_path in sorted(folder.glob("*.las")):
        try:
            wells.append(load_las_file(las_path))
        except LasValidationError as exc:
            print(f"[las_loader] WARNING: skipping '{las_path.name}': {exc}")
    return wells
