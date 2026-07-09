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
NULL_VALUE = -9999.25
NULL_TOLERANCE = 0.01  # treat anything within this of -9999.25 as null


class LasValidationError(ValueError):
    """Raised when a LAS file is missing required curves or is unparseable."""


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
