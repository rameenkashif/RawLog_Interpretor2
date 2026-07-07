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
        if filename is None:
            raise LasValidationError("filename is required when loading from a stream")
        try:
            las = lasio.read(source)
        except Exception as exc:  # pragma: no cover
            raise LasValidationError(
                f"Failed to parse LAS file '{filename}': {exc}"
            ) from exc
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
