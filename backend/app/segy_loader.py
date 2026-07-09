"""
segy_loader.py
---------------
Reads raw SEG-Y seismic files with `segyio`, validates basic structure, and
returns a clean in-memory dataset: a 2D amplitude matrix (n_traces x
n_samples), the two-way-time sample axis, and metadata.

Mirrors the shape of las_loader.py (raw file in -> validated data + metadata
out), so the rest of the pipeline (seismic_attributes.py, repository,
services, routers) follows the exact same pattern already used for LAS
wells.

NOTE: `segyio` requires a real filesystem path (it uses low-level C
bindings and cannot read directly from an in-memory byte stream), so
uploaded files are written to a temporary file first -- the same fix
already applied to LAS uploads in las_loader.py.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import segyio


class SegyValidationError(ValueError):
    """Raised when a SEG-Y file is empty, corrupt, or fails basic sanity checks."""


@dataclass
class SegyMetadata:
    """Summary metadata describing one loaded seismic dataset."""

    dataset_id: str
    source_filename: str
    n_traces: int
    n_samples: int
    sample_interval_ms: float
    duration_ms: float


@dataclass
class LoadedSegy:
    """A fully loaded seismic dataset: amplitude matrix + time axis + metadata."""

    metadata: SegyMetadata
    traces: np.ndarray  # shape (n_traces, n_samples), amplitude values
    twt_axis_ms: np.ndarray  # shape (n_samples,), two-way time in ms
    trace_x: np.ndarray  # shape (n_traces,), surface X coordinate per trace (NaN if unavailable)
    trace_y: np.ndarray  # shape (n_traces,), surface Y coordinate per trace (NaN if unavailable)


def _apply_coord_scalar(raw: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """SEG-Y coordinates are stored as integers with a separate multiplier
    (SourceGroupScalar): positive values multiply, negative values divide
    (by the absolute value), 0/absent means 1 (no scaling)."""
    out = raw.astype(float).copy()
    positive = scalar > 0
    negative = scalar < 0
    out[positive] *= scalar[positive]
    out[negative] /= -scalar[negative]
    return out


def _extract_trace_coordinates(f, n_traces: int) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort extraction of per-trace surface coordinates from the
    trace headers (CDP_X/CDP_Y, falling back to SourceX/SourceY), so wells
    carrying their own surface coordinates (see las_loader.py) can be tied
    to the nearest real trace by location instead of a manually configured
    trace index -- see well_seismic_tie.find_nearest_trace_index.

    Returns arrays of NaN if the file has no usable coordinate headers; this
    is common for vendor exports/2D lines with blank or non-standard
    geometry bytes, and is treated as "coordinates not available" rather
    than an error.
    """
    try:
        scalar_raw = np.array(f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=float)
        scalar = np.where(scalar_raw == 0, 1.0, scalar_raw)

        cdp_x = _apply_coord_scalar(
            np.array(f.attributes(segyio.TraceField.CDP_X)[:], dtype=float), scalar
        )
        cdp_y = _apply_coord_scalar(
            np.array(f.attributes(segyio.TraceField.CDP_Y)[:], dtype=float), scalar
        )
        if np.any(cdp_x) or np.any(cdp_y):
            return cdp_x, cdp_y

        src_x = _apply_coord_scalar(
            np.array(f.attributes(segyio.TraceField.SourceX)[:], dtype=float), scalar
        )
        src_y = _apply_coord_scalar(
            np.array(f.attributes(segyio.TraceField.SourceY)[:], dtype=float), scalar
        )
        if np.any(src_x) or np.any(src_y):
            return src_x, src_y
    except Exception:  # pragma: no cover - segyio header quirks vary by vendor
        pass

    return np.full(n_traces, np.nan), np.full(n_traces, np.nan)


def _dataset_id_from_filename(path: Path) -> str:
    """Derive a dataset ID from the filename, e.g. 'Line_001.sgy' -> 'LINE_001'."""
    return path.stem.upper()


def load_segy_file(
    source: str | Path | BinaryIO, filename: str | None = None
) -> LoadedSegy:
    """Load and validate a single SEG-Y file.

    Parameters
    ----------
    source : path-like, or an in-memory upload stream (e.g. FastAPI's
        UploadFile content wrapped in io.BytesIO by the caller)
    filename : original filename, required when `source` is a stream
        (used to derive the dataset ID)

    Raises
    ------
    SegyValidationError if the file can't be parsed or contains no traces.
    """
    tmp_path: str | None = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        read_path = str(path)
    else:
        if filename is None:
            raise SegyValidationError("filename is required when loading from a stream")
        raw_bytes = source.read() if hasattr(source, "read") else source
        with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        read_path = tmp_path
        path = Path(filename)

    try:
        # ignore_geometry=True avoids requiring inline/crossline byte
        # locations to be correctly set in the trace headers -- many
        # real-world SEG-Y files (especially 2D lines or vendor exports)
        # have non-standard or absent geometry, so we treat the file as a
        # flat list of traces rather than requiring a 3D survey grid.
        with segyio.open(read_path, "r", ignore_geometry=True) as f:
            f.mmap()
            n_traces = f.tracecount
            if n_traces == 0:
                raise SegyValidationError(
                    f"SEG-Y file '{filename}' contains no traces."
                )

            twt_axis_ms = np.array(f.samples, dtype=float)
            n_samples = len(twt_axis_ms)
            if n_samples == 0:
                raise SegyValidationError(
                    f"SEG-Y file '{filename}' contains no samples."
                )

            sample_interval_ms = float(
                segyio.tools.dt(f) / 1000.0
            )  # dt() returns microseconds
            traces = segyio.tools.collect(f.trace[:]).astype(
                float
            )  # (n_traces, n_samples)
            trace_x, trace_y = _extract_trace_coordinates(f, n_traces)

    except SegyValidationError:
        raise
    except Exception as exc:  # pragma: no cover - segyio raises many types
        raise SegyValidationError(
            f"Failed to parse SEG-Y file '{filename}': {exc}"
        ) from exc
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    duration_ms = float(twt_axis_ms[-1] - twt_axis_ms[0]) if n_samples > 1 else 0.0

    metadata = SegyMetadata(
        dataset_id=_dataset_id_from_filename(path),
        source_filename=filename,
        n_traces=n_traces,
        n_samples=n_samples,
        sample_interval_ms=sample_interval_ms,
        duration_ms=duration_ms,
    )

    return LoadedSegy(
        metadata=metadata, traces=traces, twt_axis_ms=twt_axis_ms, trace_x=trace_x, trace_y=trace_y
    )
