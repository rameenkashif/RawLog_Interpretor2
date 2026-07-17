"""
services/tie_service.py
------------------------
Orchestrates the well-to-seismic tie: pulls well curves + seismic trace data,
calls well_seismic_tie.py, assembles the API response.

Tie algorithm: each well's own DPTM curve (vendor-precomputed when the LAS
carries one, else petrophysics.compute_dptm's sonic-integration fallback --
see that module) is trusted directly as the time axis, and the tie search
jointly sweeps Ricker wavelet frequency, polarity, and bulk time shift across
the ENTIRE seismic window (well_seismic_tie.search_best_tie_full_window)
rather than a fixed wavelet frequency with a narrow position-only search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from app.well_seismic_tie import (
    BOUNDARY_PINNED_FRACTION,
    TieError,
    find_nearest_trace_index,
    reflectivity_from_time_axis,
    search_best_tie_full_window,
)
from app.models.schemas import (
    SurveyFootprintPoint,
    WellSeismicTieBatchResponse,
    WellSeismicTieResponse,
    WellSeismicTieRow,
)

from app.services import well_service
from app.services import seismic_service
from app.services.seismic_service import SeismicDatasetNotFoundError

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tie_config.yaml"

# How many trace-coordinate points to send back for the map's background
# survey footprint -- a real dataset can have tens of thousands of traces,
# far more than a browser needs to render a footprint scatter.
MAX_FOOTPRINT_POINTS = 1500


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _extract_well_curves(well_id: str) -> dict[str, np.ndarray]:
    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]  # list of {curve_name: value} dicts, one per depth sample

    def _extract(curve_name: str) -> np.ndarray:
        arr = np.array(
            [row.get(curve_name) if row.get(curve_name) is not None else np.nan for row in rows],
            dtype=float,
        )
        arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
        return arr

    return {name: _extract(name) for name in ("DEPT", "DT", "RHOB", "DPTM")}


def _resolve_trace(
    well_id: str,
    well_x: float | None,
    well_y: float | None,
    dataset_id: str,
    traces: np.ndarray,
    trace_x: np.ndarray,
    trace_y: np.ndarray,
    trace_inline: np.ndarray,
    trace_crossline: np.ndarray,
    config: dict,
) -> tuple[int, float | None, str, str | None]:
    """Prefer a real spatial nearest-trace match when both the well (LAS
    header, see las_loader.py) and the seismic dataset (trace headers, see
    segy_loader.py) carry surface coordinates. Falls back to a manually
    configured trace index from tie_config.yaml when coordinates aren't
    available on one side or the other -- this keeps older wells/datasets
    without coordinate headers working, just without a spatial guarantee.

    Returns (trace_idx, distance_m, tie_method, geometry_warning).
    """
    has_well_coords = well_x is not None and well_y is not None
    has_trace_coords = trace_x.size > 0 and np.isfinite(trace_x).any() and np.isfinite(trace_y).any()

    if has_well_coords and has_trace_coords:
        trace_idx, distance_m = find_nearest_trace_index(
            well_x,
            well_y,
            trace_x,
            trace_y,
            max_radius_m=config.get("max_tie_search_radius_m"),
        )
        return trace_idx, distance_m, "nearest_trace", None

    override = config["well_coordinate_overrides"].get(well_id)
    if not override or "trace_index" not in override:
        raise TieError(
            f"No coordinates available for a spatial tie (well {well_id}: "
            f"{'has' if has_well_coords else 'missing'} LAS coordinates, "
            f"dataset {dataset_id}: {'has' if has_trace_coords else 'missing'} "
            "trace coordinates), and no trace_index configured in "
            "tie_config.yaml as a fallback -- add one before requesting a tie."
        )

    trace_idx = int(override["trace_index"])
    if trace_idx >= traces.shape[0]:
        raise TieError(
            f"trace_index {trace_idx} is out of range for dataset {dataset_id} "
            f"({traces.shape[0]} traces available)."
        )
    geometry_warning = (
        "Using a manually configured trace index (tie_config.yaml) -- "
        f"well {well_id} {'has' if has_well_coords else 'has no'} coordinates in "
        f"its LAS header, and the seismic dataset {'has' if has_trace_coords else 'has no'} "
        "stored trace coordinates. This is not a spatial nearest-trace match."
    )
    return trace_idx, None, "manual_override", geometry_warning


def _trace_inline_crossline(trace_inline: np.ndarray, trace_crossline: np.ndarray, trace_idx: int) -> tuple[int | None, int | None]:
    inline = float(trace_inline[trace_idx]) if trace_inline.size > trace_idx else float("nan")
    crossline = float(trace_crossline[trace_idx]) if trace_crossline.size > trace_idx else float("nan")
    return (
        int(inline) if np.isfinite(inline) else None,
        int(crossline) if np.isfinite(crossline) else None,
    )


def get_well_seismic_tie(well_id: str, dataset_id: str) -> WellSeismicTieResponse:
    config = _load_config()
    max_shift_ms = float(config.get("tie_search_max_shift_ms", 100.0))

    well_summary = well_service.get_well_summary(well_id)
    curves = _extract_well_curves(well_id)

    # get_seismic_dataset() raises SeismicDatasetNotFoundError itself if the
    # dataset_id doesn't exist -- let it propagate, the router already
    # catches this exception type.
    metadata, traces, twt_axis_ms, trace_x, trace_y, trace_inline, trace_crossline, _attributes_df = (
        seismic_service.get_seismic_dataset(dataset_id)
    )
    seismic_dt_ms = metadata.sample_interval_ms

    trace_idx, distance_m, tie_method, geometry_warning = _resolve_trace(
        well_id,
        well_summary.well_x,
        well_summary.well_y,
        dataset_id,
        traces,
        trace_x,
        trace_y,
        trace_inline,
        trace_crossline,
        config,
    )
    inline, crossline = _trace_inline_crossline(trace_inline, trace_crossline, trace_idx)
    real_trace = traces[trace_idx].astype(float)

    t_rc, rc = reflectivity_from_time_axis(curves["DPTM"], curves["DT"], curves["RHOB"], seismic_dt_ms)
    tie = search_best_tie_full_window(
        t_rc, rc, twt_axis_ms, seismic_dt_ms, real_trace, max_shift_ms=max_shift_ms
    )
    boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - BOUNDARY_PINNED_FRACTION) * max_shift_ms

    return WellSeismicTieResponse(
        well_id=well_id,
        dataset_id=dataset_id,
        trace_index=trace_idx,
        distance_m=distance_m,
        tie_method=tie_method,
        inline=inline,
        crossline=crossline,
        best_freq_hz=tie.best_freq_hz,
        polarity=tie.polarity,
        bulk_shift_ms=tie.bulk_shift_ms,
        correlation=tie.correlation,
        max_shift_ms=max_shift_ms,
        boundary_pinned=boundary_pinned,
        n_used=tie.n_used,
        time_ms=tie.time_ms.tolist(),
        synthetic_amplitude=tie.synthetic_amplitude.tolist(),
        seismic_amplitude=tie.seismic_amplitude.tolist(),
        reflectivity=tie.reflectivity.tolist(),
        geometry_warning=geometry_warning,
    )


def get_all_well_ties(dataset_id: str) -> WellSeismicTieBatchResponse:
    """Batch tie: run get_well_seismic_tie's same algorithm for every well in
    the repository against one seismic dataset, for a results table + map
    (mirrors the notebook's per-well loop + results DataFrame). Wells that
    fail (missing curves, no coordinates, etc.) get a row with `error` set
    rather than being silently dropped -- so a partial batch is still
    visible as partial, not indistinguishable from "only these wells exist".
    """
    config = _load_config()
    max_shift_ms = float(config.get("tie_search_max_shift_ms", 100.0))

    metadata, traces, twt_axis_ms, trace_x, trace_y, trace_inline, trace_crossline, _attributes_df = (
        seismic_service.get_seismic_dataset(dataset_id)
    )
    seismic_dt_ms = metadata.sample_interval_ms

    rows: list[WellSeismicTieRow] = []
    warnings: list[str] = []

    for well_summary in well_service.list_well_summaries():
        well_id = well_summary.well_id
        try:
            curves = _extract_well_curves(well_id)
            trace_idx, distance_m, tie_method, geometry_warning = _resolve_trace(
                well_id,
                well_summary.well_x,
                well_summary.well_y,
                dataset_id,
                traces,
                trace_x,
                trace_y,
                trace_inline,
                trace_crossline,
                config,
            )
            if geometry_warning:
                warnings.append(f"{well_id}: {geometry_warning}")
            inline, crossline = _trace_inline_crossline(trace_inline, trace_crossline, trace_idx)
            real_trace = traces[trace_idx].astype(float)
            t_rc, rc = reflectivity_from_time_axis(
                curves["DPTM"], curves["DT"], curves["RHOB"], seismic_dt_ms
            )
            tie = search_best_tie_full_window(
                t_rc, rc, twt_axis_ms, seismic_dt_ms, real_trace, max_shift_ms=max_shift_ms
            )
            boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - BOUNDARY_PINNED_FRACTION) * max_shift_ms

            rows.append(
                WellSeismicTieRow(
                    well_id=well_id,
                    well_x=well_summary.well_x,
                    well_y=well_summary.well_y,
                    trace_index=trace_idx,
                    trace_x=float(trace_x[trace_idx]) if trace_x.size > trace_idx and np.isfinite(trace_x[trace_idx]) else None,
                    trace_y=float(trace_y[trace_idx]) if trace_y.size > trace_idx and np.isfinite(trace_y[trace_idx]) else None,
                    inline=inline,
                    crossline=crossline,
                    distance_m=distance_m,
                    tie_method=tie_method,
                    best_freq_hz=tie.best_freq_hz,
                    polarity=tie.polarity,
                    bulk_shift_ms=tie.bulk_shift_ms,
                    correlation=tie.correlation,
                    boundary_pinned=boundary_pinned,
                )
            )
        except (TieError, well_service.WellNotFoundError) as exc:
            rows.append(
                WellSeismicTieRow(
                    well_id=well_id,
                    well_x=well_summary.well_x,
                    well_y=well_summary.well_y,
                    error=str(exc),
                )
            )
            warnings.append(f"{well_id}: {exc}")

    survey_footprint: list[SurveyFootprintPoint] = []
    finite = np.isfinite(trace_x) & np.isfinite(trace_y)
    if finite.any():
        xs, ys = trace_x[finite], trace_y[finite]
        step = max(1, len(xs) // MAX_FOOTPRINT_POINTS)
        survey_footprint = [
            SurveyFootprintPoint(x=float(x), y=float(y)) for x, y in zip(xs[::step], ys[::step])
        ]

    return WellSeismicTieBatchResponse(
        dataset_id=dataset_id, rows=rows, survey_footprint=survey_footprint, warnings=warnings
    )
