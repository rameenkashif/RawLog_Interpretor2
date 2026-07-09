"""
services/tie_service.py
------------------------
Orchestrates the well-to-seismic tie: pulls well curves + seismic trace data,
calls well_seismic_tie.py, assembles the API response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from app.well_seismic_tie import (
    TieError,
    build_synthetic,
    cross_correlate_and_shift,
    find_nearest_trace_index,
)
from app.models.schemas import WellSeismicTieResponse

from app.services import well_service
from app.services import seismic_service
from app.services.seismic_service import SeismicDatasetNotFoundError

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tie_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_well_seismic_tie(well_id: str, dataset_id: str) -> WellSeismicTieResponse:
    config = _load_config()

    well_summary = well_service.get_well_summary(well_id)
    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]  # list of {curve_name: value} dicts, one per depth sample

    def _extract(curve_name: str) -> np.ndarray:
        return np.array(
            [row.get(curve_name) if row.get(curve_name) is not None else np.nan for row in rows],
            dtype=float,
        )

    depth = _extract("DEPT")
    dt_log = _extract("DT")
    rhob = _extract("RHOB")
    vsh = _extract("VSH")
    phie = _extract("PHIE")
    swe = _extract("SWE")
    # Guard against LAS null sentinel leaking through.
    for arr in (dt_log, rhob, vsh, phie, swe):
        if arr.size:
            arr[arr <= -9999.0] = np.nan

    # get_seismic_dataset() raises SeismicDatasetNotFoundError itself if the
    # dataset_id doesn't exist -- let it propagate, the router already
    # catches this exception type.
    metadata, traces, twt_axis_ms, trace_x, trace_y, _attributes_df = (
        seismic_service.get_seismic_dataset(dataset_id)
    )
    seismic_dt_ms = metadata.sample_interval_ms

    # Prefer a real spatial nearest-trace match when both the well (LAS
    # header, see las_loader.py) and the seismic dataset (trace headers, see
    # segy_loader.py) carry surface coordinates. Falls back to a manually
    # configured trace index from tie_config.yaml when coordinates aren't
    # available on one side or the other -- this keeps older wells/datasets
    # without coordinate headers working, just without a spatial guarantee.
    has_well_coords = well_summary.well_x is not None and well_summary.well_y is not None
    has_trace_coords = trace_x.size > 0 and np.isfinite(trace_x).any() and np.isfinite(trace_y).any()

    if has_well_coords and has_trace_coords:
        trace_idx, distance_m = find_nearest_trace_index(
            well_summary.well_x,
            well_summary.well_y,
            trace_x,
            trace_y,
            max_radius_m=config.get("max_tie_search_radius_m"),
        )
        tie_method = "nearest_trace"
        geometry_warning = None
    else:
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
        distance_m = None
        tie_method = "manual_override"
        geometry_warning = (
            "Using a manually configured trace index (tie_config.yaml) -- "
            f"well {well_id} {'has' if has_well_coords else 'has no'} coordinates in "
            f"its LAS header, and the seismic dataset {'has' if has_trace_coords else 'has no'} "
            "stored trace coordinates. This is not a spatial nearest-trace match."
        )

    real_trace = traces[trace_idx].astype(float)

    result = build_synthetic(
        depth_m=depth,
        dt_log=dt_log,
        rhob=rhob,
        seismic_dt_ms=seismic_dt_ms,
        seismic_twt_axis_ms=twt_axis_ms,
        wavelet_freq_hz=config["wavelet_freq_hz"],
        dt_unit=config["dt_unit"],
    )

    tie = cross_correlate_and_shift(result.synthetic, real_trace, seismic_dt_ms)

    return WellSeismicTieResponse(
        well_id=well_id,
        dataset_id=dataset_id,
        trace_index=trace_idx,
        distance_m=distance_m,
        tie_method=tie_method,
        twt_ms=twt_axis_ms.tolist(),
        synthetic=result.synthetic.tolist(),
        shifted_synthetic=tie["shifted_synthetic"].tolist(),
        real_trace=real_trace.tolist(),
        best_shift_ms=tie["best_shift_ms"],
        correlation=tie["correlation"],
        geometry_warning=geometry_warning,
    )