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
    metadata, traces, twt_axis_ms, _attributes_df = seismic_service.get_seismic_dataset(dataset_id)
    seismic_dt_ms = metadata.sample_interval_ms

    # NOTE: the loaded dataset carries no per-trace coordinates (confirmed --
    # SegyMetadata/LoadedSegy have no X/Y fields), and these wells' LAS files
    # have no coordinates either. So a real nearest-trace-by-location tie
    # isn't possible with the current data. Using a manually configured
    # trace index from tie_config.yaml instead -- documented as a known
    # limitation, not a spatial match.
    override = config["well_coordinate_overrides"].get(well_id)
    if not override or "trace_index" not in override:
        raise TieError(
            f"No trace_index configured for well {well_id} in tie_config.yaml -- "
            "add one before requesting a tie. (Nearest-trace-by-location isn't "
            "available: neither the seismic dataset nor the LAS files carry "
            "coordinates.)"
        )

    trace_idx = int(override["trace_index"])
    if trace_idx >= traces.shape[0]:
        raise TieError(
            f"trace_index {trace_idx} is out of range for dataset {dataset_id} "
            f"({traces.shape[0]} traces available)."
        )
    distance_m = None
    geometry_warning = (
        "Using a manually configured trace index (tie_config.yaml) -- "
        f"well {well_id} has no coordinates in its LAS header, and the seismic "
        "dataset has no stored trace coordinates either. This is not a "
        "spatial nearest-trace match."
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
        twt_ms=twt_axis_ms.tolist(),
        synthetic=result.synthetic.tolist(),
        shifted_synthetic=tie["shifted_synthetic"].tolist(),
        real_trace=real_trace.tolist(),
        best_shift_ms=tie["best_shift_ms"],
        correlation=tie["correlation"],
        geometry_warning=geometry_warning,
    )