"""
routers/synthetic.py
----------------------
Synthetic seismogram / well-tie module endpoints, under the /api/synthetic
namespace (per the feature spec) -- distinct from both the upload-pipeline
tie (/tie/{well_id}, routers/tie.py) and the Seismic Visualization well-tie
(/api/seismic/well-tie/{well_id}, routers/seismic_viz.py). Those two tie a
well against a specific dataset id or the single active SEG-Y volume
respectively without the extra machinery this module adds: unit-
standardization QC reporting, selectable density method (real RHOB /
calibrated Gardner / rock-physics), selectable wavelet (statistical
extraction / Ricker) with amplitude+phase spectra, a washout QC proxy, and
persisted manual stretch/squeeze tie points.

All the underlying computation is reused from well_seismic_tie.py and
seismic_processor.py via services/synthetic_seismogram_service.py -- this
router is thin request/response plumbing only.
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    NearestTraceResponse,
    SaveTiePointsRequest,
    SyntheticSeismogramResponse,
    TiePointsResponse,
)
from app import well_seismic_tie as wst
from app.coordinate_calibration import CoordinateCalibrationError
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import synthetic_seismogram_service as sss
from app.services.well_service import WellNotFoundError

router = APIRouter(prefix="/api/synthetic", tags=["synthetic-seismogram"])


def _handle(exc: Exception):
    if isinstance(exc, (WellNotFoundError, sp.SegyFileNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (ccs.UnresolvedCoordinateError, CoordinateCalibrationError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, (sss.SyntheticSeismogramError, sp.SegyVolumeError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/{well_id}/generate", response_model=SyntheticSeismogramResponse)
async def generate(
    well_id: str,
    wavelet_method: str = Query("statistical", description="'statistical' (from the nearest real trace) or 'ricker'"),
    wavelet_freq_hz: float = Query(25.0, gt=0, description="Ricker dominant frequency, Hz (ignored for 'statistical')"),
    density_method: str = Query("rhob", description="'rhob' (real curve), 'gardner' (calibrated), or 'rock_physics'"),
    apply_saved_tie: bool = Query(True, description="Apply this well's persisted manual stretch/squeeze, if any"),
    max_shift_ms: float = Query(
        wst.DEFAULT_MAX_SHIFT_MS, gt=0,
        description="Bulk-shift correlation search range half-width, ms -- widen for checkshot-free wells whose sonic-derived time-depth curve may be off by 100-300ms",
    ),
) -> SyntheticSeismogramResponse:
    try:
        result = sss.generate(
            well_id,
            wavelet_method=wavelet_method,
            wavelet_freq_hz=wavelet_freq_hz,
            density_method=density_method,
            apply_saved_tie=apply_saved_tie,
            max_shift_ms=max_shift_ms,
        )
        return SyntheticSeismogramResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/{well_id}/nearest-trace", response_model=NearestTraceResponse)
async def nearest_trace(well_id: str) -> NearestTraceResponse:
    try:
        return NearestTraceResponse(**sss.nearest_trace(well_id))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/{well_id}/tie", response_model=TiePointsResponse | None)
async def get_tie(well_id: str) -> TiePointsResponse | None:
    saved = sss.get_tie_points(well_id)
    if saved is None:
        return None
    return TiePointsResponse(
        well_id=saved.well_id,
        points=[{"md_m": p.md_m, "time_shift_ms": p.time_shift_ms} for p in saved.points],
        wavelet_method=saved.wavelet_method,
        wavelet_freq_hz=saved.wavelet_freq_hz,
        segy_filename=saved.segy_filename,
    )


@router.put("/{well_id}/tie", response_model=TiePointsResponse)
async def save_tie(well_id: str, body: SaveTiePointsRequest) -> TiePointsResponse:
    """Persist manual stretch/squeeze control points for a well -- stored,
    not recomputed from scratch on the next /generate call
    (apply_saved_tie=true, the default)."""
    try:
        saved = sss.save_tie_points(
            well_id,
            [{"md_m": p.md_m, "time_shift_ms": p.time_shift_ms} for p in body.points],
            body.wavelet_method,
            body.wavelet_freq_hz,
        )
        return TiePointsResponse(
            well_id=saved.well_id,
            points=[{"md_m": p.md_m, "time_shift_ms": p.time_shift_ms} for p in saved.points],
            wavelet_method=saved.wavelet_method,
            wavelet_freq_hz=saved.wavelet_freq_hz,
            segy_filename=saved.segy_filename,
        )
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.delete("/{well_id}/tie")
async def delete_tie(well_id: str) -> dict:
    deleted = sss.delete_tie_points(well_id)
    return {"well_id": well_id, "deleted": deleted}


@router.get("/{well_id}/export")
async def export_tie_report(
    well_id: str,
    wavelet_method: str = Query("statistical"),
    wavelet_freq_hz: float = Query(25.0, gt=0),
    density_method: str = Query("rhob"),
) -> StreamingResponse:
    """CSV export: per-sample synthetic vs. real trace (on the seismic's
    time axis) plus a short tie-quality/QC summary header block."""
    try:
        result = sss.generate(
            well_id,
            wavelet_method=wavelet_method,
            wavelet_freq_hz=wavelet_freq_hz,
            density_method=density_method,
        )
    except Exception as exc:  # noqa: BLE001
        _handle(exc)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["# Synthetic seismogram tie report", well_id])
    writer.writerow(["# nearest_inline", result["nearest_inline"]])
    writer.writerow(["# nearest_crossline", result["nearest_crossline"]])
    writer.writerow(["# tie_method", result["tie_method"]])
    distance_m = result["distance_m"]
    writer.writerow(["# distance_m", f"{distance_m:.2f}" if distance_m is not None else "n/a (manual override)"])
    writer.writerow(["# correlation", f"{result['correlation']:.4f}"])
    writer.writerow(["# best_shift_ms", f"{result['best_shift_ms']:.2f}"])
    writer.writerow(["# max_shift_ms", f"{result['max_shift_ms']:.2f}"])
    writer.writerow(["# boundary_pinned", result["boundary_pinned"]])
    writer.writerow(["# datum_check_plausible", result["datum_check"]["plausible"]])
    writer.writerow(["# density_method", result["density_method"]])
    writer.writerow(["# density_note", result["density_note"]])
    writer.writerow(["# wavelet_method", result["wavelet_method"]])
    writer.writerow(["# vertical_assumption", result["vertical_assumption_note"]])
    writer.writerow(["# time_depth_note", result["time_depth_note"]])
    writer.writerow([])
    writer.writerow(["twt_ms", "synthetic", "shifted_synthetic", "real_trace"])
    for t, syn, shifted, real in zip(
        result["seismic_twt_ms"], result["synthetic"], result["shifted_synthetic"], result["real_trace"]
    ):
        writer.writerow([t, syn, shifted, real])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{well_id}_synthetic_tie.csv"'},
    )
