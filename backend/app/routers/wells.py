"""
routers/wells.py
-----------------
Endpoints for uploading, listing, and inspecting individual wells
(sections 4 of the brief).
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from app.las_loader import LasValidationError
from app.models.schemas import (
    CrossplotResponse,
    WellCurvesResponse,
    WellSummary,
    WellUploadResponse,
    WellZonesResponse,
)
from app.services import well_service

router = APIRouter(prefix="/wells", tags=["wells"])


@router.post("/upload", response_model=WellUploadResponse)
async def upload_wells(files: list[UploadFile] = File(...)) -> WellUploadResponse:
    """Upload one or more raw LAS files. Each file is validated, run through
    the full petrophysical interpretation pipeline, and persisted.
    """
    uploaded: list[WellSummary] = []
    errors: list[str] = []

    for file in files:
        try:
            content = await file.read()
            summary = well_service.process_and_store_las_bytes(content, file.filename)
            uploaded.append(summary)
        except LasValidationError as exc:
            errors.append(f"{file.filename}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- surface unexpected errors per-file
            errors.append(f"{file.filename}: unexpected error -- {exc}")

    if not uploaded and errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    return WellUploadResponse(uploaded=uploaded, errors=errors)


@router.get("", response_model=list[WellSummary])
async def list_wells() -> list[WellSummary]:
    """List all processed wells with summary stats."""
    return well_service.list_well_summaries()


@router.get("/{well_id}/curves", response_model=WellCurvesResponse)
async def get_well_curves(well_id: str) -> WellCurvesResponse:
    """Full processed curve data (raw + computed) as JSON."""
    try:
        data = well_service.get_well_curves(well_id)
    except well_service.WellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WellCurvesResponse(**data)


@router.get("/{well_id}/zones", response_model=WellZonesResponse)
async def get_well_zones(well_id: str) -> WellZonesResponse:
    """Zonation summary table (thickness/avg PHIE/SWE/VSH per zone)."""
    try:
        return well_service.get_well_zones(well_id)
    except well_service.WellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{well_id}/crossplot", response_model=CrossplotResponse)
async def get_crossplot(
    well_id: str,
    x: str = Query(..., description="Curve name for the X axis, e.g. NPHI"),
    y: str = Query(..., description="Curve name for the Y axis, e.g. RHOB"),
    color: str | None = Query(
        None, description="Optional curve name to color-code points by"
    ),
) -> CrossplotResponse:
    """Generic crossplot data endpoint -- supports any curve pair + optional color-by curve."""
    try:
        return well_service.get_crossplot(well_id, x, y, color)
    except well_service.WellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{well_id}/export")
async def export_well(
    well_id: str,
    format: str = Query("csv", pattern="^(csv|las)$", description="'csv' or 'las'"),
) -> PlainTextResponse:
    """Export processed LAS/CSV of the interpreted log."""
    try:
        if format == "las":
            content = well_service.export_well_las(well_id)
            media_type = "application/octet-stream"
            filename = f"{well_id}_interpreted.las"
        else:
            content = well_service.export_well_csv(well_id)
            media_type = "text/csv"
            filename = f"{well_id}_interpreted.csv"
    except well_service.WellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return PlainTextResponse(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
