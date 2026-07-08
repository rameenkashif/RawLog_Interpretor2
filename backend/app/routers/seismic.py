"""
routers/seismic.py
--------------------
Endpoints for uploading and inspecting SEG-Y seismic datasets, mirroring
routers/wells.py's pattern for LAS wells.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.models.schemas import (
    SeismicAttributesResponse,
    SeismicSectionResponse,
    SeismicSummary,
    SeismicUploadResponse,
)
from app.segy_loader import SegyValidationError
from app.services import seismic_service

router = APIRouter(prefix="/seismic", tags=["seismic"])


@router.post("/upload", response_model=SeismicUploadResponse)
async def upload_seismic(files: list[UploadFile] = File(...)) -> SeismicUploadResponse:
    """Upload one or more raw SEG-Y files. Each file is validated, run
    through the seismic attribute pipeline (including the heuristic
    VSH/PHIE/SWE proxies), and persisted.
    """
    uploaded: list[SeismicSummary] = []
    errors: list[str] = []

    for file in files:
        try:
            content = await file.read()
            summary = seismic_service.process_and_store_segy_bytes(
                content, file.filename
            )
            uploaded.append(summary)
        except SegyValidationError as exc:
            errors.append(f"{file.filename}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- surface unexpected errors per-file
            errors.append(f"{file.filename}: unexpected error -- {exc}")

    if not uploaded and errors:
        raise HTTPException(status_code=422, detail={"errors": errors})

    return SeismicUploadResponse(uploaded=uploaded, errors=errors)


@router.get("", response_model=list[SeismicSummary])
async def list_seismic() -> list[SeismicSummary]:
    """List all processed seismic datasets with summary stats."""
    return seismic_service.list_seismic_summaries()


@router.get("/{dataset_id}/section", response_model=SeismicSectionResponse)
async def get_seismic_section(dataset_id: str) -> SeismicSectionResponse:
    """Subsampled raw amplitude section (trace x two-way-time) for display."""
    try:
        return seismic_service.get_seismic_section(dataset_id)
    except seismic_service.SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/attributes", response_model=SeismicAttributesResponse)
async def get_seismic_attributes(dataset_id: str) -> SeismicAttributesResponse:
    """Per-trace computed seismic attributes, including the heuristic
    VSH/PHIE/SWE proxies (see seismic_attributes.py for caveats).
    """
    try:
        return seismic_service.get_seismic_attribute_series(dataset_id)
    except seismic_service.SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{dataset_id}/export")
async def export_seismic(dataset_id: str) -> PlainTextResponse:
    """Export the per-trace computed seismic attributes as CSV."""
    try:
        content = seismic_service.export_seismic_attributes_csv(dataset_id)
    except seismic_service.SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return PlainTextResponse(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{dataset_id}_attributes.csv"'
        },
    )
