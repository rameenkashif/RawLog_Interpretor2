"""
routers/dashboard.py
---------------------
Aggregated multi-well statistics for the field-wide dashboard (section 4/6),
plus the combined well+seismic upload entry point that auto-processes both
in the background and feeds Wells/Dashboard, Seismic, and Synthetic
Seismogram pages from one action.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.las_loader import LasValidationError
from app.models.schemas import (
    DashboardSummary,
    DashboardUploadResponse,
    DashboardUploadStatusResponse,
)
from app.services import dashboard_upload_service, well_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary() -> DashboardSummary:
    """Field-wide summary: well count, total footage, average VSH/PHIE/SWE,
    and per-well summaries (used for the wells table + bar charts).
    """
    return well_service.get_dashboard_summary()


@router.post("/upload", response_model=DashboardUploadResponse)
async def dashboard_upload(
    background_tasks: BackgroundTasks,
    las_file: UploadFile = File(...),
    segy_file: UploadFile = File(...),
) -> DashboardUploadResponse:
    """Upload a well (LAS) and its corresponding seismic data (SEG-Y)
    together. The well is parsed and interpreted immediately (small file,
    fast); the SEG-Y is validated, becomes the active seismic volume, and
    is tied against the well -- along with a synthetic seismogram and a
    spectral summary -- as a background job, since that's the slow part
    (a ~75-80MB SEG-Y parse plus a full frequency/polarity/shift tie
    search). Poll GET /dashboard/upload/{well_id}/status for progress.
    """
    if not las_file.filename or not las_file.filename.lower().endswith(".las"):
        raise HTTPException(status_code=422, detail="las_file must be a .las file")
    if not segy_file.filename or not segy_file.filename.lower().endswith((".sgy", ".segy")):
        raise HTTPException(status_code=422, detail="segy_file must be a .sgy or .segy file")

    las_bytes = await las_file.read()
    try:
        well_summary = well_service.process_and_store_las_bytes(las_bytes, las_file.filename)
    except LasValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not dashboard_upload_service.seismic_deps_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "The well was processed, but seismic dependencies (segyio/scipy) are not "
                "installed, so seismic/tie/synthetic data cannot be added. Run "
                "`pip install -r requirements.txt` and restart the backend."
            ),
        )

    segy_bytes = await segy_file.read()
    run_token = dashboard_upload_service.start_upload(well_summary.well_id, segy_file.filename)
    background_tasks.add_task(
        dashboard_upload_service.run_upload_pipeline,
        well_summary.well_id,
        run_token,
        segy_bytes,
        segy_file.filename,
    )

    return DashboardUploadResponse(well_id=well_summary.well_id, well_summary=well_summary, status="processing")


@router.get("/upload/{well_id}/status", response_model=DashboardUploadStatusResponse)
async def dashboard_upload_status(well_id: str) -> DashboardUploadStatusResponse:
    """Poll the background pipeline's progress/result for a well uploaded
    via POST /dashboard/upload. A failed or low-confidence tie/synthetic
    result is never silently reported as normal -- it's surfaced through
    the explicit *_available/*_low_confidence/error fields even while
    status itself is 'ready' (status tracks whether the pipeline ran
    without crashing, not whether every sub-result was trustworthy).
    """
    from app.well_processing_cache_repository import get_well_processing_cache_repository

    repo = get_well_processing_cache_repository()
    record = repo.get(well_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No dashboard upload found for well '{well_id}'. Use POST /dashboard/upload first.",
        )

    return DashboardUploadStatusResponse(
        well_id=record.well_id,
        status=record.status,
        dataset_id=record.dataset_id,
        segy_filename=record.segy_filename,
        stale=dashboard_upload_service.is_active_volume_stale(record),
        error=record.error,
        tie_available=record.tie_available,
        tie_error=record.tie_error,
        tie_correlation=record.tie_correlation,
        tie_boundary_pinned=record.tie_boundary_pinned,
        tie_low_confidence=bool(record.tie_low_confidence),
        synthetic_available=record.synthetic_available,
        synthetic_error=record.synthetic_error,
        synthetic_correlation=record.synthetic_correlation,
        synthetic_boundary_pinned=record.synthetic_boundary_pinned,
        synthetic_low_confidence=bool(record.synthetic_low_confidence),
        spectral_available=record.spectral_available,
        spectral_dominant_freq_hz=record.spectral_dominant_freq_hz,
        updated_at=record.updated_at,
    )
