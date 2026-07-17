"""
routers/tie.py
---------------
Well-to-seismic tie endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import WellSeismicTieBatchResponse, WellSeismicTieResponse
from app.services import tie_service
from app.services.seismic_service import SeismicDatasetNotFoundError
from app.well_seismic_tie import TieError

router = APIRouter(prefix="/tie", tags=["tie"])


@router.get("/all", response_model=WellSeismicTieBatchResponse)
async def get_all_well_seismic_ties(seismic_dataset_id: str) -> WellSeismicTieBatchResponse:
    """Batch well tie: runs the same DPTM + full-window frequency/polarity/
    shift search as GET /tie/{well_id} for every well in the repository
    against one seismic dataset, for a results table + coordinate map.
    Wells that fail to tie get a row with `error` set rather than being
    dropped."""
    try:
        return tie_service.get_all_well_ties(seismic_dataset_id)
    except SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{well_id}", response_model=WellSeismicTieResponse)
async def get_well_seismic_tie(well_id: str, seismic_dataset_id: str) -> WellSeismicTieResponse:
    """Well tie: converts DT+RHOB logs into a reflectivity series against
    the well's own DPTM (vendor-precomputed when available, else sonic-
    integration approximation -- see petrophysics.compute_dptm) and jointly
    searches Ricker wavelet frequency, polarity, and bulk time shift across
    the full seismic window (well_seismic_tie.search_best_tie_full_window)
    to maximize correlation against the nearest real seismic trace."""
    try:
        return tie_service.get_well_seismic_tie(well_id, seismic_dataset_id)
    except SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TieError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc