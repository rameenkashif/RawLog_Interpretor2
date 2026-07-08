"""
routers/tie.py
---------------
Well-to-seismic tie endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import WellSeismicTieResponse
from app.services import tie_service
from app.services.seismic_service import SeismicDatasetNotFoundError
from app.well_seismic_tie import TieError

router = APIRouter(prefix="/tie", tags=["tie"])


@router.get("/{well_id}", response_model=WellSeismicTieResponse)
async def get_well_seismic_tie(well_id: str, seismic_dataset_id: str) -> WellSeismicTieResponse:
    """Synthetic-seismogram-based well tie: converts DT+RHOB logs into a
    synthetic trace and correlates it against the nearest real seismic trace.
    This is a real geophysical calculation (Ricker wavelet synthetic, sonic
    depth-time integration) -- not the amplitude-heuristic proxy."""
    try:
        return tie_service.get_well_seismic_tie(well_id, seismic_dataset_id)
    except SeismicDatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TieError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc