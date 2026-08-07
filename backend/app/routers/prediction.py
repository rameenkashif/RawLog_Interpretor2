"""
routers/prediction.py
-----------------------
Blind-well property prediction endpoint, under /api/prediction. See
services/blind_well_prediction_service.py's module docstring for the full
pipeline (neighborhood-expanded CWT/SSWT + instantaneous attributes,
leave-one-well-out-stable feature selection, a 4-base-learner stacking
ensemble, and a genuine blind-well holdout -- the held-out well never
participates in any training step).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import BlindWellPredictionResponse
from app.services import blind_well_prediction_service as bwp
from app.services import seismic_processor as sp
from app.services.well_service import WellNotFoundError

router = APIRouter(prefix="/api/prediction", tags=["prediction"])


def _handle(exc: Exception):
    if isinstance(exc, (WellNotFoundError, sp.SegyFileNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (bwp.BlindWellPredictionError, sp.SegyVolumeError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/blind-well", response_model=BlindWellPredictionResponse)
async def blind_well_prediction(
    blind_well_id: str = Query(
        bwp.DEFAULT_BLIND_WELL_ID,
        description="Well to hold out entirely and predict -- never used in neighborhood sampling, feature "
        "selection, or any model training.",
    ),
) -> BlindWellPredictionResponse:
    try:
        return BlindWellPredictionResponse(**bwp.run_blind_well_prediction(blind_well_id))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)
