"""
routers/sweet_spot.py
-------------------------
Sweet-Spot prediction endpoints, under /api/sweet-spot. See
services/sweet_spot_prediction_service.py's module docstring for the full
pipeline (V11-style 2-stage cascade: phase-rotated well tie, 55-attribute
feature engine curated to 22, dynamic calibration, LOGO-CV model
selection, facies modulation, and a genuine blind-well holdout).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app import well_seismic_tie as wst
from app.models.schemas import SweetSpotRegionPredictionResponse, SweetSpotTrainingResponse
from app.services import seismic_processor as sp
from app.services import sweet_spot_prediction_service as ssp
from app.services.well_service import WellNotFoundError

router = APIRouter(prefix="/api/sweet-spot", tags=["sweet-spot"])


def _handle(exc: Exception):
    if isinstance(exc, (WellNotFoundError, sp.SegyFileNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (ssp.SweetSpotPredictionError, sp.SegyVolumeError, wst.TieError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def _to_training_response(trained: dict) -> SweetSpotTrainingResponse:
    if trained["status"] != "validated":
        return SweetSpotTrainingResponse(
            status=trained["status"], message=trained["message"], blind_well_id=trained["blind_well_id"],
            training_well_ids=trained["training_well_ids"], excluded_wells=trained["excluded_wells"],
            feature_names=trained["feature_names"], results=None,
        )
    return SweetSpotTrainingResponse(
        status="validated", message=None, blind_well_id=trained["blind_well_id"],
        training_well_ids=trained["training_well_ids"], excluded_wells=trained["excluded_wells"],
        feature_names=trained["feature_names"], results=trained["blind_results"],
    )


@router.get("/train", response_model=SweetSpotTrainingResponse)
async def train_sweet_spot(
    blind_well_id: str = Query(
        ssp.DEFAULT_BLIND_WELL_ID,
        description="Well to hold out entirely and validate against -- never used in training-data "
        "construction, feature selection, or any model training/selection step.",
    ),
    refresh: bool = Query(False, description="Force retraining instead of returning a cached result."),
) -> SweetSpotTrainingResponse:
    try:
        trained = ssp.get_or_train_cascade(blind_well_id, refresh=refresh)
        return _to_training_response(trained)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/region-prediction", response_model=SweetSpotRegionPredictionResponse)
async def sweet_spot_region_prediction(
    inline_min: int,
    inline_max: int,
    crossline_min: int,
    crossline_max: int,
    properties: str = Query(
        ..., description="Comma-separated property names, e.g. 'gr,vsh,phie,swe' (ai/dt/phit/gr/rhob/vsh/phie/swe)."
    ),
    blind_well_id: str = Query(
        ssp.DEFAULT_BLIND_WELL_ID, description="Which trained cascade to use (trains on demand if not cached)."
    ),
) -> SweetSpotRegionPredictionResponse:
    try:
        property_names = [p.strip() for p in properties.split(",") if p.strip()]
        if not property_names:
            raise ssp.SweetSpotPredictionError("No properties requested.")
        result = ssp.predict_region(
            (inline_min, inline_max), (crossline_min, crossline_max), property_names, blind_well_id
        )
        return SweetSpotRegionPredictionResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)
