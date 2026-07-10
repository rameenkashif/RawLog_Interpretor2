"""
routers/seismic_viz.py
------------------------
"Seismic Visualization" endpoints: inline/crossline sections, time slices,
well ties, and amplitude spectra read directly off the raw SEG-Y volume in
backend/data/seismic_raw/ (see app/services/seismic_processor.py).

Deliberately a separate router/file from routers/seismic.py, which serves
the *upload* pipeline (multiple named datasets, stored attributes, export)
via app/segy_loader.py + app/seismic_repository.py -- this module instead
opens the raw file directly to get at inline/crossline geometry that
pipeline never stores, and only ever serves a single active volume.
Mounted at a different prefix (/api/seismic) so the two don't collide.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    AmplitudeSpectrumResponse,
    CoordinateCalibrationReportResponse,
    CrosslineSectionResponse,
    InlineSectionResponse,
    RecalibrateRequest,
    RecalibrateResponse,
    SpectralDecompositionResponse,
    SpectralFrequencySliceResponse,
    SpectralTraceResponse,
    SurveyInfoResponse,
    TimeSliceResponse,
    WellCalibrationReportItem,
    WellTieVizResponse,
    WellTraceOverrideRequest,
    WellTraceOverrideResponse,
    WellZoneTieMapResponse,
)
from app.coordinate_calibration import CoordinateCalibrationError
from app.coordinate_tie_override_repository import WellTraceOverride, get_coordinate_tie_override_repository
from app.services import coordinate_calibration_service as ccs
from app.services import seismic_processor as sp
from app.services import well_zone_tie_service as wzt
from app.services.well_service import WellNotFoundError

router = APIRouter(prefix="/api/seismic", tags=["seismic-viz"])


def _handle(exc: Exception):
    if isinstance(exc, (WellNotFoundError, sp.SegyFileNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, wzt.WellZoneTieError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, (ccs.UnresolvedCoordinateError, CoordinateCalibrationError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, sp.SegyVolumeError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/survey-info", response_model=SurveyInfoResponse)
async def survey_info() -> SurveyInfoResponse:
    """Geometry summary (inline/crossline range, sample interval, time
    range, trace count) so the frontend can bound its sliders on load."""
    try:
        volume = sp.get_segy_volume()
        return SurveyInfoResponse(**vars(volume.survey_info()))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/inline/{inline_number}", response_model=InlineSectionResponse)
async def inline_section(inline_number: int) -> InlineSectionResponse:
    try:
        volume = sp.get_segy_volume()
        return InlineSectionResponse(**volume.get_inline_section(inline_number))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/crossline/{crossline_number}", response_model=CrosslineSectionResponse)
async def crossline_section(crossline_number: int) -> CrosslineSectionResponse:
    try:
        volume = sp.get_segy_volume()
        return CrosslineSectionResponse(**volume.get_crossline_section(crossline_number))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/timeslice", response_model=TimeSliceResponse)
async def time_slice(time_ms: float = Query(..., description="Requested TWT in ms; clamps to nearest sample")) -> TimeSliceResponse:
    try:
        volume = sp.get_segy_volume()
        return TimeSliceResponse(**volume.get_time_slice(time_ms))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/well-tie/{well_id}", response_model=WellTieVizResponse)
async def well_tie(
    well_id: str,
    wavelet_freq_hz: float = Query(25.0, gt=0, description="Ricker wavelet dominant frequency, Hz"),
) -> WellTieVizResponse:
    try:
        volume = sp.get_segy_volume()
        return WellTieVizResponse(**volume.get_well_tie(well_id, wavelet_freq_hz=wavelet_freq_hz))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/well-zone-tie-map", response_model=WellZoneTieMapResponse)
async def well_zone_tie_map(
    power: float = Query(2.0, gt=0, description="Inverse-distance-weighting power (higher = more locally-dominated by the nearest well)"),
) -> WellZoneTieMapResponse:
    """'Well-Seismic Tie' map: every well's Pay-zone mean VSH, tied to the
    survey via real coordinates and spatially interpolated (IDW) across
    the full inline/crossline grid -- see well_zone_tie_service for the
    important caveat that this is geometric interpolation, not a seismic
    inversion."""
    try:
        return WellZoneTieMapResponse(**wzt.compute_well_zone_tie_map(power=power))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/coordinate-calibration", response_model=CoordinateCalibrationReportResponse)
async def coordinate_calibration_report() -> CoordinateCalibrationReportResponse:
    """Diagnostic report for every well with known coordinates: the
    per-axis well<->seismic calibration's estimate, residual-vs-bin-
    spacing validation, extrapolation flag, and manual override status --
    see coordinate_calibration_service.py. NOT a seismic inversion or CRS
    reprojection; only wells flagged trustworthy (or with a manual
    override) should be used for downstream tie/prediction workflows."""
    try:
        volume = sp.get_segy_volume()
        reports = ccs.get_calibration_report(volume)
        return CoordinateCalibrationReportResponse(
            wells=[WellCalibrationReportItem(**vars(r)) for r in reports],
            method_note=(
                "Per-axis linear fit (X_seismic = a*X_well + b, Y_seismic = c*Y_well + d) between "
                "well and seismic coordinates, calibrated from the wells' own coordinate extent -- "
                "NOT a real CRS reprojection (no known CRS/EPSG exists for either dataset). Only "
                "trust a well flagged trustworthy=true, or one with a manual override; treat any "
                "other well's tie as unresolved."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.post("/coordinate-calibration/recalibrate", response_model=RecalibrateResponse)
async def recalibrate(body: RecalibrateRequest) -> RecalibrateResponse:
    """Explicitly (re)fit the calibration baseline -- from every well with
    known coordinates if well_ids is omitted, or from a curated subset
    (e.g. excluding a well known to be bad) if given. This is the real
    fix path when the calibration itself looks wrong, vs. a manual
    tie-point override for a single problem well."""
    try:
        volume = sp.get_segy_volume()
        _cal, well_ids_used, bin_spacing_m = ccs.fit_and_store_calibration(volume, well_ids=body.well_ids)
        return RecalibrateResponse(well_ids_used=well_ids_used, bin_spacing_m=bin_spacing_m)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/coordinate-calibration/overrides", response_model=list[WellTraceOverrideResponse])
async def list_coordinate_overrides() -> list[WellTraceOverrideResponse]:
    repo = get_coordinate_tie_override_repository()
    return [WellTraceOverrideResponse(**vars(o)) for o in repo.list_overrides()]


@router.put("/coordinate-calibration/overrides/{well_id}", response_model=WellTraceOverrideResponse)
async def save_coordinate_override(well_id: str, body: WellTraceOverrideRequest) -> WellTraceOverrideResponse:
    """Manual well->trace tie-point override -- the real fix path for a
    well the calibration can't resolve with confidence (fix #5): once
    saved, this takes priority over the calibrated fit everywhere the
    well needs to be located on the seismic survey."""
    repo = get_coordinate_tie_override_repository()
    override = WellTraceOverride(well_id=well_id, inline=body.inline, crossline=body.crossline, note=body.note)
    repo.save_override(override)
    return WellTraceOverrideResponse(**vars(override))


@router.delete("/coordinate-calibration/overrides/{well_id}")
async def delete_coordinate_override(well_id: str) -> dict:
    repo = get_coordinate_tie_override_repository()
    deleted = repo.delete_override(well_id)
    return {"well_id": well_id, "deleted": deleted}


@router.get("/spectrum", response_model=AmplitudeSpectrumResponse)
async def spectrum(
    inline_number: int | None = Query(None, description="Restrict to one inline; omit to sample across the whole volume"),
) -> AmplitudeSpectrumResponse:
    try:
        volume = sp.get_segy_volume()
        return AmplitudeSpectrumResponse(**volume.get_amplitude_spectrum(inline_number=inline_number))
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get(
    "/spectral-decomp/inline/{inline_number}",
    response_model=SpectralDecompositionResponse | SpectralFrequencySliceResponse,
)
async def spectral_decomp_inline(
    inline_number: int,
    method: str = Query("stft", description="'stft' or 'cwt'"),
    frequency_hz: float | None = Query(
        None,
        description=(
            "If given, return just this frequency's energy across the section (fast path for "
            "a frontend slider). If omitted, return the full time x freq x position volume "
            "(heavier -- initial load or export)."
        ),
    ),
) -> SpectralDecompositionResponse | SpectralFrequencySliceResponse:
    try:
        volume = sp.get_segy_volume()
        result = volume.get_spectral_decomposition_inline(
            inline_number, method=method, frequency_hz=frequency_hz
        )
        if frequency_hz is None:
            return SpectralDecompositionResponse(**result)
        return SpectralFrequencySliceResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/spectral-decomp/trace", response_model=SpectralTraceResponse)
async def spectral_decomp_trace(
    inline_number: int,
    crossline_number: int,
    method: str = Query("stft", description="'stft' or 'cwt'"),
) -> SpectralTraceResponse:
    try:
        volume = sp.get_segy_volume()
        return SpectralTraceResponse(
            **volume.get_spectral_decomposition_trace(inline_number, crossline_number, method=method)
        )
    except Exception as exc:  # noqa: BLE001
        _handle(exc)
