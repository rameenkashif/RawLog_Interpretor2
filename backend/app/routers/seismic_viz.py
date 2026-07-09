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
    CrosslineSectionResponse,
    InlineSectionResponse,
    SpectralDecompositionResponse,
    SpectralFrequencySliceResponse,
    SpectralTraceResponse,
    SurveyInfoResponse,
    TimeSliceResponse,
    WellTieVizResponse,
)
from app.services import seismic_processor as sp
from app.services.well_service import WellNotFoundError

router = APIRouter(prefix="/api/seismic", tags=["seismic-viz"])


def _handle(exc: Exception):
    if isinstance(exc, (WellNotFoundError, sp.SegyFileNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
