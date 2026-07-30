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

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from app.models.schemas import (
    AmplitudeSpectrumResponse,
    CheckshotStatusResponse,
    CheckshotUploadResponse,
    CoordinateCalibrationReportResponse,
    CrosslineSectionResponse,
    InlineSectionResponse,
    PredictionGridSearchResponse,
    PredictionResponse,
    RecalibrateRequest,
    RecalibrateResponse,
    SectionWellLogsResponse,
    SpectralDecompositionResponse,
    SpectralFrequencySliceResponse,
    SpectralPetroCorrelationResponse,
    SpectralPropertyModelResponse,
    SpectralSwtSliceResponse,
    SpectralSwtTraceResponse,
    SpectralTraceResponse,
    SswtPetroCorrelationResponse,
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
from app.services import checkshot_service
from app.services import coordinate_calibration_service as ccs
from app.services import prediction_pipeline_service as pps
from app.services import section_image_service as swi
from app.services import section_well_log_service as swl
from app.services import seismic_processor as sp
from app.services import spectral_petro_correlation_service as spc
from app.services import spectral_property_prediction_service as sppp
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


@router.get("/section-well-logs", response_model=SectionWellLogsResponse)
async def section_well_logs(
    orientation: str = Query(..., description="'inline' or 'crossline' -- which section this is for"),
    line_number: int = Query(..., description="The section's own inline/crossline number (bounds the time clip)"),
) -> SectionWellLogsResponse:
    """Every well's VSH/PHIE/SWE curve, converted to two-way time via the
    direct nearest-trace tie (section_well_log_service.py), for drawing as
    a log-on-section overlay next to the Inline/Crossline Section view."""
    try:
        return SectionWellLogsResponse(**swl.get_section_well_logs(orientation, line_number))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/section-image")
async def section_image(
    orientation: str = Query(..., description="'inline' or 'crossline'"),
    line_number: int = Query(..., description="The section's own inline/crossline number"),
) -> Response:
    """Static (Matplotlib) PNG rendering of the same section + well-log
    overlay section-well-logs/the interactive Plotly view show -- an
    additional, cleaner-looking option, not a replacement (see
    section_image_service.py)."""
    try:
        png_bytes = swi.render_section_image(orientation, line_number)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/prediction", response_model=PredictionResponse)
async def prediction(
    blind_well_id: str = Query(..., description="Well to hold out and predict blind"),
    target: str = Query(..., description="'vsh', 'phie', or 'swe'"),
) -> PredictionResponse:
    """Blind-well VSH/PHIE/SWE prediction -- the "improved pipeline"
    (PCA + instantaneous attrs + per-property best model, chosen by an
    actual in-app grid search, see prediction_pipeline_service.py's
    run_grid_search/get_best_config). The winning recipe isn't
    caller-selectable -- it's returned in model_config_description."""
    try:
        return PredictionResponse(**pps.get_prediction_result(blind_well_id, target))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/prediction-grid-search", response_model=PredictionGridSearchResponse)
async def prediction_grid_search(
    target: str = Query(..., description="'vsh', 'phie', or 'swe'"),
    force: bool = Query(
        False,
        description=(
            "Re-run the search even if a cached result exists (e.g. after wells/ties changed). "
            "A full search is expensive -- tens of seconds to a few minutes -- so this defaults "
            "to reusing the in-process cache, not re-searching on every call."
        ),
    ),
) -> PredictionGridSearchResponse:
    """The full 48-candidate leaderboard (spectrum x PCA option x
    instantaneous-attrs x model family) for `target`, scored by pooled
    leave-one-well-out R^2 -- see prediction_pipeline_service.py's
    run_grid_search. This is what actually picks the config every other
    /prediction* endpoint uses for that target."""
    try:
        return PredictionGridSearchResponse(**pps.get_grid_search_result(target, force=force))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.post("/checkshot/upload", response_model=CheckshotUploadResponse)
async def upload_checkshot(file: UploadFile = File(...)) -> CheckshotUploadResponse:
    """Upload a real checkshot / time-depth-survey workbook (.xlsx, one
    sheet per well, TWT(ms) + Depth(m) columns -- see checkshot_service.py
    for the exact layout expected). Every well-to-seismic tie resolved via
    direct_tie_service (Spectral Property Prediction, the Inline/
    Crossline Section well-log overlay, and the Prediction page) picks
    this up automatically on its next call -- no separate "apply" step.
    Grid-search-selected model configs are cached per target
    (prediction_pipeline_service._GRID_SEARCH_CACHE); re-run a search
    with force=true (see /prediction-grid-search) to pick up a tie that
    changed because of this upload."""
    try:
        content = await file.read()
        counts = checkshot_service.store_checkshot_workbook(content)
        return CheckshotUploadResponse(wells=counts)
    except checkshot_service.CheckshotValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/checkshot/status", response_model=CheckshotStatusResponse)
async def checkshot_status() -> CheckshotStatusResponse:
    """well_id -> number of stored checkshot points, for every well with
    any uploaded checkshot coverage. A well NOT in this list has no
    checkshot data and will use the full statistical tie search."""
    return CheckshotStatusResponse(wells=checkshot_service.get_checkshot_status())


@router.get("/prediction-image")
async def prediction_image(
    blind_well_id: str = Query(..., description="Well to hold out and predict blind"),
    target: str = Query(..., description="'vsh', 'phie', or 'swe'"),
) -> Response:
    """Static (Matplotlib) PNG: side-by-side TRUE vs. PREDICTED inline
    section for the blind well -- see prediction_pipeline_service.py's
    render_prediction_image."""
    try:
        png_bytes = pps.render_prediction_image(blind_well_id, target)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/prediction-loocv-heatmap")
async def prediction_loocv_heatmap() -> Response:
    """Static (Matplotlib) PNG: TRUE pooled leave-one-well-out R^2 (every
    well takes a turn held out) for VSH/PHIE/SWE, each under its own
    BEST_CONFIG model -- see prediction_pipeline_service.py's
    render_full_loocv_heatmap_image. Not scoped to any one blind well."""
    try:
        png_bytes = pps.render_full_loocv_heatmap_image()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/prediction-frequency-map")
async def prediction_frequency_map(
    blind_well_id: str = Query(..., description="Well whose inline to show the frequency map for"),
) -> Response:
    """Static (Matplotlib) PNG: amplitude-spectrum frequency map for the
    inline through the given well -- see prediction_pipeline_service.py's
    render_frequency_map_image."""
    try:
        png_bytes = pps.render_frequency_map_image(blind_well_id)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/prediction-inline-maps")
async def prediction_inline_maps(
    blind_well_id: str = Query(..., description="Well to hold out and predict blind"),
) -> Response:
    """Static (Matplotlib) PNG: real seismic vs. predicted VSH/PHIE/SWE
    painted across the WHOLE inline (not just the well's own location),
    each property under its own BEST_CONFIG model -- EXPLORATORY, see
    prediction_pipeline_service.py's render_property_inline_maps_image
    docstring for the R^2 caveat this renders directly into the image."""
    try:
        png_bytes = pps.render_property_inline_maps_image(blind_well_id)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    response_model=SpectralDecompositionResponse | SpectralFrequencySliceResponse | SpectralSwtSliceResponse,
)
async def spectral_decomp_inline(
    inline_number: int,
    method: str = Query("stft", description="'stft', 'cwt', or 'swt'"),
    frequency_hz: float | None = Query(
        None,
        description=(
            "STFT/CWT only. If given, return just this frequency's energy across the section "
            "(fast path for a frontend slider). If omitted, return the full time x freq x "
            "position volume (heavier -- initial load or export)."
        ),
    ),
    level: int | None = Query(
        None,
        description="SWT only. Decomposition level, 1-6 (default 3). Ignored for 'stft'/'cwt'.",
    ),
    wavelet: str = Query(
        sp.SWT_DEFAULT_WAVELET, description="SWT only. 'sym8' (Symlet-8, default) or 'coif3' (Coiflet-3)."
    ),
) -> SpectralDecompositionResponse | SpectralFrequencySliceResponse | SpectralSwtSliceResponse:
    try:
        volume = sp.get_segy_volume()
        result = volume.get_spectral_decomposition_inline(
            inline_number, method=method, frequency_hz=frequency_hz, level=level, wavelet=wavelet
        )
        if method.lower() == "swt":
            return SpectralSwtSliceResponse(**result)
        if frequency_hz is None:
            return SpectralDecompositionResponse(**result)
        return SpectralFrequencySliceResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get(
    "/spectral-decomp/trace",
    response_model=SpectralTraceResponse | SpectralSwtTraceResponse,
)
async def spectral_decomp_trace(
    inline_number: int,
    crossline_number: int,
    method: str = Query("stft", description="'stft', 'cwt', or 'swt'"),
    wavelet: str = Query(
        sp.SWT_DEFAULT_WAVELET, description="SWT only. 'sym8' (Symlet-8, default) or 'coif3' (Coiflet-3)."
    ),
    include_sswt: bool = Query(
        False,
        description=(
            "CWT only, ignored for 'stft'/'swt'. If true, also compute and return the "
            "Synchrosqueezed Wavelet Transform (SSWT) of this trace via ssqueezepy -- sharpens the "
            "plain CWT's time-frequency smearing, but costs roughly an order of magnitude more "
            "(see backend log); opt-in, additive to the existing CWT fields, not a replacement."
        ),
    ),
) -> SpectralTraceResponse | SpectralSwtTraceResponse:
    try:
        volume = sp.get_segy_volume()
        result = volume.get_spectral_decomposition_trace(
            inline_number, crossline_number, method=method, wavelet=wavelet, include_sswt=include_sswt
        )
        if method.lower() == "swt":
            return SpectralSwtTraceResponse(**result)
        return SpectralTraceResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/spectral-petro-correlation", response_model=SpectralPetroCorrelationResponse)
async def spectral_petro_correlation(
    well_id: str | None = Query(None, description="Required unless all_wells=true."),
    all_wells: bool = Query(
        False,
        description="Loop over every well with a resolvable tie and DT/petrophysical logs, plus an averaged summary. well_id is ignored if true.",
    ),
    swt_level: int = Query(
        sp.SWT_DEFAULT_LEVEL,
        description="SWT decomposition level, 1-6 (default 3) -- also fixes the matched CWT comparison frequency (this level's dyadic band center).",
    ),
    wavelet: str = Query(
        sp.SWT_DEFAULT_WAVELET, description="SWT only. 'sym8' (Symlet-8, default) or 'coif3' (Coiflet-3)."
    ),
) -> SpectralPetroCorrelationResponse:
    """"CWT vs SWT -- Petrophysical Correlation": at a matched frequency
    band (CWT sampled at the SWT level's own band-center frequency),
    Pearson-correlates each spectral method's amplitude against VSH/PHIE/
    SWE over a well's tie interval -- see spectral_petro_correlation_service
    for why this is a like-for-like comparison rather than CWT's adaptive
    peak frequency against a fixed SWT level."""
    try:
        result = spc.get_correlation(well_id=well_id, all_wells=all_wells, swt_level=swt_level, wavelet=wavelet)
        return SpectralPetroCorrelationResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/spectral-petro-correlation-sswt", response_model=SswtPetroCorrelationResponse)
async def spectral_petro_correlation_sswt(
    well_id: str | None = Query(None, description="Required unless all_wells=true."),
    all_wells: bool = Query(
        False,
        description="Loop over every well with a resolvable tie and DT/petrophysical logs, plus an averaged summary. well_id is ignored if true.",
    ),
    frequency_hz: float = Query(
        spc.DEFAULT_SSWT_COMPARISON_FREQUENCY_HZ,
        description="Comparison frequency, Hz -- CWT and SSWT are each independently snapped to their own nearest available frequency bin to this value.",
    ),
) -> SswtPetroCorrelationResponse:
    """"CWT vs SSWT -- Petrophysical Correlation": unlike CWT-vs-SWT (which
    matches CWT to a fixed SWT octave band), both CWT and SSWT have a
    continuous frequency axis, so both are snapped to their own nearest
    bin to the SAME requested frequency and Pearson-correlated against
    VSH/PHIE/SWE over a well's tie interval -- see
    spectral_petro_correlation_service.get_sswt_correlation. SSWT costs
    roughly an order of magnitude more than the plain CWT per trace (see
    seismic_processor._decompose_sswt), so this stays a per-well/per-trace
    comparison, same as the CWT-vs-SWT endpoint above."""
    try:
        result = spc.get_sswt_correlation(well_id=well_id, all_wells=all_wells, frequency_hz=frequency_hz)
        return SswtPetroCorrelationResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)


@router.get("/spectral-property-model", response_model=SpectralPropertyModelResponse)
async def spectral_property_model() -> SpectralPropertyModelResponse:
    """Multi-frequency CWT/SSWT amplitude -> VSH/PHIE/SWE prediction,
    validated with leave-one-well-out cross-validation across every well
    with a usable synthetic-seismogram tie (not a random depth-sample
    split -- see spectral_property_prediction_service for why).
    POINT-SOURCE validation only, not a volume-wide prediction -- a good
    loocv_r2 here is a prerequisite for, not the same as, a trustworthy
    spatial map. status='insufficient_data' (with results=null) is a
    first-class outcome when fewer than 2 wells have a usable tie,
    surfaced explicitly rather than as an error or a fabricated score."""
    try:
        result = sppp.get_property_models()
        return SpectralPropertyModelResponse(**result)
    except Exception as exc:  # noqa: BLE001
        _handle(exc)
