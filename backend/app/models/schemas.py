"""
schemas.py
----------
Pydantic request/response models for the FastAPI routers. Keeping these
separate from the routers themselves makes it easy to reuse them from the
Anthropic agent's tool layer (services/anthropic_agent.py) too.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
class CurveUnitInfo(BaseModel):
    curve: str
    declared_unit_raw: str | None = None
    resolved_unit: str | None = None
    inferred: bool = Field(..., description="True if resolved_unit came from value-range inference, not the LAS header")
    value_range_used: tuple[float, float] | None = None
    conversion_applied: bool = Field(False, description="True if the curve's values were converted in place (DT only, us/m -> us/ft)")


class WellSummary(BaseModel):
    well_id: str
    well_name: str
    start_depth: float
    stop_depth: float
    step: float
    n_samples: int
    footage_logged: float = Field(
        ..., description="stop_depth - start_depth, in metres"
    )
    avg_vsh: float | None = None
    avg_phie: float | None = None
    avg_swe: float | None = None
    net_pay_thickness: float | None = Field(
        None,
        description="Sum of step_depth over all samples classified as Pay (ZONES==1)",
    )
    null_counts: dict[str, int] = Field(default_factory=dict)
    well_x: float | None = Field(
        None, description="Surface X coordinate (easting, m), unit-standardized -- see coordinate_unit_detected"
    )
    well_y: float | None = Field(
        None, description="Surface Y coordinate (northing, m), unit-standardized -- see coordinate_unit_detected"
    )
    kb_m: float | None = Field(None, description="Kelly Bushing elevation, meters (unit-standardized)")
    td_m: float | None = Field(None, description="Total depth, meters (unit-standardized)")
    coordinate_unit_detected: str | None = Field(
        None,
        description=(
            "'feet' if X/Y/KB/TD were detected as feet (mislabeled '.m' in the LAS header) and "
            "converted, 'meters' if they were already consistent with STOP's units, None if "
            "unvalidated (TD or STOP missing)."
        ),
    )
    unit_conversion_applied: bool = Field(
        False, description="True if X/Y/KB/TD were converted from feet to meters on load"
    )
    td_stop_ratio: float | None = Field(
        None, description="TD/STOP ratio used to detect feet-vs-meters (~3.28 indicates feet)"
    )
    curve_units: list[CurveUnitInfo] = Field(
        default_factory=list,
        description="Per-curve unit provenance (declared vs. inferred from value range) for DT/RHOB/GR/NPHI",
    )


class WellUploadResponse(BaseModel):
    uploaded: list[WellSummary]
    errors: list[str] = Field(default_factory=list)


class CurvePoint(BaseModel):
    """One depth sample with every curve value present in the processed well."""

    values: dict[str, float | str | None]


class WellCurvesResponse(BaseModel):
    well_id: str
    curve_names: list[str]
    depth_step: float
    n_samples: int
    data: list[dict[str, Any]] = Field(
        ..., description="Row-oriented curve data, one dict per depth sample"
    )


class ZoneSummaryRow(BaseModel):
    zone_code: int
    zone_label: str
    thickness: float
    n_samples: int
    avg_phie: float | None = None
    avg_swe: float | None = None
    avg_vsh: float | None = None


class WellZonesResponse(BaseModel):
    well_id: str
    zones: list[ZoneSummaryRow]


class CrossplotPoint(BaseModel):
    x: float | None
    y: float | None
    color: float | str | None = None
    depth: float


class CrossplotResponse(BaseModel):
    well_id: str
    x_curve: str
    y_curve: str
    color_curve: str | None
    points: list[CrossplotPoint]


# -----------------------------------------------------------------------------
# Seismic (SEG-Y)
# -----------------------------------------------------------------------------
class SeismicSummary(BaseModel):
    dataset_id: str
    source_filename: str
    n_traces: int
    n_samples: int
    sample_interval_ms: float
    duration_ms: float
    avg_rms_amplitude: float | None = None
    avg_vsh_proxy: float | None = Field(
        None,
        description="Amplitude-based lithology-contrast proxy -- NOT a measured shale volume. See caveat in seismic_attributes.py.",
    )
    avg_phie_proxy: float | None = Field(
        None,
        description="Amplitude-based porosity-trend proxy -- NOT a measured porosity. See caveat in seismic_attributes.py.",
    )
    textual_header_encoding: str = Field(
        "cp037", description="Encoding the textual header actually decoded with, auto-detected (not assumed cp037)"
    )
    source_byte_locations: dict[str, int] = Field(
        default_factory=dict, description="Resolved SourceX/SourceY trace-header byte locations"
    )
    source_byte_locations_declared: dict[str, bool] = Field(
        default_factory=dict, description="True per field if the textual header declared it; False if defaulted to rev1 standard"
    )
    delay_recording_time_ms: float = Field(0.0, description="TWT of the first sample, read from trace headers")
    delay_recording_time_uniform: bool = Field(
        True, description="False if DelayRecordingTime varies across traces"
    )
    avg_swe_proxy: float | None = Field(
        None,
        description="Bright-spot-based hydrocarbon-indicator proxy -- NOT a measured water saturation. See caveat in seismic_attributes.py.",
    )


class SeismicUploadResponse(BaseModel):
    uploaded: list[SeismicSummary]
    errors: list[str] = Field(default_factory=list)


class SeismicSectionResponse(BaseModel):
    """Raw amplitude section for display, subsampled to keep the payload small."""

    dataset_id: str
    trace_indices: list[int]
    twt_axis_ms: list[float]
    amplitude: list[list[float]] = Field(
        ...,
        description="Shape (len(trace_indices), len(twt_axis_ms)), row-major by trace",
    )


class SeismicAttributesResponse(BaseModel):
    """Per-trace computed seismic attributes, including the heuristic VSH/PHIE/SWE proxies."""

    dataset_id: str
    trace_index: list[int]
    rms_amplitude: list[float]
    avg_envelope: list[float]
    dominant_freq_hz: list[float]
    vsh_seismic_proxy: list[float]
    phie_seismic_proxy: list[float]
    swe_seismic_proxy: list[float]


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
class DashboardSummary(BaseModel):
    n_wells: int
    total_footage: float
    avg_vsh: float | None
    avg_phie: float | None
    avg_swe: float | None
    wells: list[WellSummary]
    n_seismic_datasets: int = 0
    seismic_datasets: list[SeismicSummary] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Chat / Agent
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description='"user" or "assistant"')
    content: str


class ChatRequest(BaseModel):
    message: str
    well_id: str | None = None
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Record of tool calls the agent made, for transparency",
    )


# -----------------------------------------------------------------------------
# Dashboard combined upload (well + seismic, auto-processed in the background)
# -----------------------------------------------------------------------------
class DashboardUploadResponse(BaseModel):
    well_id: str
    well_summary: WellSummary
    status: str = Field("processing", description="Always 'processing' -- poll GET /dashboard/upload/{well_id}/status")


class DashboardUploadStatusResponse(BaseModel):
    well_id: str
    status: str = Field(..., description="'processing' | 'ready' | 'failed'")
    dataset_id: str | None = None
    segy_filename: str | None = None
    stale: bool = Field(
        False,
        description="True if segy_filename is no longer the currently active SEG-Y volume (a later upload replaced it)",
    )
    error: str | None = None

    tie_available: bool = False
    tie_error: str | None = None
    tie_correlation: float | None = None
    tie_boundary_pinned: bool | None = None
    tie_low_confidence: bool = Field(
        False, description="True if correlation < 0.3 or boundary_pinned -- a distinct, must-not-be-silent flag"
    )

    synthetic_available: bool = False
    synthetic_error: str | None = None
    synthetic_correlation: float | None = None
    synthetic_boundary_pinned: bool | None = None
    synthetic_low_confidence: bool = False

    spectral_available: bool = False
    spectral_dominant_freq_hz: float | None = None

    updated_at: str


class WellSeismicTieResponse(BaseModel):
    well_id: str
    dataset_id: str
    trace_index: int
    distance_m: float | None = None
    tie_method: str = Field(
        "manual_override",
        description=(
            "'nearest_trace' if the trace was picked by real well/seismic "
            "coordinates, 'manual_override' if a configured trace_index from "
            "tie_config.yaml was used instead (no coordinates available)."
        ),
    )
    inline: int | None = Field(None, description="Nearest trace's inline number, if the dataset carries geometry headers")
    crossline: int | None = Field(None, description="Nearest trace's crossline number, if the dataset carries geometry headers")
    best_freq_hz: float = Field(..., description="Winning Ricker wavelet frequency from the full-window frequency/polarity/shift search")
    polarity: int = Field(..., description="+1 or -1 -- winning polarity from the search")
    bulk_shift_ms: float = Field(..., description="Winning bulk time shift from the search, ms")
    correlation: float
    max_shift_ms: float = Field(100.0, description="Bulk-shift search range half-width used, ms")
    boundary_pinned: bool = Field(
        False, description="True if bulk_shift_ms landed within ~5% of max_shift_ms -- diagnostic of a spurious match, not a genuine tie"
    )
    n_used: int = Field(..., description="Number of samples actually overlapping the seismic window at the winning shift")
    time_ms: list[float] = Field(..., description="TWT axis (already shifted by bulk_shift_ms), covering the well's own reflectivity interval only")
    synthetic_amplitude: list[float] = Field(..., description="Normalized, polarity-applied synthetic, same length as time_ms")
    seismic_amplitude: list[float] = Field(..., description="Real seismic trace interpolated onto time_ms and normalized")
    reflectivity: list[float] = Field(..., description="Unshifted reflectivity series, same length as time_ms")
    geometry_warning: str | None = None


class WellSeismicTieRow(BaseModel):
    """One row of the all-wells tie summary table -- the batch analogue of
    WellSeismicTieResponse's scalar fields, for GET /tie/all's results table
    + map. error is set (with the other tie-result fields null) for a well
    that couldn't be tied (missing curves, no coordinates, etc.), rather
    than dropping it from the table silently."""

    well_id: str
    well_x: float | None = None
    well_y: float | None = None
    trace_index: int | None = None
    trace_x: float | None = None
    trace_y: float | None = None
    inline: int | None = None
    crossline: int | None = None
    distance_m: float | None = None
    tie_method: str | None = None
    best_freq_hz: float | None = None
    polarity: int | None = None
    bulk_shift_ms: float | None = None
    correlation: float | None = None
    boundary_pinned: bool | None = None
    error: str | None = None


class SurveyFootprintPoint(BaseModel):
    x: float
    y: float


class WellSeismicTieBatchResponse(BaseModel):
    dataset_id: str
    rows: list[WellSeismicTieRow]
    survey_footprint: list[SurveyFootprintPoint] = Field(
        default_factory=list,
        description="Downsampled trace X/Y coordinates for a background map footprint; empty if the dataset has no trace coordinates.",
    )
    warnings: list[str] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# Seismic Visualization (direct SEG-Y inline/crossline/time-slice/spectrum)
# -----------------------------------------------------------------------------
class SurveyInfoResponse(BaseModel):
    source_filename: str
    n_traces: int
    n_samples: int
    sample_interval_ms: float
    twt_start_ms: float
    twt_end_ms: float
    inline_min: int
    inline_max: int
    crossline_min: int
    crossline_max: int
    n_inlines: int
    n_crosslines: int
    best_time_ms: float
    textual_header_encoding: str = Field(
        ..., description="Encoding the textual header actually decoded with ('cp037'/'ascii'/'latin-1'), auto-detected"
    )
    byte_locations: dict[str, int] = Field(
        ..., description="Resolved trace-header byte locations (inline/crossline/source_x/source_y)"
    )
    byte_locations_declared: dict[str, bool] = Field(
        ..., description="True per field if the textual header declared it explicitly; False if defaulted to rev1 standard"
    )
    delay_recording_time_ms: float = Field(..., description="TWT of the first sample, read from trace headers")
    delay_recording_time_uniform: bool = Field(
        ..., description="False if DelayRecordingTime varies across traces (traces would not be directly comparable sample-for-sample)"
    )


class InlineSectionResponse(BaseModel):
    inline_number: int
    crossline_axis: list[int]
    twt_axis_ms: list[float]
    amplitude: list[list[float]] = Field(
        ..., description="Shape (n_samples, n_traces_in_line), row-major by sample"
    )


class CrosslineSectionResponse(BaseModel):
    crossline_number: int
    inline_axis: list[int]
    twt_axis_ms: list[float]
    amplitude: list[list[float]] = Field(
        ..., description="Shape (n_samples, n_traces_in_line), row-major by sample"
    )


class SectionWellLogCurve(BaseModel):
    well_id: str
    position_on_axis: int = Field(
        ..., description="This well's own tied crossline (inline section) or inline (crossline section)"
    )
    correlation: float = Field(..., description="Direct-tie correlation (direct_tie_service.resolve_direct_tie)")
    twt_ms: list[float]
    vsh: list[float | None]
    phie: list[float | None]
    swe: list[float | None]


class SectionWellLogsResponse(BaseModel):
    orientation: str = Field(..., description="'inline' or 'crossline' -- which section this is for")
    line_number: int
    wells: list[SectionWellLogCurve] = Field(
        ..., description="Every well with a usable direct tie, projected onto this section at its own position"
    )
    skipped_wells: list[dict] = Field(
        default_factory=list, description="Wells not drawn, each with a human-readable reason"
    )


class TimeSliceResponse(BaseModel):
    time_ms: float = Field(..., description="Actual sample time used (nearest-neighbor to requested_time_ms)")
    requested_time_ms: float
    inline_axis: list[int]
    crossline_axis: list[int]
    amplitude: list[list[float]] = Field(
        ..., description="Shape (n_inlines, n_crosslines); NaN for any gap in the grid"
    )


class WellTieVizResponse(BaseModel):
    well_id: str
    wavelet_freq_hz: float
    twt_ms: list[float]
    synthetic: list[float]
    real_trace: list[float]
    nearest_inline: int
    nearest_crossline: int
    distance_m: float | None = Field(None, description="None for a manual override (asserted directly, no residual)")
    tie_method: str = Field("calibrated_fit", description="'calibrated_fit' or 'manual_override'")
    note: str = Field(
        ..., description="Simplifications/caveats in this tie (e.g. sonic-only depth-time conversion)"
    )


class WellZoneTiePoint(BaseModel):
    well_id: str
    well_name: str
    inline: int
    crossline: int
    distance_m: float
    mean_vsh_pay: float = Field(..., description="Mean VSH over samples classified ZONES==Pay for this well")
    n_pay_samples: int


class WellZoneTieMapResponse(BaseModel):
    inline_axis: list[int]
    crossline_axis: list[int]
    predicted_vsh: list[list[float | None]] = Field(
        ..., description="(n_inlines x n_crosslines) IDW-interpolated VSH map; null where the grid has no trace"
    )
    wells: list[WellZoneTiePoint]
    warnings: list[str] = Field(
        default_factory=list, description="Wells skipped (no coordinates, CRS mismatch, no Pay-zone samples, etc.)"
    )
    method_note: str = Field(
        ..., description="Caveat: this is geometric IDW interpolation between wells, not a seismic inversion/ML prediction"
    )


class WellCalibrationReportItem(BaseModel):
    well_id: str
    well_name: str
    well_x: float
    well_y: float
    transformed_x: float = Field(..., description="Well coordinate mapped into seismic-survey coordinate space by the calibration")
    transformed_y: float
    nearest_inline: int
    nearest_crossline: int
    nearest_trace_distance_m: float
    is_extrapolated: bool = Field(
        ..., description="Coordinates fall outside both the calibration's fit range and the survey's own extent"
    )
    within_bin_tolerance: bool = Field(..., description="Nearest-trace residual is within ~2x the survey's bin spacing")
    trustworthy: bool = Field(..., description="within_bin_tolerance AND NOT is_extrapolated")
    used_in_calibration: bool = Field(..., description="This well was part of the calibration baseline's own fit")
    has_manual_override: bool
    override_inline: int | None = None
    override_crossline: int | None = None


class CoordinateCalibrationReportResponse(BaseModel):
    wells: list[WellCalibrationReportItem]
    method_note: str = Field(
        ...,
        description=(
            "Caveat: this is a 2-point-per-axis linear fit between well and seismic coordinates, "
            "not a real CRS reprojection -- only trust wells flagged trustworthy=true, or a well "
            "with a manual override."
        ),
    )


class WellTraceOverrideRequest(BaseModel):
    inline: int
    crossline: int
    note: str = ""


class WellTraceOverrideResponse(BaseModel):
    well_id: str
    inline: int
    crossline: int
    note: str = ""


class RecalibrateRequest(BaseModel):
    well_ids: list[str] | None = Field(
        None, description="Explicit subset of wells to calibrate from; omit to use every well with known coordinates"
    )


class RecalibrateResponse(BaseModel):
    well_ids_used: list[str]
    bin_spacing_m: float


class AmplitudeSpectrumResponse(BaseModel):
    inline_number: int | None
    n_traces_sampled: int
    freq_hz: list[float]
    amplitude: list[float]
    dominant_freq_hz: float
    bandwidth_hz: float
    snr_proxy: float | None = Field(
        None, description="Uncalibrated QC proxy: mean in-band amplitude / mean out-of-band amplitude"
    )


class SpectralDecompositionResponse(BaseModel):
    """Full time-frequency decomposition for every trace along an inline."""

    inline_number: int
    method: str = Field(..., description="'stft' or 'cwt'")
    crossline_axis: list[int]
    time_ms: list[float]
    freq_hz: list[float]
    nyquist_hz: float
    typical_band_hz: list[float] = Field(
        ..., description="[min, max] Hz of the typically-useful seismic band, for frontend default framing"
    )
    energy: list[list[list[float]]] = Field(
        ..., description="Shape (n_time, n_freq, n_traces_in_line)"
    )


class SpectralFrequencySliceResponse(BaseModel):
    """A single frequency's energy across an inline section -- the fast path
    for a frontend frequency slider; same section-position shape convention
    as InlineSectionResponse.amplitude so the same heatmap renderer works."""

    inline_number: int
    method: str = Field(..., description="'stft' or 'cwt'")
    requested_frequency_hz: float
    frequency_hz: float = Field(..., description="Actual frequency used (nearest available bin)")
    crossline_axis: list[int]
    time_ms: list[float]
    amplitude: list[list[float]] = Field(
        ..., description="Shape (n_time, n_traces_in_line), energy at frequency_hz"
    )


class SpectralTraceResponse(BaseModel):
    """Time-frequency decomposition for a single trace."""

    inline_number: int
    crossline_number: int
    method: str = Field(..., description="'stft' or 'cwt'")
    time_ms: list[float]
    freq_hz: list[float]
    nyquist_hz: float
    typical_band_hz: list[float]
    energy: list[list[float]] = Field(..., description="Shape (n_time, n_freq)")
    sswt_freq_hz: list[float] | None = Field(
        None,
        description="CWT + include_sswt=true only: Synchrosqueezed Wavelet Transform frequency axis (Hz), via ssqueezepy's ssq_cwt",
    )
    sswt_amplitude: list[list[float]] | None = Field(
        None, description="Shape (n_time, n_sswt_freq) -- |Tx| from ssq_cwt, the sharpened/reassigned CWT"
    )
    sswt_compute_ms: float | None = Field(
        None, description="Wall-clock time to compute the SSWT for this trace, ms -- see backend log for a side-by-side comparison against the plain CWT's own compute time"
    )


class SpectralSwtSliceResponse(BaseModel):
    """A single SWT detail level's envelope across an inline section -- the
    SWT equivalent of SpectralFrequencySliceResponse (same section-position
    shape convention, so the same heatmap renderer works unchanged). SWT
    has no continuous frequency axis to browse a "full volume" of, only a
    handful of discrete dyadic levels, so this fast-path shape is the only
    response SWT ever returns for an inline."""

    inline_number: int
    method: str = Field("swt", description="Always 'swt'")
    level: int = Field(..., description="Decomposition level actually used, 1-6")
    wavelet: str = Field(..., description="'sym8' or 'coif3'")
    band_hz: list[float] = Field(..., description="[lo, hi] Hz -- approximate dyadic band for this level")
    nyquist_hz: float
    crossline_axis: list[int]
    time_ms: list[float]
    amplitude: list[list[float]] = Field(
        ..., description="Shape (n_time, n_traces_in_line), Hilbert-envelope amplitude of the level's detail coefficients"
    )


class SpectralSwtTraceResponse(BaseModel):
    """SWT decomposition (all levels) for a single trace."""

    inline_number: int
    crossline_number: int
    method: str = Field("swt", description="Always 'swt'")
    wavelet: str = Field(..., description="'sym8' or 'coif3'")
    time_ms: list[float]
    levels: list[int] = Field(..., description="1-6")
    bands_hz: list[list[float]] = Field(..., description="Shape (n_level, 2), [lo, hi] Hz per level")
    nyquist_hz: float
    energy: list[list[float]] = Field(..., description="Shape (n_time, n_level)")


class PetroCorrelationPair(BaseModel):
    """One property's correlation against CWT (at the matched frequency)
    and SWT (at the requested level), over one well's tie interval."""

    cwt_r: float | None = Field(
        None, description="Pearson r between CWT amplitude and this property; None if too few valid samples or a constant series"
    )
    cwt_n: int = Field(..., description="Sample count used for the CWT correlation")
    swt_r: float | None = Field(None, description="Pearson r between SWT amplitude and this property")
    swt_n: int = Field(..., description="Sample count used for the SWT correlation")


class SpectralPetroCorrelationWellResult(BaseModel):
    well_id: str
    nearest_inline: int
    nearest_crossline: int
    distance_m: float | None = None
    tie_method: str = Field(..., description="'calibrated_fit' or 'manual_override'")
    vsh: PetroCorrelationPair
    phie: PetroCorrelationPair
    swe: PetroCorrelationPair
    low_sample_warning: bool = Field(
        ..., description="True if any correlation pair's sample count is below the reliability threshold (20)"
    )


class PetroCorrelationAverage(BaseModel):
    cwt_r: float | None = None
    swt_r: float | None = None
    n_wells: int = Field(..., description="Number of wells contributing a non-null correlation to this average")


class SpectralPetroCorrelationAverages(BaseModel):
    vsh: PetroCorrelationAverage
    phie: PetroCorrelationAverage
    swe: PetroCorrelationAverage


class SpectralPetroCorrelationResponse(BaseModel):
    mode: str = Field(..., description="'single' or 'all_wells'")
    swt_level: int = Field(..., description="1-6")
    swt_band_hz: list[float] = Field(..., description="[lo, hi] Hz -- this level's dyadic band")
    cwt_frequency_hz: float = Field(..., description="CWT frequency matched to the SWT band's center")
    wavelet: str = Field(..., description="'sym8' or 'coif3'")
    wells: list[SpectralPetroCorrelationWellResult]
    skipped_well_ids: list[str] = Field(
        default_factory=list,
        description="all_wells mode only: wells excluded (no resolvable tie, missing DT, no overlap, etc.)",
    )
    averages: SpectralPetroCorrelationAverages | None = Field(
        None, description="all_wells mode only: mean correlation per property across contributing wells"
    )


class SswtCorrelationPair(BaseModel):
    """One property's correlation against CWT and SSWT, both sampled at
    the same matched frequency, over one well's tie interval."""

    cwt_r: float | None = Field(None, description="Pearson r between CWT amplitude and this property")
    cwt_n: int = Field(..., description="Sample count used for the CWT correlation")
    sswt_r: float | None = Field(None, description="Pearson r between SSWT amplitude and this property")
    sswt_n: int = Field(..., description="Sample count used for the SSWT correlation")


class SswtCorrelationScatter(BaseModel):
    """The raw, per-sample paired series each SswtCorrelationPair's Pearson
    r is computed from, for a crossplot -- only populated in 'single' well
    mode (see get_sswt_correlation); omitted for 'all_wells' rows to avoid
    ballooning that response with per-well raw arrays."""

    depth_m: list[float] = Field(..., description="Depth of each sample, for hover labels/color-by")
    vsh: list[float | None]
    phie: list[float | None]
    swe: list[float | None]
    cwt_amplitude: list[float]
    sswt_amplitude: list[float]


class SswtPetroCorrelationWellResult(BaseModel):
    well_id: str
    nearest_inline: int
    nearest_crossline: int
    distance_m: float | None = None
    tie_method: str = Field(..., description="'calibrated_fit', 'manual_override', or 'direct_unvalidated'")
    vsh: SswtCorrelationPair
    phie: SswtCorrelationPair
    swe: SswtCorrelationPair
    low_sample_warning: bool = Field(
        ..., description="True if any correlation pair's sample count is below the reliability threshold (20)"
    )
    scatter: SswtCorrelationScatter | None = Field(
        None, description="Raw paired samples for a crossplot -- 'single' well mode only"
    )


class SswtCorrelationAverage(BaseModel):
    cwt_r: float | None = None
    sswt_r: float | None = None
    n_wells: int = Field(..., description="Number of wells contributing a non-null correlation to this average")


class SswtPetroCorrelationAverages(BaseModel):
    vsh: SswtCorrelationAverage
    phie: SswtCorrelationAverage
    swe: SswtCorrelationAverage


class SswtPetroCorrelationResponse(BaseModel):
    mode: str = Field(..., description="'single' or 'all_wells'")
    requested_frequency_hz: float
    cwt_frequency_hz: float = Field(..., description="Nearest available CWT frequency bin to requested_frequency_hz")
    sswt_frequency_hz: float = Field(..., description="Nearest available SSWT frequency bin to requested_frequency_hz")
    nyquist_hz: float
    wells: list[SswtPetroCorrelationWellResult]
    skipped_well_ids: list[str] = Field(
        default_factory=list,
        description="all_wells mode only: wells excluded (no resolvable tie, missing DT, no overlap, etc.)",
    )
    averages: SswtPetroCorrelationAverages | None = Field(
        None, description="all_wells mode only: mean correlation per property across contributing wells"
    )


# -----------------------------------------------------------------------------
# Spectral property prediction (multi-frequency CWT/SSWT -> VSH/PHIE/SWE,
# validated with leave-one-well-out cross-validation -- POINT-SOURCE
# validation only, not a volume-wide prediction. See
# spectral_property_prediction_service.py.
# -----------------------------------------------------------------------------
class SpectralPropertyExcludedWell(BaseModel):
    well_id: str
    reason: str = Field(..., description="Why this well was excluded from training -- never silently dropped")


class SpectralPropertyWellResult(BaseModel):
    well_id: str
    r2: float | None = Field(None, description="Held-out R^2 for this well when it was the leave-one-out fold")
    n_samples: int


class SpectralPropertyFeatureImportance(BaseModel):
    frequency_hz: float
    importance: float = Field(..., description="RandomForest feature_importances_ for this frequency bin")


class SpectralPropertyMethodResult(BaseModel):
    loocv_r2: float | None = Field(
        None, description="Pooled leave-one-well-out R^2 across all held-out predictions -- the headline validation score"
    )
    n_wells_used: int
    per_well: list[SpectralPropertyWellResult]
    feature_importance: list[SpectralPropertyFeatureImportance] = Field(
        ...,
        description="From a separate model fit on ALL usable wells -- in-sample, for interpretation only, NOT the validation score above",
    )


class SpectralPropertyModelResponse(BaseModel):
    status: str = Field(..., description="'validated' or 'insufficient_data' -- a first-class, explicit outcome, never a fabricated score")
    message: str | None = Field(None, description="Set when status='insufficient_data', explaining why and what would help")
    eligible_well_ids: list[str]
    excluded_wells: list[SpectralPropertyExcludedWell]
    n_wells_used: int
    results: dict[str, dict[str, SpectralPropertyMethodResult | None]] | None = Field(
        None,
        description="{'vsh'|'phie'|'swe': {'sswt'|'cwt': result_or_null}} -- null per (property, method) if too few wells had enough valid samples for that specific property, even when status='validated' overall",
    )


# -----------------------------------------------------------------------------
# Synthetic Seismogram module (/api/synthetic/*)
# -----------------------------------------------------------------------------
class WellHeaderQc(BaseModel):
    well_x: float | None
    well_y: float | None
    kb_m: float | None
    td_m: float | None
    coordinate_unit_detected: str | None = Field(
        None, description="'feet' if converted, 'meters' if already consistent, None if unvalidated"
    )
    unit_conversion_applied: bool
    td_stop_ratio: float | None


class GardnerCoefficients(BaseModel):
    a: float
    b: float
    calibrated: bool = Field(..., description="True if fit against this field's real RHOB, False if generic defaults")


class TiePointModel(BaseModel):
    md_m: float
    time_shift_ms: float


class DatumCheckModel(BaseModel):
    delay_ms: float
    implied_depth_m: float = Field(..., description="Depth implied by the delay at a plausible average overburden velocity")
    logged_top_depth_m: float
    relative_error: float
    avg_velocity_m_s: float
    plausible: bool = Field(..., description="False if implied_depth_m is wildly off from the logged interval's top")


class SyntheticSeismogramResponse(BaseModel):
    well_id: str
    well_header: WellHeaderQc
    vertical_assumption_note: str
    time_depth_note: str
    density_method: str = Field(..., description="'rhob', 'gardner', or 'rock_physics'")
    density_note: str
    gardner_coefficients: GardnerCoefficients | None = None
    nearest_inline: int
    nearest_crossline: int
    distance_m: float | None = Field(
        None, description="Distance to the tied trace, meters. None for a manual override (asserted directly, no residual)."
    )
    tie_method: str = Field("calibrated_fit", description="'calibrated_fit' or 'manual_override' -- see coordinate_calibration_service.py")
    depth_m: list[float]
    twt_ms: list[float]
    acoustic_impedance: list[float]
    reflectivity_depth_m: list[float]
    reflectivity: list[float]
    reflectivity_twt_ms: list[float]
    washout_depth_m: list[float]
    washout_flag: list[bool] = Field(
        ..., description="Soft QC proxy (NPHI-RHOB crossover / DT spikes) -- not a real caliper substitute"
    )
    wavelet_method: str = Field(..., description="'statistical' or 'ricker'")
    wavelet_freq_hz: float
    wavelet_t_ms: list[float]
    wavelet_amplitude: list[float]
    wavelet_spectrum_freq_hz: list[float]
    wavelet_spectrum_amplitude: list[float]
    wavelet_spectrum_phase_deg: list[float]
    seismic_twt_ms: list[float]
    synthetic: list[float]
    shifted_synthetic: list[float]
    real_trace: list[float]
    trace_spectrum_freq_hz: list[float]
    real_trace_spectrum_amplitude: list[float]
    synthetic_spectrum_amplitude: list[float]
    best_shift_ms: float
    correlation: float
    max_shift_ms: float = Field(..., description="Bulk-shift search range half-width used, ms")
    boundary_pinned: bool = Field(
        ..., description="True if best_shift_ms landed within ~5% of max_shift_ms -- diagnostic of a spurious match, not a genuine tie"
    )
    polarity: int = Field(1, description="+1 (normal) or -1 (reversed) -- always 1 unless auto_optimize_tie found -1 better")
    auto_optimize_tie: bool = Field(
        False, description="True if wavelet frequency (ricker only) and polarity were searched, not fixed to the request"
    )
    tie_search_note: str | None = Field(
        None, description="Set when auto_optimize_tie=true: which combination won and how many were tried"
    )
    datum_check: DatumCheckModel
    applied_tie_points: list[TiePointModel]


class SaveTiePointsRequest(BaseModel):
    points: list[TiePointModel]
    wavelet_method: str = "statistical"
    wavelet_freq_hz: float = 25.0


class TiePointsResponse(BaseModel):
    well_id: str
    points: list[TiePointModel]
    wavelet_method: str
    wavelet_freq_hz: float
    segy_filename: str | None = None


class NearestTraceResponse(BaseModel):
    well_id: str
    trace_index: int
    inline: int
    crossline: int
    distance_m: float | None = Field(None, description="None for a manual override (asserted directly, no residual)")
    tie_method: str = Field("calibrated_fit", description="'calibrated_fit' or 'manual_override'")


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str
