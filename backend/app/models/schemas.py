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
    twt_ms: list[float]
    synthetic: list[float]
    shifted_synthetic: list[float]
    real_trace: list[float]
    best_shift_ms: float
    correlation: float
    max_shift_ms: float = Field(300.0, description="Bulk-shift search range half-width used, ms")
    boundary_pinned: bool = Field(
        False, description="True if best_shift_ms landed within ~5% of max_shift_ms -- diagnostic of a spurious match, not a genuine tie"
    )
    geometry_warning: str | None = None

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
    best_shift_ms: float
    correlation: float
    max_shift_ms: float = Field(..., description="Bulk-shift search range half-width used, ms")
    boundary_pinned: bool = Field(
        ..., description="True if best_shift_ms landed within ~5% of max_shift_ms -- diagnostic of a spurious match, not a genuine tie"
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
