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
        None, description="Surface X coordinate (easting, m) from the LAS header, if present"
    )
    well_y: float | None = Field(
        None, description="Surface Y coordinate (northing, m) from the LAS header, if present"
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
    distance_m: float
    note: str = Field(
        ..., description="Simplifications/caveats in this tie (e.g. sonic-only depth-time conversion)"
    )


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


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str
