/**
 * types.ts
 * --------
 * TypeScript mirrors of backend/app/models/schemas.py. Keep these in sync
 * with the Pydantic models -- FastAPI's OpenAPI schema could also be used
 * to codegen these later, but for now they're hand-maintained since the
 * schema is small and stable.
 */

export interface WellSummary {
  well_id: string;
  well_name: string;
  start_depth: number;
  stop_depth: number;
  step: number;
  n_samples: number;
  footage_logged: number;
  avg_vsh: number | null;
  avg_phie: number | null;
  avg_swe: number | null;
  net_pay_thickness: number | null;
  null_counts: Record<string, number>;
  well_x: number | null;
  well_y: number | null;
}

export interface WellUploadResponse {
  uploaded: WellSummary[];
  errors: string[];
}

export interface WellCurvesResponse {
  well_id: string;
  curve_names: string[];
  depth_step: number;
  n_samples: number;
  data: Record<string, number | string | null>[];
}

export interface ZoneSummaryRow {
  zone_code: number;
  zone_label: string;
  thickness: number;
  n_samples: number;
  avg_phie: number | null;
  avg_swe: number | null;
  avg_vsh: number | null;
}

export interface WellZonesResponse {
  well_id: string;
  zones: ZoneSummaryRow[];
}

export interface CrossplotPoint {
  x: number | null;
  y: number | null;
  color: number | string | null;
  depth: number;
}

export interface CrossplotResponse {
  well_id: string;
  x_curve: string;
  y_curve: string;
  color_curve: string | null;
  points: CrossplotPoint[];
}

export interface DashboardSummary {
  n_wells: number;
  total_footage: number;
  avg_vsh: number | null;
  avg_phie: number | null;
  avg_swe: number | null;
  wells: WellSummary[];
  n_seismic_datasets: number;
  seismic_datasets: SeismicSummary[];
}

// -----------------------------------------------------------------------------
// Seismic (SEG-Y)
// -----------------------------------------------------------------------------
export interface SeismicSummary {
  dataset_id: string;
  source_filename: string;
  n_traces: number;
  n_samples: number;
  sample_interval_ms: number;
  duration_ms: number;
  avg_rms_amplitude: number | null;
  /** Uncalibrated amplitude-based lithology-contrast proxy -- NOT a measured shale volume. */
  avg_vsh_proxy: number | null;
  /** Uncalibrated amplitude-based porosity-trend proxy -- NOT a measured porosity. */
  avg_phie_proxy: number | null;
  /** Uncalibrated bright-spot hydrocarbon-indicator proxy -- NOT a measured water saturation. */
  avg_swe_proxy: number | null;
}

export interface SeismicUploadResponse {
  uploaded: SeismicSummary[];
  errors: string[];
}

export interface SeismicSectionResponse {
  dataset_id: string;
  trace_indices: number[];
  twt_axis_ms: number[];
  /** Shape (trace_indices.length, twt_axis_ms.length) */
  amplitude: number[][];
}

export interface SeismicAttributesResponse {
  dataset_id: string;
  trace_index: number[];
  rms_amplitude: number[];
  avg_envelope: number[];
  dominant_freq_hz: number[];
  vsh_seismic_proxy: number[];
  phie_seismic_proxy: number[];
  swe_seismic_proxy: number[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type ChatStreamEvent =
  | { type: "text_delta"; text: string }
  | {
      type: "tool_call";
      name: string;
      input: Record<string, unknown>;
      output: unknown;
    }
  | { type: "done" }
  | { type: "error"; message: string };

export interface WellSeismicTieResponse {
  well_id: string;
  dataset_id: string;
  trace_index: number;
  distance_m: number | null;
  tie_method: "nearest_trace" | "manual_override";
  twt_ms: number[];
  synthetic: number[];
  shifted_synthetic: number[];
  real_trace: number[];
  best_shift_ms: number;
  correlation: number;
  geometry_warning: string | null;
}

// -----------------------------------------------------------------------------
// Seismic Visualization (direct SEG-Y inline/crossline/time-slice/spectrum)
// -----------------------------------------------------------------------------
export interface SurveyInfoResponse {
  source_filename: string;
  n_traces: number;
  n_samples: number;
  sample_interval_ms: number;
  twt_start_ms: number;
  twt_end_ms: number;
  inline_min: number;
  inline_max: number;
  crossline_min: number;
  crossline_max: number;
  n_inlines: number;
  n_crosslines: number;
}

export interface InlineSectionResponse {
  inline_number: number;
  crossline_axis: number[];
  twt_axis_ms: number[];
  amplitude: number[][]; // shape (n_samples, n_traces_in_line)
}

export interface CrosslineSectionResponse {
  crossline_number: number;
  inline_axis: number[];
  twt_axis_ms: number[];
  amplitude: number[][]; // shape (n_samples, n_traces_in_line)
}

export interface TimeSliceResponse {
  time_ms: number;
  requested_time_ms: number;
  inline_axis: number[];
  crossline_axis: number[];
  amplitude: number[][]; // shape (n_inlines, n_crosslines)
}

export interface WellTieVizResponse {
  well_id: string;
  wavelet_freq_hz: number;
  twt_ms: number[];
  synthetic: number[];
  real_trace: number[];
  nearest_inline: number;
  nearest_crossline: number;
  distance_m: number;
  note: string;
}

export interface AmplitudeSpectrumResponse {
  inline_number: number | null;
  n_traces_sampled: number;
  freq_hz: number[];
  amplitude: number[];
  dominant_freq_hz: number;
  bandwidth_hz: number;
  snr_proxy: number | null;
}

export type SpectralMethod = "stft" | "cwt";

export interface SpectralDecompositionResponse {
  inline_number: number;
  method: SpectralMethod;
  crossline_axis: number[];
  time_ms: number[];
  freq_hz: number[];
  nyquist_hz: number;
  typical_band_hz: [number, number];
  energy: number[][][]; // shape (n_time, n_freq, n_traces_in_line)
}

export interface SpectralFrequencySliceResponse {
  inline_number: number;
  method: SpectralMethod;
  requested_frequency_hz: number;
  frequency_hz: number;
  crossline_axis: number[];
  time_ms: number[];
  amplitude: number[][]; // shape (n_time, n_traces_in_line), same convention as InlineSectionResponse
}

export interface SpectralTraceResponse {
  inline_number: number;
  crossline_number: number;
  method: SpectralMethod;
  time_ms: number[];
  freq_hz: number[];
  nyquist_hz: number;
  typical_band_hz: [number, number];
  energy: number[][]; // shape (n_time, n_freq)
}

export const CURVE_NAMES = [
  "DEPT",
  "GR",
  "RESISTIVITY",
  "RHOB",
  "NPHI",
  "DT",
  "VSH",
  "PHIT",
  "PHIE",
  "PHIE_DN",
  "SWE",
  "PERM_TIXIER",
  "CORE_PERM_PRED",
  "VVOLC",
  "ZONES",
  "DPTM",
] as const;

export type CurveName = (typeof CURVE_NAMES)[number];
