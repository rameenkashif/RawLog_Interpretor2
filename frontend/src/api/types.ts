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
  max_shift_ms: number;
  boundary_pinned: boolean;
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
  best_time_ms: number;
  textual_header_encoding: string;
  byte_locations: Record<string, number>;
  byte_locations_declared: Record<string, boolean>;
  delay_recording_time_ms: number;
  delay_recording_time_uniform: boolean;
}

export interface WellCalibrationReportItem {
  well_id: string;
  well_name: string;
  well_x: number;
  well_y: number;
  transformed_x: number;
  transformed_y: number;
  nearest_inline: number;
  nearest_crossline: number;
  nearest_trace_distance_m: number;
  is_extrapolated: boolean;
  within_bin_tolerance: boolean;
  trustworthy: boolean;
  used_in_calibration: boolean;
  has_manual_override: boolean;
  override_inline: number | null;
  override_crossline: number | null;
}

export interface CoordinateCalibrationReportResponse {
  wells: WellCalibrationReportItem[];
  method_note: string;
}

export interface WellTraceOverrideRequest {
  inline: number;
  crossline: number;
  note?: string;
}

export interface WellTraceOverrideResponse {
  well_id: string;
  inline: number;
  crossline: number;
  note: string;
}

export interface RecalibrateRequest {
  well_ids?: string[] | null;
}

export interface RecalibrateResponse {
  well_ids_used: string[];
  bin_spacing_m: number;
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
  distance_m: number | null;
  tie_method: "calibrated_fit" | "manual_override" | "direct_unvalidated";
  note: string;
}

export interface WellZoneTiePoint {
  well_id: string;
  well_name: string;
  inline: number;
  crossline: number;
  distance_m: number;
  mean_vsh_pay: number;
  n_pay_samples: number;
}

export interface WellZoneTieMapResponse {
  inline_axis: number[];
  crossline_axis: number[];
  predicted_vsh: (number | null)[][]; // shape (n_inlines, n_crosslines)
  wells: WellZoneTiePoint[];
  warnings: string[];
  method_note: string;
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

// -----------------------------------------------------------------------------
// Synthetic Seismogram module
// -----------------------------------------------------------------------------
export type DensityMethod = "rhob" | "gardner" | "rock_physics";
export type WaveletMethod = "statistical" | "ricker";

export interface WellHeaderQc {
  well_x: number | null;
  well_y: number | null;
  kb_m: number | null;
  td_m: number | null;
  coordinate_unit_detected: "feet" | "meters" | null;
  unit_conversion_applied: boolean;
  td_stop_ratio: number | null;
}

export interface GardnerCoefficients {
  a: number;
  b: number;
  calibrated: boolean;
}

export interface TiePointModel {
  md_m: number;
  time_shift_ms: number;
}

export interface DatumCheckModel {
  delay_ms: number;
  implied_depth_m: number;
  logged_top_depth_m: number;
  relative_error: number;
  avg_velocity_m_s: number;
  plausible: boolean;
}

export interface SyntheticSeismogramResponse {
  well_id: string;
  well_header: WellHeaderQc;
  vertical_assumption_note: string;
  time_depth_note: string;
  density_method: DensityMethod;
  density_note: string;
  gardner_coefficients: GardnerCoefficients | null;
  nearest_inline: number;
  nearest_crossline: number;
  distance_m: number | null;
  tie_method: "calibrated_fit" | "manual_override" | "direct_unvalidated";
  depth_m: number[];
  twt_ms: number[];
  acoustic_impedance: number[];
  reflectivity_depth_m: number[];
  reflectivity: number[];
  reflectivity_twt_ms: number[];
  washout_depth_m: number[];
  washout_flag: boolean[];
  wavelet_method: WaveletMethod;
  wavelet_freq_hz: number;
  wavelet_t_ms: number[];
  wavelet_amplitude: number[];
  wavelet_spectrum_freq_hz: number[];
  wavelet_spectrum_amplitude: number[];
  wavelet_spectrum_phase_deg: number[];
  seismic_twt_ms: number[];
  synthetic: number[];
  shifted_synthetic: number[];
  real_trace: number[];
  best_shift_ms: number;
  correlation: number;
  max_shift_ms: number;
  boundary_pinned: boolean;
  datum_check: DatumCheckModel;
  applied_tie_points: TiePointModel[];
}

export interface SaveTiePointsRequest {
  points: TiePointModel[];
  wavelet_method: WaveletMethod;
  wavelet_freq_hz: number;
}

export interface TiePointsResponse {
  well_id: string;
  points: TiePointModel[];
  wavelet_method: WaveletMethod;
  wavelet_freq_hz: number;
  segy_filename: string | null;
}

export interface NearestTraceResponse {
  well_id: string;
  trace_index: number;
  inline: number;
  crossline: number;
  distance_m: number | null;
  tie_method: "calibrated_fit" | "manual_override" | "direct_unvalidated";
}
