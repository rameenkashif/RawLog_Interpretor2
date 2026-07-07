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
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type ChatStreamEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; name: string; input: Record<string, unknown>; output: unknown }
  | { type: "done" }
  | { type: "error"; message: string };

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
