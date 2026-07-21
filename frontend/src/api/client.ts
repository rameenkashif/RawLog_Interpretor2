/**
 * client.ts
 * ---------
 * Thin fetch/axios wrapper around the FastAPI backend. Every function here
 * corresponds 1:1 to an endpoint in backend/app/routers/.
 */

import axios from "axios";
import type {
  AmplitudeSpectrumResponse,
  ChatStreamEvent,
  CoordinateCalibrationReportResponse,
  CrosslineSectionResponse,
  CrossplotResponse,
  DashboardSummary,
  DashboardUploadResponse,
  DashboardUploadStatusResponse,
  DensityMethod,
  InlineSectionResponse,
  NearestTraceResponse,
  RecalibrateResponse,
  SaveTiePointsRequest,
  SeismicAttributesResponse,
  SeismicSectionResponse,
  SeismicSummary,
  SeismicUploadResponse,
  SpectralDecompositionResponse,
  SpectralFrequencySliceResponse,
  SpectralMethod,
  SpectralPetroCorrelationResponse,
  SpectralSwtSliceResponse,
  SpectralTraceResponse,
  SswtPetroCorrelationResponse,
  SurveyInfoResponse,
  SwtWavelet,
  SyntheticSeismogramResponse,
  TiePointsResponse,
  TimeSliceResponse,
  WaveletMethod,
  WellCurvesResponse,
  WellSeismicTieBatchResponse,
  WellSeismicTieResponse,
  WellSummary,
  WellTieVizResponse,
  WellTraceOverrideRequest,
  WellTraceOverrideResponse,
  WellUploadResponse,
  WellZonesResponse,
  WellZoneTieMapResponse,
} from "./types";

// In dev, Vite proxies /wells, /dashboard, /chat to the FastAPI backend (see vite.config.ts).
// In production, set VITE_API_BASE_URL to the deployed backend origin.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

const http = axios.create({ baseURL: BASE_URL });

export async function uploadWells(files: File[]): Promise<WellUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const { data } = await http.post<WellUploadResponse>("/wells/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listWells(): Promise<WellSummary[]> {
  const { data } = await http.get<WellSummary[]>("/wells");
  return data;
}

export async function getWellCurves(
  wellId: string,
): Promise<WellCurvesResponse> {
  const { data } = await http.get<WellCurvesResponse>(
    `/wells/${wellId}/curves`,
  );
  return data;
}

export async function getWellZones(wellId: string): Promise<WellZonesResponse> {
  const { data } = await http.get<WellZonesResponse>(`/wells/${wellId}/zones`);
  return data;
}

export async function getCrossplot(
  wellId: string,
  x: string,
  y: string,
  color?: string | null,
): Promise<CrossplotResponse> {
  const { data } = await http.get<CrossplotResponse>(
    `/wells/${wellId}/crossplot`,
    {
      params: { x, y, color: color || undefined },
    },
  );
  return data;
}

export async function getDashboardSummary(): Promise<DashboardSummary> {
  const { data } = await http.get<DashboardSummary>("/dashboard/summary");
  return data;
}

/** Combined well + seismic upload -- both are required. The well is
 * processed immediately; seismic/tie/synthetic/spectral processing runs in
 * the background, poll getDashboardUploadStatus(well_id) for progress. */
export async function uploadDashboard(
  lasFile: File,
  segyFile: File,
): Promise<DashboardUploadResponse> {
  const form = new FormData();
  form.append("las_file", lasFile);
  form.append("segy_file", segyFile);
  const { data } = await http.post<DashboardUploadResponse>(
    "/dashboard/upload",
    form,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}

export async function getDashboardUploadStatus(
  wellId: string,
): Promise<DashboardUploadStatusResponse> {
  const { data } = await http.get<DashboardUploadStatusResponse>(
    `/dashboard/upload/${wellId}/status`,
  );
  return data;
}

export function getExportUrl(wellId: string, format: "csv" | "las"): string {
  return `${BASE_URL}/wells/${wellId}/export?format=${format}`;
}

// -----------------------------------------------------------------------------
// Seismic (SEG-Y)
// -----------------------------------------------------------------------------
export async function uploadSeismic(
  files: File[],
): Promise<SeismicUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const { data } = await http.post<SeismicUploadResponse>(
    "/seismic/upload",
    form,
    {
      headers: { "Content-Type": "multipart/form-data" },
    },
  );
  return data;
}

export async function listSeismic(): Promise<SeismicSummary[]> {
  const { data } = await http.get<SeismicSummary[]>("/seismic");
  return data;
}

export async function getSeismicSection(
  datasetId: string,
): Promise<SeismicSectionResponse> {
  const { data } = await http.get<SeismicSectionResponse>(
    `/seismic/${datasetId}/section`,
  );
  return data;
}

export async function getSeismicAttributes(
  datasetId: string,
): Promise<SeismicAttributesResponse> {
  const { data } = await http.get<SeismicAttributesResponse>(
    `/seismic/${datasetId}/attributes`,
  );
  return data;
}

export function getSeismicExportUrl(datasetId: string): string {
  return `${BASE_URL}/seismic/${datasetId}/export`;
}

// -----------------------------------------------------------------------------
// Well-to-Seismic Tie
// -----------------------------------------------------------------------------
export async function getWellSeismicTie(
  wellId: string,
  seismicDatasetId: string,
): Promise<WellSeismicTieResponse> {
  const { data } = await http.get<WellSeismicTieResponse>(`/tie/${wellId}`, {
    params: { seismic_dataset_id: seismicDatasetId },
  });
  return data;
}

export async function getAllWellSeismicTies(
  seismicDatasetId: string,
): Promise<WellSeismicTieBatchResponse> {
  const { data } = await http.get<WellSeismicTieBatchResponse>("/tie/all", {
    params: { seismic_dataset_id: seismicDatasetId },
  });
  return data;
}

// -----------------------------------------------------------------------------
// Seismic Visualization (direct SEG-Y inline/crossline/time-slice/spectrum)
// -----------------------------------------------------------------------------
export async function getSurveyInfo(): Promise<SurveyInfoResponse> {
  const { data } = await http.get<SurveyInfoResponse>("/api/seismic/survey-info");
  return data;
}

export async function getInlineSection(
  inlineNumber: number,
): Promise<InlineSectionResponse> {
  const { data } = await http.get<InlineSectionResponse>(
    `/api/seismic/inline/${inlineNumber}`,
  );
  return data;
}

export async function getCrosslineSection(
  crosslineNumber: number,
): Promise<CrosslineSectionResponse> {
  const { data } = await http.get<CrosslineSectionResponse>(
    `/api/seismic/crossline/${crosslineNumber}`,
  );
  return data;
}

export async function getTimeSlice(timeMs: number): Promise<TimeSliceResponse> {
  const { data } = await http.get<TimeSliceResponse>("/api/seismic/timeslice", {
    params: { time_ms: timeMs },
  });
  return data;
}

export async function getWellTieViz(
  wellId: string,
  waveletFreqHz?: number,
): Promise<WellTieVizResponse> {
  const { data } = await http.get<WellTieVizResponse>(
    `/api/seismic/well-tie/${wellId}`,
    { params: { wavelet_freq_hz: waveletFreqHz } },
  );
  return data;
}

export async function getWellZoneTieMap(power?: number): Promise<WellZoneTieMapResponse> {
  const { data } = await http.get<WellZoneTieMapResponse>("/api/seismic/well-zone-tie-map", {
    params: { power },
  });
  return data;
}

export async function getCoordinateCalibrationReport(): Promise<CoordinateCalibrationReportResponse> {
  const { data } = await http.get<CoordinateCalibrationReportResponse>("/api/seismic/coordinate-calibration");
  return data;
}

export async function recalibrateCoordinates(wellIds?: string[] | null): Promise<RecalibrateResponse> {
  const { data } = await http.post<RecalibrateResponse>("/api/seismic/coordinate-calibration/recalibrate", {
    well_ids: wellIds ?? null,
  });
  return data;
}

export async function listCoordinateOverrides(): Promise<WellTraceOverrideResponse[]> {
  const { data } = await http.get<WellTraceOverrideResponse[]>("/api/seismic/coordinate-calibration/overrides");
  return data;
}

export async function saveCoordinateOverride(
  wellId: string,
  body: WellTraceOverrideRequest,
): Promise<WellTraceOverrideResponse> {
  const { data } = await http.put<WellTraceOverrideResponse>(
    `/api/seismic/coordinate-calibration/overrides/${wellId}`,
    body,
  );
  return data;
}

export async function deleteCoordinateOverride(wellId: string): Promise<{ well_id: string; deleted: boolean }> {
  const { data } = await http.delete(`/api/seismic/coordinate-calibration/overrides/${wellId}`);
  return data;
}

export async function getAmplitudeSpectrum(
  inlineNumber?: number | null,
): Promise<AmplitudeSpectrumResponse> {
  const { data } = await http.get<AmplitudeSpectrumResponse>(
    "/api/seismic/spectrum",
    { params: { inline_number: inlineNumber ?? undefined } },
  );
  return data;
}

/** Full time x freq x position decomposition for an inline (heavier -- initial load/export). */
export async function getSpectralDecompositionInline(
  inlineNumber: number,
  method: SpectralMethod,
): Promise<SpectralDecompositionResponse> {
  const { data } = await http.get<SpectralDecompositionResponse>(
    `/api/seismic/spectral-decomp/inline/${inlineNumber}`,
    { params: { method } },
  );
  return data;
}

/** Single-frequency energy slice across an inline -- the fast path for a frequency slider. */
export async function getSpectralFrequencySlice(
  inlineNumber: number,
  method: SpectralMethod,
  frequencyHz: number,
): Promise<SpectralFrequencySliceResponse> {
  const { data } = await http.get<SpectralFrequencySliceResponse>(
    `/api/seismic/spectral-decomp/inline/${inlineNumber}`,
    { params: { method, frequency_hz: frequencyHz } },
  );
  return data;
}

export async function getSpectralDecompositionTrace(
  inlineNumber: number,
  crosslineNumber: number,
  method: SpectralMethod,
  includeSswt: boolean = false,
): Promise<SpectralTraceResponse> {
  const { data } = await http.get<SpectralTraceResponse>(
    "/api/seismic/spectral-decomp/trace",
    {
      params: {
        inline_number: inlineNumber,
        crossline_number: crosslineNumber,
        method,
        include_sswt: includeSswt,
      },
    },
  );
  return data;
}

/** SWT (Stationary Wavelet Transform) single-level envelope slice across an
 * inline -- the SWT equivalent of getSpectralFrequencySlice's fast path
 * (same section-position shape), addressed by discrete level (1-6) rather
 * than a continuous frequency. */
export async function getSpectralSwtSlice(
  inlineNumber: number,
  level: number,
  wavelet: SwtWavelet,
): Promise<SpectralSwtSliceResponse> {
  const { data } = await http.get<SpectralSwtSliceResponse>(
    `/api/seismic/spectral-decomp/inline/${inlineNumber}`,
    { params: { method: "swt", level, wavelet } },
  );
  return data;
}

export async function getSpectralPetroCorrelation(
  opts: { wellId?: string; allWells?: boolean; swtLevel?: number; wavelet?: SwtWavelet } = {},
): Promise<SpectralPetroCorrelationResponse> {
  const { data } = await http.get<SpectralPetroCorrelationResponse>(
    "/api/seismic/spectral-petro-correlation",
    {
      params: {
        well_id: opts.wellId,
        all_wells: opts.allWells,
        swt_level: opts.swtLevel,
        wavelet: opts.wavelet,
      },
    },
  );
  return data;
}

export async function getSswtPetroCorrelation(
  opts: { wellId?: string; allWells?: boolean; frequencyHz?: number } = {},
): Promise<SswtPetroCorrelationResponse> {
  const { data } = await http.get<SswtPetroCorrelationResponse>(
    "/api/seismic/spectral-petro-correlation-sswt",
    {
      params: {
        well_id: opts.wellId,
        all_wells: opts.allWells,
        frequency_hz: opts.frequencyHz,
      },
    },
  );
  return data;
}

// -----------------------------------------------------------------------------
// Synthetic Seismogram module
// -----------------------------------------------------------------------------
export async function generateSyntheticSeismogram(
  wellId: string,
  opts: {
    waveletMethod?: WaveletMethod;
    waveletFreqHz?: number;
    densityMethod?: DensityMethod;
    applySavedTie?: boolean;
    autoOptimizeTie?: boolean;
  } = {},
): Promise<SyntheticSeismogramResponse> {
  const { data } = await http.get<SyntheticSeismogramResponse>(
    `/api/synthetic/${wellId}/generate`,
    {
      params: {
        wavelet_method: opts.waveletMethod,
        wavelet_freq_hz: opts.waveletFreqHz,
        density_method: opts.densityMethod,
        apply_saved_tie: opts.applySavedTie,
        auto_optimize_tie: opts.autoOptimizeTie,
      },
    },
  );
  return data;
}

export async function getSyntheticNearestTrace(wellId: string): Promise<NearestTraceResponse> {
  const { data } = await http.get<NearestTraceResponse>(`/api/synthetic/${wellId}/nearest-trace`);
  return data;
}

export async function getSyntheticTiePoints(wellId: string): Promise<TiePointsResponse | null> {
  const { data } = await http.get<TiePointsResponse | null>(`/api/synthetic/${wellId}/tie`);
  return data;
}

export async function saveSyntheticTiePoints(
  wellId: string,
  body: SaveTiePointsRequest,
): Promise<TiePointsResponse> {
  const { data } = await http.put<TiePointsResponse>(`/api/synthetic/${wellId}/tie`, body);
  return data;
}

export async function deleteSyntheticTiePoints(wellId: string): Promise<{ well_id: string; deleted: boolean }> {
  const { data } = await http.delete<{ well_id: string; deleted: boolean }>(`/api/synthetic/${wellId}/tie`);
  return data;
}

export function getSyntheticExportUrl(
  wellId: string,
  opts: {
    waveletMethod?: WaveletMethod;
    waveletFreqHz?: number;
    densityMethod?: DensityMethod;
    autoOptimizeTie?: boolean;
  } = {},
): string {
  const params = new URLSearchParams();
  if (opts.waveletMethod) params.set("wavelet_method", opts.waveletMethod);
  if (opts.waveletFreqHz) params.set("wavelet_freq_hz", String(opts.waveletFreqHz));
  if (opts.densityMethod) params.set("density_method", opts.densityMethod);
  if (opts.autoOptimizeTie) params.set("auto_optimize_tie", String(opts.autoOptimizeTie));
  return `${BASE_URL}/api/synthetic/${wellId}/export?${params.toString()}`;
}

/**
 * Streams the /chat SSE endpoint, invoking `onEvent` for each parsed event
 * as it arrives. Returns a function that aborts the stream early if called.
 */
export function streamChat(
  message: string,
  wellId: string | null,
  conversationHistory: { role: string; content: string }[],
  onEvent: (event: ChatStreamEvent) => void,
  onError: (err: Error) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          well_id: wellId,
          conversation_history: conversationHistory,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(
          `Chat request failed: ${response.status} ${response.statusText}`,
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";

        for (const chunk of chunks) {
          const line = chunk.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice("data:".length).trim();
          if (!jsonStr) continue;
          try {
            const event = JSON.parse(jsonStr) as ChatStreamEvent;
            onEvent(event);
          } catch {
            // Ignore malformed SSE chunks rather than crashing the whole stream.
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        onError(err as Error);
      }
    }
  })();

  return () => controller.abort();
}