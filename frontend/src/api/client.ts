/**
 * client.ts
 * ---------
 * Thin fetch/axios wrapper around the FastAPI backend. Every function here
 * corresponds 1:1 to an endpoint in backend/app/routers/.
 */

import axios from "axios";
import type {
  ChatStreamEvent,
  CrossplotResponse,
  DashboardSummary,
  SeismicAttributesResponse,
  SeismicSectionResponse,
  SeismicSummary,
  SeismicUploadResponse,
  WellCurvesResponse,
  WellSummary,
  WellUploadResponse,
  WellZonesResponse,
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
