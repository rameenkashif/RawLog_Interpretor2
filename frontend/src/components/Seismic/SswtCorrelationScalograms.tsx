import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { AxiosError } from "axios";
import { getSpectralDecompositionTrace } from "@/api/client";
import { useChartColors } from "@/styles/tokens";
import { buildFigure } from "./SpectralDecompView";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Side-by-side CWT and SSWT time-frequency energy images at the well's
 * own tied inline/crossline -- unlike TraceScalogramView.tsx (a manually
 * entered inline/crossline, CWT/SSWT toggled one at a time), this is
 * anchored to whichever well is selected in the correlation view above,
 * shown together for a direct visual comparison. Reuses SpectralDecompView's
 * buildFigure heatmap builder and the existing include_sswt trace endpoint
 * unmodified -- no new backend computation, just a paired rendering of data
 * TraceScalogramView.tsx already knows how to fetch/draw one at a time.
 */
export default function SswtCorrelationScalograms({
  inlineNumber,
  crosslineNumber,
}: {
  inlineNumber: number;
  crosslineNumber: number;
}) {
  const colors = useChartColors();

  const traceQuery = useQuery({
    queryKey: ["sswt-correlation-scalogram", inlineNumber, crosslineNumber],
    queryFn: () => getSpectralDecompositionTrace(inlineNumber, crosslineNumber, "cwt", true),
  });

  const data = traceQuery.data;
  const cwtFigure = data ? buildFigure(data.freq_hz, data.time_ms, data.energy, colors, "Frequency (Hz)") : null;
  const sswtFigure =
    data && data.sswt_freq_hz && data.sswt_amplitude
      ? buildFigure(data.sswt_freq_hz, data.time_ms, data.sswt_amplitude, colors, "Frequency (Hz)")
      : null;

  return (
    <div className="space-y-2">
      <p className="text-xs text-ink-faint">
        Same trace (inline {inlineNumber} / crossline {crosslineNumber}, this well's tied location),
        every frequency's energy over time -- CWT on the left, SSWT's reassigned, sharpened
        version on the right.
      </p>

      {traceQuery.isLoading && <div className="h-[340px] rounded-xl bg-surface-sunken animate-pulse" />}
      {traceQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load scalograms: {errorMessage(traceQuery.error)}
        </div>
      )}

      {cwtFigure && sswtFigure && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
            <p className="text-xs font-semibold text-ink-muted px-1 pb-1">CWT</p>
            <Plot
              data={cwtFigure.data}
              layout={cwtFigure.layout}
              style={{ width: "100%", height: "320px" }}
              config={{ displaylogo: false, responsive: true }}
            />
          </div>
          <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
            <p className="text-xs font-semibold text-ink-muted px-1 pb-1">
              SSWT
              {data?.sswt_compute_ms != null && (
                <span className="ml-2 font-normal text-ink-faint">
                  ({data.sswt_compute_ms.toFixed(0)} ms compute)
                </span>
              )}
            </p>
            <Plot
              data={sswtFigure.data}
              layout={sswtFigure.layout}
              style={{ width: "100%", height: "320px" }}
              config={{ displaylogo: false, responsive: true }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
