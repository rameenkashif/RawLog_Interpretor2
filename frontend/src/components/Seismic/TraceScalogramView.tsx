import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { AxiosError } from "axios";
import { getSpectralDecompositionTrace } from "@/api/client";
import type { SurveyInfoResponse } from "@/api/types";
import { useChartColors } from "@/styles/tokens";
import { buildFigure } from "./SpectralDecompView";

type ScalogramMethod = "cwt" | "sswt";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Single-trace time-vs-frequency scalogram, with a CWT/SSWT toggle --
 * distinct from the inline-section heatmaps above (which show one
 * frequency's energy across many traces); this shows every frequency's
 * energy for ONE trace, which is what a synchrosqueezed comparison needs.
 * SSWT (via ssqueezepy's ssq_cwt, see seismic_processor.py's
 * _decompose_sswt) reassigns each CWT coefficient to its true
 * instantaneous frequency, sharpening the same blurry frequency smearing
 * the plain CWT shows below for closely-spaced thin-bed signatures --
 * deliberately trace-level-only and opt-in (include_sswt) since it costs
 * roughly an order of magnitude more than the existing CWT per trace, so
 * it is NOT wired into the inline section view or any full-volume path.
 * Reuses SpectralDecompView's buildFigure heatmap builder unmodified
 * (confirmed the backend already returns SSWT amplitude in the same
 * (n_time, n_freq) shape convention as CWT/STFT/SWT, just with x=frequency
 * instead of x=crossline) rather than writing new rendering logic.
 */
export default function TraceScalogramView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const colors = useChartColors();
  const [inlineNumber, setInlineNumber] = useState(surveyInfo.inline_min);
  const [crosslineNumber, setCrosslineNumber] = useState(surveyInfo.crossline_min);
  const [scalogramMethod, setScalogramMethod] = useState<ScalogramMethod>("cwt");

  const isSswt = scalogramMethod === "sswt";

  // include_sswt is only sent when the user actually toggles to SSWT --
  // it costs roughly an order of magnitude more than the plain CWT call,
  // so switching inline/crossline while on the CWT toggle stays fast and
  // never pays that cost unasked.
  const traceQuery = useQuery({
    queryKey: ["seismic-viz-trace-scalogram", inlineNumber, crosslineNumber, isSswt],
    queryFn: () => getSpectralDecompositionTrace(inlineNumber, crosslineNumber, "cwt", isSswt),
  });

  const figure = traceQuery.data
    ? isSswt && traceQuery.data.sswt_freq_hz && traceQuery.data.sswt_amplitude
      ? buildFigure(traceQuery.data.sswt_freq_hz, traceQuery.data.time_ms, traceQuery.data.sswt_amplitude, colors, "Frequency (Hz)")
      : buildFigure(traceQuery.data.freq_hz, traceQuery.data.time_ms, traceQuery.data.energy, colors, "Frequency (Hz)")
    : null;

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div>
        <h3 className="text-sm font-semibold text-ink">Trace Scalogram — CWT vs SSWT</h3>
        <p className="text-xs text-ink-muted mt-0.5">
          Every frequency's energy for a single trace, not one frequency across the section above.
        </p>
      </div>

      <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
        SSWT (Synchrosqueezed Wavelet Transform) reassigns the CWT's coefficients to their true
        instantaneous frequency, sharpening the blurry smearing plain CWT shows when two thin-bed
        frequency signatures sit close together — compare the two here before deciding whether the
        extra compute cost (roughly an order of magnitude more than CWT per trace) is worth it for
        your interpretation. Not wired into the section view or the model feature pipeline yet.
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(["cwt", "sswt"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setScalogramMethod(m)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all uppercase ${
                scalogramMethod === m
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
          Inline
          <input
            type="number"
            min={surveyInfo.inline_min}
            max={surveyInfo.inline_max}
            value={inlineNumber}
            onChange={(e) => setInlineNumber(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
        </label>

        <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
          Crossline
          <input
            type="number"
            min={surveyInfo.crossline_min}
            max={surveyInfo.crossline_max}
            value={crosslineNumber}
            onChange={(e) => setCrosslineNumber(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
          <span className="text-ink-faint font-normal">
            ({surveyInfo.inline_min}-{surveyInfo.inline_max} / {surveyInfo.crossline_min}-
            {surveyInfo.crossline_max})
          </span>
        </label>

        {isSswt && traceQuery.data?.sswt_compute_ms != null && (
          <span className="text-xs font-semibold text-ink-faint bg-surface-sunken px-2.5 py-1 rounded-full">
            SSWT compute: {traceQuery.data.sswt_compute_ms.toFixed(0)} ms
          </span>
        )}
      </div>

      {traceQuery.isLoading && <div className="h-[420px] rounded-xl bg-surface-sunken animate-pulse" />}
      {traceQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load trace scalogram: {errorMessage(traceQuery.error)}
        </div>
      )}
      {figure && (
        <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
          <Plot
            data={figure.data}
            layout={figure.layout}
            style={{ width: "100%", height: "420px" }}
            config={{ displaylogo: false, responsive: true }}
          />
        </div>
      )}
    </div>
  );
}
