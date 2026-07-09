import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { getTimeSlice } from "@/api/client";
import type { SurveyInfoResponse } from "@/api/types";
import { colors } from "@/styles/tokens";

const AXIS_STYLE = {
  gridcolor: colors.gridLine,
  linecolor: colors.borderStrong,
  tickfont: { color: colors.inkMuted },
};

/**
 * Map-view amplitude time slice: a fixed two-way-time cut across the
 * inline x crossline grid, as a Plotly heatmap. Time slider bounded by
 * the survey's actual time range/sample interval (survey-info), same
 * diverging red/blue colormap convention as the section views.
 */
export default function TimeSliceView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const [timeMs, setTimeMs] = useState(surveyInfo.twt_start_ms);

  const query = useQuery({
    queryKey: ["seismic-viz-timeslice", timeMs],
    queryFn: () => getTimeSlice(timeMs),
  });

  const figure = useMemo(() => {
    if (!query.data) return null;
    return buildFigure(query.data.crossline_axis, query.data.inline_axis, query.data.amplitude);
  }, [query.data]);

  return (
    <div className="space-y-3">
      <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
        Two-Way Time (ms)
        <input
          type="range"
          min={surveyInfo.twt_start_ms}
          max={surveyInfo.twt_end_ms}
          step={surveyInfo.sample_interval_ms}
          value={timeMs}
          onChange={(e) => setTimeMs(Number(e.target.value))}
          className="w-56 accent-accent"
        />
        <input
          type="number"
          min={surveyInfo.twt_start_ms}
          max={surveyInfo.twt_end_ms}
          step={surveyInfo.sample_interval_ms}
          value={timeMs}
          onChange={(e) => setTimeMs(Number(e.target.value))}
          className="w-24 text-xs border border-border-strong rounded-lg px-2 py-1"
        />
        <span className="text-ink-faint font-normal">
          ({surveyInfo.twt_start_ms}-{surveyInfo.twt_end_ms} ms, step {surveyInfo.sample_interval_ms} ms)
        </span>
        {query.data && (
          <span className="text-ink-faint font-normal">
            actual sample: {query.data.time_ms} ms
          </span>
        )}
      </label>

      {query.isLoading && <div className="h-[480px] rounded-xl bg-surface-sunken animate-pulse" />}
      {query.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Failed to load time slice: {(query.error as Error).message}
        </div>
      )}
      {figure && (
        <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
          <Plot
            data={figure.data}
            layout={figure.layout}
            style={{ width: "100%", height: "480px" }}
            config={{ displaylogo: false, responsive: true }}
          />
        </div>
      )}
    </div>
  );
}

function buildFigure(
  crosslineAxis: number[],
  inlineAxis: number[],
  amplitude: number[][], // (n_inlines, n_crosslines)
): { data: Data[]; layout: Partial<Layout> } {
  let maxAbs = 1e-6;
  for (const row of amplitude) {
    for (const value of row) {
      if (Number.isFinite(value)) {
        const abs = Math.abs(value);
        if (abs > maxAbs) maxAbs = abs;
      }
    }
  }

  const trace = {
    type: "heatmap",
    x: crosslineAxis,
    y: inlineAxis,
    z: amplitude,
    zmid: 0,
    zmin: -maxAbs,
    zmax: maxAbs,
    colorscale: "RdBu",
    reversescale: true,
    colorbar: { title: { text: "Amplitude", font: { size: 10 } }, tickfont: { size: 9 } },
  } as Data;

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: "Crossline" }, ...AXIS_STYLE },
    yaxis: { title: { text: "Inline" }, ...AXIS_STYLE },
  };

  return { data: [trace], layout };
}
