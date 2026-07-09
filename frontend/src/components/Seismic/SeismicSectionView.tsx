import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { getCrosslineSection, getInlineSection } from "@/api/client";
import type { SurveyInfoResponse } from "@/api/types";
import { colors } from "@/styles/tokens";

const AXIS_STYLE = {
  gridcolor: colors.gridLine,
  linecolor: colors.borderStrong,
  tickfont: { color: colors.inkMuted },
};

/**
 * Inline/crossline amplitude section viewer: pick a direction and a
 * line number (bounded by the survey's actual inline/crossline range),
 * rendered as a diverging red/blue Plotly heatmap -- same pattern as the
 * raw-amplitude section on the existing Seismic page, which already
 * handles this data shape and volume well.
 */
export default function SeismicSectionView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const [direction, setDirection] = useState<"inline" | "crossline">("inline");
  const [inlineNumber, setInlineNumber] = useState(surveyInfo.inline_min);
  const [crosslineNumber, setCrosslineNumber] = useState(surveyInfo.crossline_min);

  const inlineQuery = useQuery({
    queryKey: ["seismic-viz-inline", inlineNumber],
    queryFn: () => getInlineSection(inlineNumber),
    enabled: direction === "inline",
  });

  const crosslineQuery = useQuery({
    queryKey: ["seismic-viz-crossline", crosslineNumber],
    queryFn: () => getCrosslineSection(crosslineNumber),
    enabled: direction === "crossline",
  });

  const activeQuery = direction === "inline" ? inlineQuery : crosslineQuery;

  const figure = useMemo(() => {
    if (direction === "inline" && inlineQuery.data) {
      return buildFigure(
        inlineQuery.data.crossline_axis,
        inlineQuery.data.twt_axis_ms,
        inlineQuery.data.amplitude,
        "Crossline",
      );
    }
    if (direction === "crossline" && crosslineQuery.data) {
      return buildFigure(
        crosslineQuery.data.inline_axis,
        crosslineQuery.data.twt_axis_ms,
        crosslineQuery.data.amplitude,
        "Inline",
      );
    }
    return null;
  }, [direction, inlineQuery.data, crosslineQuery.data]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(["inline", "crossline"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDirection(d)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all capitalize ${
                direction === d
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {direction === "inline" ? (
          <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
            Inline
            <input
              type="range"
              min={surveyInfo.inline_min}
              max={surveyInfo.inline_max}
              value={inlineNumber}
              onChange={(e) => setInlineNumber(Number(e.target.value))}
              className="w-40 accent-accent"
            />
            <input
              type="number"
              min={surveyInfo.inline_min}
              max={surveyInfo.inline_max}
              value={inlineNumber}
              onChange={(e) => setInlineNumber(Number(e.target.value))}
              className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
            />
            <span className="text-ink-faint font-normal">
              ({surveyInfo.inline_min}-{surveyInfo.inline_max})
            </span>
          </label>
        ) : (
          <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
            Crossline
            <input
              type="range"
              min={surveyInfo.crossline_min}
              max={surveyInfo.crossline_max}
              value={crosslineNumber}
              onChange={(e) => setCrosslineNumber(Number(e.target.value))}
              className="w-40 accent-accent"
            />
            <input
              type="number"
              min={surveyInfo.crossline_min}
              max={surveyInfo.crossline_max}
              value={crosslineNumber}
              onChange={(e) => setCrosslineNumber(Number(e.target.value))}
              className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
            />
            <span className="text-ink-faint font-normal">
              ({surveyInfo.crossline_min}-{surveyInfo.crossline_max})
            </span>
          </label>
        )}
      </div>

      {activeQuery.isLoading && (
        <div className="h-[480px] rounded-xl bg-surface-sunken animate-pulse" />
      )}
      {activeQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Failed to load section: {(activeQuery.error as Error).message}
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
  positionAxis: number[],
  twtAxisMs: number[],
  amplitude: number[][], // already (n_samples, n_traces) from the API
  positionLabel: string,
): { data: Data[]; layout: Partial<Layout> } {
  let maxAbs = 1e-6;
  for (const row of amplitude) {
    for (const value of row) {
      const abs = Math.abs(value);
      if (abs > maxAbs) maxAbs = abs;
    }
  }

  const trace = {
    type: "heatmap",
    x: positionAxis,
    y: twtAxisMs,
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
    xaxis: { title: { text: positionLabel }, ...AXIS_STYLE },
    yaxis: { title: { text: "Two-Way Time (ms)" }, autorange: "reversed", ...AXIS_STYLE },
  };

  return { data: [trace], layout };
}
