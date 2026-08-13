import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { AxiosError } from "axios";
import { getSectionWellLogs, getSweetSpotRegionPrediction } from "@/api/client";
import type { SectionWellLogCurve, SurveyInfoResponse, SweetSpotPropertyName } from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";

export const SWEET_SPOT_PROPERTIES: { key: SweetSpotPropertyName; label: string; unit: string }[] = [
  { key: "gr", label: "GR", unit: "API" },
  { key: "vsh", label: "VSH", unit: "frac" },
  { key: "phie", label: "PHIE", unit: "frac" },
  { key: "swe", label: "SWE", unit: "frac" },
  { key: "ai", label: "AI", unit: "(m/s)(g/cc)" },
  { key: "dt", label: "DT", unit: "us/ft" },
  { key: "phit", label: "PHIT", unit: "frac" },
  { key: "rhob", label: "RHOB", unit: "g/cc" },
];

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function axisStyle(colors: ChartColors) {
  return { gridcolor: colors.gridLine, linecolor: colors.borderStrong, tickfont: { color: colors.inkMuted } };
}

/**
 * Sweet-spot section: pick an inline, and see a chosen property's
 * PREDICTED value (not raw seismic amplitude) painted across the whole
 * crossline range of that inline as a sequential-colorscale heatmap --
 * hovering anywhere reads out "XL {x} | TWT {y}ms: {PROPERTY} = {z}",
 * with each nearby well marked at its own tied position.
 */
export default function SweetSpotSectionView({
  surveyInfo,
  blindWellId,
  property,
  trained,
}: {
  surveyInfo: SurveyInfoResponse;
  blindWellId: string;
  property: SweetSpotPropertyName;
  trained: boolean;
}) {
  const colors = useChartColors();
  // Default to the SURVEY'S MIDDLE inline, not inline_min -- an edge
  // inline can never have a neighbor on both sides (see INLINE_PAD
  // below), so it would always show an empty/degraded section by
  // default even once trained.
  const [inlineNumber, setInlineNumber] = useState(
    Math.round((surveyInfo.inline_min + surveyInfo.inline_max) / 2),
  );

  const propertyMeta = SWEET_SPOT_PROPERTIES.find((p) => p.key === property)!;

  // Neighboring-inline gradient features (part of the curated 22) need
  // BOTH an inline+1 and inline-1 trace within the fetched region to be
  // non-NaN -- a single-inline-wide request can never satisfy that (its
  // only inline has no neighbor at all within the response), which would
  // silently NaN out every position. Fetch a small inline PAD around the
  // chosen line so the middle inline has real neighbors, then slice that
  // one inline's own columns back out below for display.
  const INLINE_PAD = 2;
  const inlineRangeQueried: [number, number] = [
    Math.max(surveyInfo.inline_min, inlineNumber - INLINE_PAD),
    Math.min(surveyInfo.inline_max, inlineNumber + INLINE_PAD),
  ];

  const regionQuery = useQuery({
    queryKey: ["sweet-spot-region", blindWellId, inlineNumber, property],
    queryFn: () =>
      getSweetSpotRegionPrediction(
        inlineRangeQueried,
        [surveyInfo.crossline_min, surveyInfo.crossline_max],
        [property],
        blindWellId,
      ),
    enabled: trained,
  });

  const wellLogsQuery = useQuery({
    queryKey: ["sweet-spot-section-well-logs", inlineNumber],
    queryFn: () => getSectionWellLogs("inline", inlineNumber),
    enabled: trained,
  });

  const figure = useMemo(() => {
    if (!regionQuery.data) return null;
    return buildFigure(regionQuery.data, inlineNumber, property, propertyMeta, colors, wellLogsQuery.data?.wells);
  }, [regionQuery.data, inlineNumber, property, propertyMeta, colors, wellLogsQuery.data]);

  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
        Inline (sweet spot)
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

      {!trained && (
        <div className="h-[480px] rounded-xl bg-surface-sunken flex items-center justify-center text-xs text-ink-faint">
          Train the models above to see predictions mapped onto a section.
        </div>
      )}

      {trained && regionQuery.isLoading && <div className="h-[480px] rounded-xl bg-surface-sunken animate-pulse" />}

      {trained && regionQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load prediction: {errorMessage(regionQuery.error)}
        </div>
      )}

      {trained && figure && (
        <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
          <Plot
            data={figure.data}
            layout={figure.layout}
            style={{ width: "100%", height: "480px" }}
            config={{ displaylogo: false, responsive: true }}
          />
          <p className="px-2 pb-1 pt-2 text-xs text-ink-faint">
            Hover the section to read the predicted {propertyMeta.label} at any point -- dashed markers show nearby
            wells at their own tied position.
          </p>
        </div>
      )}
    </div>
  );
}

function buildFigure(
  region: { inline_axis: number[]; crossline_axis: number[]; twt_axis_ms: number[]; predictions: Record<string, (number | null)[][]> },
  inlineNumber: number,
  property: SweetSpotPropertyName,
  propertyMeta: { label: string; unit: string },
  colors: ChartColors,
  wellLogs?: SectionWellLogCurve[],
): { data: Data[]; layout: Partial<Layout> } {
  const AXIS_STYLE = axisStyle(colors);
  const nXl = region.crossline_axis.length;
  // predictions[property] is (n_time, n_inline*n_crossline), columns
  // grouped by inline (crossline varies fastest within each inline's
  // block of nXl columns) -- the region was fetched wider than one
  // inline so neighbor-gradient features had real neighbors to use (see
  // the INLINE_PAD comment above), so slice back out just the requested
  // inline's own columns for this single-inline section view.
  const ilIdx = region.inline_axis.indexOf(inlineNumber);
  const full = region.predictions[property] ?? [];
  const z = ilIdx >= 0 ? full.map((row) => row.slice(ilIdx * nXl, (ilIdx + 1) * nXl)) : [];

  const trace = {
    type: "heatmap",
    x: region.crossline_axis,
    y: region.twt_axis_ms,
    z,
    colorscale: "Jet",
    colorbar: { title: { text: `${propertyMeta.label} (${propertyMeta.unit})`, font: { size: 10 } }, tickfont: { size: 9 } },
    hovertemplate: `XL %{x} | TWT %{y}ms: ${propertyMeta.label} = %{z:.2f} ${propertyMeta.unit}<extra></extra>`,
  } as Data;

  const markers: Data[] = [];
  if (wellLogs && region.twt_axis_ms.length > 0) {
    const yMin = Math.min(...region.twt_axis_ms);
    const yMax = Math.max(...region.twt_axis_ms);
    for (const well of wellLogs) {
      markers.push({
        type: "scatter",
        mode: "lines",
        x: [well.position_on_axis, well.position_on_axis],
        y: [yMin, yMax],
        line: { color: colors.ink, width: 1.5, dash: "dot" },
        hoverinfo: "skip",
        showlegend: false,
      } as Data);
      markers.push({
        type: "scatter",
        mode: "text",
        x: [well.position_on_axis],
        y: [yMin],
        text: [`${well.well_id} (r=${well.correlation.toFixed(2)})`],
        textposition: "top center",
        textfont: { size: 9, color: colors.ink },
        hoverinfo: "skip",
        showlegend: false,
      } as Data);
    }
  }

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: "Crossline" }, ...AXIS_STYLE },
    yaxis: { title: { text: "Two-Way Time (ms)" }, autorange: "reversed", ...AXIS_STYLE },
    showlegend: false,
  };

  return { data: [trace, ...markers], layout };
}
