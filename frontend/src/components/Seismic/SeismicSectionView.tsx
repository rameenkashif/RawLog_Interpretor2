import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { getCrosslineSection, getInlineSection, getSectionWellLogs } from "@/api/client";
import type { SectionWellLogCurve, SurveyInfoResponse } from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";

// Fixed, high-contrast colors for the log-on-section wiggles -- deliberately
// NOT theme-aware like the rest of this app's charts, since these need to
// stay legible against the heatmap's own fixed red/blue diverging scale in
// both light and dark mode.
const LOG_CURVE_COLORS = { vsh: "#FBBF24", phie: "#34D399", swe: "#F472B6" } as const;

function axisStyle(colors: ChartColors) {
  return {
    gridcolor: colors.gridLine,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  };
}

/**
 * Inline/crossline amplitude section viewer: pick a direction and a
 * line number (bounded by the survey's actual inline/crossline range),
 * rendered as a diverging red/blue Plotly heatmap -- same pattern as the
 * raw-amplitude section on the existing Seismic page, which already
 * handles this data shape and volume well.
 */
export default function SeismicSectionView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const colors = useChartColors();
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
  const lineNumber = direction === "inline" ? inlineNumber : crosslineNumber;

  // Every well's VSH/PHIE/SWE, projected onto this section at the well's
  // own tied position (not necessarily exactly on this line -- see
  // section_well_log_service.py's docstring) -- refetches whenever the
  // direction or line number changes since position_on_axis and the
  // time-window clip both depend on them.
  const wellLogsQuery = useQuery({
    queryKey: ["seismic-viz-section-well-logs", direction, lineNumber],
    queryFn: () => getSectionWellLogs(direction, lineNumber),
  });

  const figure = useMemo(() => {
    if (direction === "inline" && inlineQuery.data) {
      return buildFigure(
        inlineQuery.data.crossline_axis,
        inlineQuery.data.twt_axis_ms,
        inlineQuery.data.amplitude,
        "Crossline",
        colors,
        wellLogsQuery.data?.wells,
      );
    }
    if (direction === "crossline" && crosslineQuery.data) {
      return buildFigure(
        crosslineQuery.data.inline_axis,
        crosslineQuery.data.twt_axis_ms,
        crosslineQuery.data.amplitude,
        "Inline",
        colors,
        wellLogsQuery.data?.wells,
      );
    }
    return null;
  }, [direction, inlineQuery.data, crosslineQuery.data, wellLogsQuery.data, colors]);

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
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
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
          <div className="flex flex-wrap items-center gap-4 px-2 pb-1 pt-2 text-xs font-semibold text-ink-muted">
            <span className="text-ink-faint font-normal">
              Well logs (VSH / PHIE / SWE), projected onto this section at each well's own tied position:
            </span>
            {(["vsh", "phie", "swe"] as const).map((key) => (
              <span key={key} className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: LOG_CURVE_COLORS[key] }}
                />
                {key.toUpperCase()}
              </span>
            ))}
          </div>
          {wellLogsQuery.data && wellLogsQuery.data.wells.length === 0 && (
            <p className="px-2 pb-2 text-xs text-ink-faint">
              No wells have a usable direct tie to draw yet.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// Fraction of the position axis's own width used for each curve lane's
// full 0->1 deflection, and the gap between the three lanes' baselines --
// keeps the wiggles a sensible size regardless of a survey's actual
// inline/crossline numbering (which is arbitrary, not physical units).
const LANE_DEFLECTION_FRACTION = 0.015;
const LANE_GAP_FRACTION = 0.02;

function buildFigure(
  positionAxis: number[],
  twtAxisMs: number[],
  amplitude: number[][], // already (n_samples, n_traces) from the API
  positionLabel: string,
  colors: ChartColors,
  wellLogs?: SectionWellLogCurve[],
): { data: Data[]; layout: Partial<Layout> } {
  const AXIS_STYLE = axisStyle(colors);
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

  const data: Data[] = [trace, ...buildWellLogWiggles(positionAxis, wellLogs, colors)];

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: positionLabel }, ...AXIS_STYLE },
    yaxis: { title: { text: "Two-Way Time (ms)" }, autorange: "reversed", ...AXIS_STYLE },
    showlegend: false,
  };

  return { data, layout };
}

/**
 * VSH/PHIE/SWE as three thin "log-on-section" wiggle lanes per well,
 * anchored at the well's own position on the section's cross-axis (see
 * section_well_log_service.py -- a projection, not a claim the well sits
 * exactly on this line). Each curve is drawn as a filled deflection from
 * its own lane baseline (0 -> the baseline, 1 -> baseline + full
 * deflection width), the same convention LogTrackViewer.tsx uses for VSH
 * on a plain depth track, just laid out as three side-by-side lanes here
 * since all three share one time axis with the seismic image instead of
 * each getting its own dedicated x-axis.
 */
function buildWellLogWiggles(
  positionAxis: number[],
  wellLogs: SectionWellLogCurve[] | undefined,
  colors: ChartColors,
): Data[] {
  if (!wellLogs || wellLogs.length === 0 || positionAxis.length === 0) return [];

  const axisMin = Math.min(...positionAxis);
  const axisMax = Math.max(...positionAxis);
  const axisSpan = Math.max(axisMax - axisMin, 1);
  const deflection = axisSpan * LANE_DEFLECTION_FRACTION;
  const gap = axisSpan * LANE_GAP_FRACTION;

  const lanes: { key: "vsh" | "phie" | "swe"; label: string; offset: number; color: string }[] = [
    { key: "vsh", label: "VSH", offset: -gap, color: LOG_CURVE_COLORS.vsh },
    { key: "phie", label: "PHIE", offset: 0, color: LOG_CURVE_COLORS.phie },
    { key: "swe", label: "SWE", offset: gap, color: LOG_CURVE_COLORS.swe },
  ];

  const traces: Data[] = [];

  for (const well of wellLogs) {
    // A thin vertical marker through the full section at the well's own
    // position, labeled once at the top -- so it's clear where the wells
    // this projection is even placing sit, before reading the wiggles.
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [well.position_on_axis, well.position_on_axis],
      y: [Math.min(...well.twt_ms), Math.max(...well.twt_ms)],
      line: { color: colors.inkFaint, width: 1, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false,
    } as Data);
    traces.push({
      type: "scatter",
      mode: "text",
      x: [well.position_on_axis],
      y: [Math.min(...well.twt_ms)],
      text: [`${well.well_id} (r=${well.correlation.toFixed(2)})`],
      textposition: "top center",
      textfont: { size: 9, color: colors.inkMuted },
      hoverinfo: "skip",
      showlegend: false,
    } as Data);

    for (const lane of lanes) {
      const baseline = well.position_on_axis + lane.offset;
      const values = well[lane.key];
      traces.push({
        type: "scatter",
        mode: "lines",
        name: lane.label,
        legendgroup: lane.key,
        // Missing samples draw at the lane's own baseline (zero
        // deflection) rather than as a gap -- simpler and avoids fill
        // artifacts from a broken line mid-lane.
        x: values.map((v) => (v === null ? baseline : baseline + v * deflection)),
        y: well.twt_ms,
        line: { color: lane.color, width: 1 },
        fill: "tonextx",
        fillcolor: lane.color + "33", // ~20% alpha fill under the curve
        hovertemplate: `${well.well_id} ${lane.label}: %{customdata:.3f}<extra></extra>`,
        customdata: values,
      } as Data);
      // Invisible baseline trace so "tonextx" above fills against the
      // lane's own baseline, not whatever trace happens to precede it.
      traces.push({
        type: "scatter",
        mode: "lines",
        x: values.map(() => baseline),
        y: well.twt_ms,
        line: { color: "transparent", width: 0 },
        hoverinfo: "skip",
        showlegend: false,
      } as Data);
    }
  }

  return traces;
}
