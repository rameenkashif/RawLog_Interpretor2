import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { WellCurvesResponse } from "@/api/types";
import { colors } from "@/styles/tokens";

interface LogTrackViewerProps {
  curves: WellCurvesResponse;
  /** Optional depth window [min, max] to zoom into; defaults to full log. */
  depthRange?: [number, number];
}

const AXIS_STYLE = {
  gridcolor: colors.gridLine,
  zerolinecolor: colors.border,
  linecolor: colors.borderStrong,
  tickfont: { color: colors.inkMuted, size: 10 },
  titlefont: { color: colors.inkMuted, size: 11 },
  showline: true,
  mirror: true,
};

/**
 * Classic multi-track wireline log display, built once and reused for every
 * well. Implemented as a single Plotly figure with one shared, depth-
 * reversed y-axis and multiple x-axis "tracks" laid out side-by-side via
 * axis domains -- the standard technique for well-log plots in Plotly.
 *
 * Tracks (left to right):
 *   1. GR (line) + VSH (filled area, overlaid secondary x-axis)
 *   2. Resistivity (log scale)
 *   3. RHOB / NPHI overlay (density-neutron, classic crossover display)
 *   4. DT (sonic)
 *   5. PHIE / PHIT / SWE (computed porosity + saturation)
 *   6. ZONES (categorical color column: green = Pay, amber = Reservoir
 *      non-pay, gray = Non-reservoir)
 */
export default function LogTrackViewer({ curves, depthRange }: LogTrackViewerProps) {
  const { data, layout } = useMemo(() => buildFigure(curves, depthRange), [curves, depthRange]);

  return (
    <div className="bg-surface border border-border rounded-lg p-2 shadow-sm">
      <Plot
        data={data}
        layout={layout}
        style={{ width: "100%", height: "720px" }}
        config={{ displaylogo: false, responsive: true }}
      />
    </div>
  );
}

function buildFigure(
  curves: WellCurvesResponse,
  depthRange?: [number, number]
): { data: Data[]; layout: Partial<Layout> } {
  const rows = curves.data;
  const depth = rows.map((r) => r.DEPT as number);

  const col = (name: string): (number | null)[] =>
    rows.map((r) => (typeof r[name] === "number" ? (r[name] as number) : null));

  const gr = col("GR");
  const vsh = col("VSH");
  const rt = col("RESISTIVITY");
  const rhob = col("RHOB");
  const nphi = col("NPHI");
  const dt = col("DT");
  const phie = col("PHIE");
  const phit = col("PHIT");
  const swe = col("SWE");
  const zones = col("ZONES");

  // Discrete-looking colorscale trick for the categorical ZONES heatmap column:
  // zone codes 1 (Pay), 2 (Reservoir non-pay), 3 (Non-reservoir) map to
  // normalized positions 0, 0.5, 1 -- with hard steps rather than gradients.
  const zonesColorscale: [number, string][] = [
    [0, colors.pay],
    [0.4999, colors.pay],
    [0.5, colors.reservoir],
    [0.9999, colors.reservoir],
    [1, colors.nonReservoir],
  ];

  const traces: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      name: "GR",
      x: gr,
      y: depth,
      xaxis: "x",
      yaxis: "y",
      line: { color: "#166534", width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "VSH",
      x: vsh,
      y: depth,
      xaxis: "x7",
      yaxis: "y",
      fill: "tozerox",
      fillcolor: "rgba(148, 163, 184, 0.35)",
      line: { color: colors.inkFaint, width: 1 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Resistivity",
      x: rt,
      y: depth,
      xaxis: "x2",
      yaxis: "y",
      line: { color: colors.danger, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "RHOB",
      x: rhob,
      y: depth,
      xaxis: "x3",
      yaxis: "y",
      line: { color: colors.danger, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "NPHI",
      x: nphi,
      y: depth,
      xaxis: "x8",
      yaxis: "y",
      line: { color: colors.accent, width: 1.5, dash: "dot" },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "DT",
      x: dt,
      y: depth,
      xaxis: "x4",
      yaxis: "y",
      line: { color: "#7C3AED", width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "PHIE",
      x: phie,
      y: depth,
      xaxis: "x5",
      yaxis: "y",
      line: { color: colors.accent, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "PHIT",
      x: phit,
      y: depth,
      xaxis: "x5",
      yaxis: "y",
      line: { color: colors.inkFaint, width: 1, dash: "dash" },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "SWE",
      x: swe,
      y: depth,
      xaxis: "x5",
      yaxis: "y",
      line: { color: "#0EA5E9", width: 1.5 },
    },
    {
      type: "heatmap",
      name: "ZONES",
      x: [0, 1],
      y: depth,
      z: zones.map((z) => [z ?? 3, z ?? 3]),
      xaxis: "x6",
      yaxis: "y",
      zmin: 1,
      zmax: 3,
      colorscale: zonesColorscale,
      showscale: false,
      hoverinfo: "y+z",
    },
  ];

  const yRange = depthRange ? [depthRange[1], depthRange[0]] : undefined;

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 60, r: 20, b: 30, l: 60 },
    showlegend: false,
    grid: { rows: 1, columns: 6, pattern: "independent" },
    yaxis: {
      title: "Depth (m)",
      autorange: yRange ? undefined : "reversed",
      range: yRange,
      domain: [0, 1],
      ...AXIS_STYLE,
    },
    xaxis: { title: "GR (API)", domain: [0, 0.14], anchor: "y", side: "top", ...AXIS_STYLE },
    xaxis7: {
      overlaying: "x",
      domain: [0, 0.14],
      anchor: "y",
      side: "top",
      range: [0, 1],
      showgrid: false,
      title: "VSH",
      titlefont: { color: colors.inkFaint, size: 10 },
      tickfont: { color: colors.inkFaint, size: 9 },
    },
    xaxis2: {
      title: "Resistivity (ohm.m)",
      domain: [0.17, 0.31],
      anchor: "y",
      side: "top",
      type: "log",
      ...AXIS_STYLE,
    },
    xaxis3: {
      title: "RHOB (g/cc)",
      domain: [0.34, 0.48],
      anchor: "y",
      side: "top",
      range: [1.95, 2.95],
      ...AXIS_STYLE,
    },
    xaxis8: {
      overlaying: "x3",
      domain: [0.34, 0.48],
      anchor: "y",
      side: "top",
      range: [0.45, -0.15],
      showgrid: false,
      title: "NPHI",
      titlefont: { color: colors.accent, size: 10 },
      tickfont: { color: colors.accent, size: 9 },
    },
    xaxis4: {
      title: "DT (us/ft)",
      domain: [0.51, 0.65],
      anchor: "y",
      side: "top",
      ...AXIS_STYLE,
    },
    xaxis5: {
      title: "PHIE / PHIT / SWE",
      domain: [0.68, 0.86],
      anchor: "y",
      side: "top",
      range: [0, 1],
      ...AXIS_STYLE,
    },
    xaxis6: {
      title: "ZONES",
      domain: [0.89, 0.97],
      anchor: "y",
      side: "top",
      showticklabels: false,
      ...AXIS_STYLE,
    },
  };

  return { data: traces, layout };
}
