import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { SyntheticSeismogramResponse } from "@/api/types";
import { colors } from "@/styles/tokens";

const AXIS_STYLE = {
  gridcolor: colors.gridLine,
  linecolor: colors.borderStrong,
  tickfont: { color: colors.inkMuted },
};

/**
 * Acoustic impedance (depth domain) and reflectivity (depth + time domain)
 * -- two side-by-side depth tracks like a classic log display, using
 * Plotly for the shared reversed-depth-axis convention already used
 * elsewhere in the app (LogTrackViewer.tsx).
 */
export default function AcousticImpedanceChart({ result }: { result: SyntheticSeismogramResponse }) {
  const { data, layout } = useMemo(() => buildFigure(result), [result]);

  return (
    <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
      <Plot data={data} layout={layout} style={{ width: "100%", height: "460px" }} config={{ displaylogo: false, responsive: true }} />
    </div>
  );
}

function buildFigure(result: SyntheticSeismogramResponse): { data: Data[]; layout: Partial<Layout> } {
  const aiTrace = {
    type: "scatter",
    mode: "lines",
    name: "Acoustic Impedance",
    x: result.acoustic_impedance,
    y: result.depth_m,
    line: { color: colors.accent, width: 1.2 },
    xaxis: "x",
    yaxis: "y",
  } as Data;

  const rcTrace = {
    type: "scatter",
    mode: "lines",
    name: "Reflectivity",
    x: result.reflectivity,
    y: result.reflectivity_depth_m,
    line: { color: colors.orange, width: 1 },
    xaxis: "x2",
    yaxis: "y",
  } as Data;

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 30, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: "Acoustic Impedance" }, domain: [0, 0.46], ...AXIS_STYLE },
    xaxis2: { title: { text: "Reflectivity" }, domain: [0.54, 1], anchor: "y", ...AXIS_STYLE },
    yaxis: { title: { text: "Depth (m)" }, autorange: "reversed", ...AXIS_STYLE },
    showlegend: false,
  };

  return { data: [aiTrace, rcTrace], layout };
}
