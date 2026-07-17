import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { SeismicAttributesResponse } from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";

function axisStyle(colors: ChartColors) {
  return {
    gridcolor: colors.gridLine,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  };
}

/**
 * Per-trace seismic attribute trends: raw attributes (RMS amplitude,
 * envelope) in one panel, and the heuristic VSH/PHIE/SWE seismic proxies
 * (0-1 scale) in a second panel -- with an explicit on-screen caveat, since
 * those proxies are uncalibrated amplitude heuristics, not measured rock
 * properties (see backend/app/seismic_attributes.py for the full caveat).
 */
export default function SeismicAttributesChart({
  attributes,
}: {
  attributes: SeismicAttributesResponse;
}) {
  const colors = useChartColors();
  const { rawData, rawLayout, proxyData, proxyLayout } = useMemo(
    () => buildFigures(attributes, colors),
    [attributes, colors]
  );

  return (
    <div className="space-y-4">
      <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
        <h3 className="text-sm font-semibold text-ink mb-2">Raw Seismic Attributes</h3>
        <Plot
          data={rawData}
          layout={rawLayout}
          style={{ width: "100%", height: "280px" }}
          config={{ displaylogo: false, responsive: true }}
        />
      </div>

      <div className="bg-surface border border-orange/30 rounded-xl p-4 shadow-card bg-orange-soft/30">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-ink">Seismic VSH / PHIE / SWE Proxies</h3>
          <span className="text-xs font-semibold text-orange-strong bg-orange-soft px-2 py-0.5 rounded-full">
            Uncalibrated heuristic
          </span>
        </div>
        <p className="text-xs text-ink-muted mb-3 leading-relaxed">
          These are amplitude-based proxies, NOT measured shale volume, porosity, or water
          saturation. They require a real well tie before being used for interpretation -- see
          the well-derived VSH/PHIE/SWE curves on the single-well page for actual log-based
          values.
        </p>
        <Plot
          data={proxyData}
          layout={proxyLayout}
          style={{ width: "100%", height: "280px" }}
          config={{ displaylogo: false, responsive: true }}
        />
      </div>
    </div>
  );
}

function buildFigures(attributes: SeismicAttributesResponse, colors: ChartColors) {
  const AXIS_STYLE = axisStyle(colors);
  const rawData: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      name: "RMS Amplitude",
      x: attributes.trace_index,
      y: attributes.rms_amplitude,
      line: { color: colors.accent, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Avg Envelope",
      x: attributes.trace_index,
      y: attributes.avg_envelope,
      line: { color: colors.orange, width: 1.5 },
      yaxis: "y2",
    },
  ];

  const rawLayout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 10, r: 50, b: 40, l: 50 },
    xaxis: { title: "Trace Index", ...AXIS_STYLE },
    yaxis: { title: "RMS Amplitude", ...AXIS_STYLE },
    yaxis2: {
      title: "Avg Envelope",
      overlaying: "y",
      side: "right",
      showgrid: false,
      tickfont: { color: colors.inkMuted },
      titlefont: { color: colors.orange },
    },
    legend: { orientation: "h", y: -0.25, font: { size: 11, color: colors.inkMuted } },
  };

  const proxyData: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      name: "VSH proxy",
      x: attributes.trace_index,
      y: attributes.vsh_seismic_proxy,
      line: { color: colors.accentDeep, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "PHIE proxy",
      x: attributes.trace_index,
      y: attributes.phie_seismic_proxy,
      line: { color: colors.accent, width: 1.5 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "SWE proxy",
      x: attributes.trace_index,
      y: attributes.swe_seismic_proxy,
      line: { color: colors.orange, width: 1.5 },
    },
  ];

  const proxyLayout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 10, r: 20, b: 40, l: 50 },
    xaxis: { title: "Trace Index", ...AXIS_STYLE },
    yaxis: { title: "Proxy Value (0-1)", range: [0, 1], ...AXIS_STYLE },
    legend: { orientation: "h", y: -0.25, font: { size: 11, color: colors.inkMuted } },
  };

  return { rawData, rawLayout, proxyData, proxyLayout };
}
