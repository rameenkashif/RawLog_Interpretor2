import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { SeismicSectionResponse } from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";

/**
 * Raw seismic section display: amplitude as a function of trace index (x)
 * and two-way time (y, reversed like a depth track). Rendered as a Plotly
 * heatmap using a diverging red-white-blue "seismic" colormap -- this is
 * standard practice for seismic amplitude display and applies only to the
 * data itself, not the surrounding page chrome, which stays on the
 * light background per the app's UI requirement.
 */
export default function SeismicSection({
  section,
}: {
  section: SeismicSectionResponse;
}) {
  const colors = useChartColors();
  const { data, layout } = useMemo(() => buildFigure(section, colors), [section, colors]);

  return (
    <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
      <Plot
        data={data}
        layout={layout}
        style={{ width: "100%", height: "520px" }}
        config={{ displaylogo: false, responsive: true }}
      />
    </div>
  );
}

function buildFigure(
  section: SeismicSectionResponse,
  colors: ChartColors,
): {
  data: Data[];
  layout: Partial<Layout>;
} {
  // amplitude is (n_traces, n_samples); Plotly heatmap wants z as
  // (rows=y, cols=x), so transpose to (n_samples, n_traces).
  const nTraces = section.trace_indices.length;
  const nSamples = section.twt_axis_ms.length;
  const z: number[][] = Array.from({ length: nSamples }, (_, sampleIdx) =>
    Array.from(
      { length: nTraces },
      (_, traceIdx) => section.amplitude[traceIdx][sampleIdx],
    ),
  );

  // NOTE: deliberately NOT using Math.min(...z.flat()) / Math.max(...z.flat())
  // here -- spreading a large flattened array (up to MAX_SECTION_TRACES x
  // MAX_SECTION_SAMPLES = hundreds of thousands of numbers) as individual
  // function arguments throws "RangeError: Maximum call stack size exceeded"
  // in most JS engines well before reaching that size. A manual reduce loop
  // has no such limit.
  let maxAbs = 1e-6;
  for (const row of z) {
    for (const value of row) {
      const abs = Math.abs(value);
      if (abs > maxAbs) maxAbs = abs;
    }
  }

  const trace: Data = {
    type: "heatmap",
    x: section.trace_indices,
    y: section.twt_axis_ms,
    z,
    zmid: 0,
    zmin: -maxAbs,
    zmax: maxAbs,
    colorscale: "RdBu",
    reversescale: true,
    colorbar: {
      title: "Amplitude",
      titlefont: { size: 10 },
      tickfont: { size: 9 },
    },
  };

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: {
      title: "Trace Index",
      gridcolor: colors.gridLine,
      linecolor: colors.borderStrong,
      tickfont: { color: colors.inkMuted },
    },
    yaxis: {
      title: "Two-Way Time (ms)",
      autorange: "reversed",
      gridcolor: colors.gridLine,
      linecolor: colors.borderStrong,
      tickfont: { color: colors.inkMuted },
    },
  };

  return { data: [trace], layout };
}
