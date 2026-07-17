import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { AxiosError } from "axios";
import { getWellZoneTieMap } from "@/api/client";
import { useChartColors, type ChartColors } from "@/styles/tokens";

function axisStyle(colors: ChartColors) {
  return {
    gridcolor: colors.gridLine,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  };
}

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * "Well-Seismic Tie" map: every well's Pay-zone mean VSH tied to the
 * survey via real coordinates (well_zone_tie_service.py), then spatially
 * interpolated (inverse-distance weighting) across the full inline x
 * crossline grid so sparse well control reads as a continuous "Predicted
 * VSH" map, wells overlaid as labeled markers -- same map-view convention
 * as TimeSliceView, plus a scatter trace for the well positions.
 *
 * This is geometric interpolation between known well values, NOT a
 * seismic inversion or ML prediction (it never looks at seismic
 * amplitude) -- the backend's method_note carries that caveat and is
 * always shown alongside the map.
 */
export default function WellZoneTieMapView() {
  const colors = useChartColors();
  const [idwPower, setIdwPower] = useState(2);

  const query = useQuery({
    queryKey: ["seismic-viz-well-zone-tie-map", idwPower],
    queryFn: () => getWellZoneTieMap(idwPower),
    retry: false,
  });

  const figure = useMemo(() => {
    if (!query.data) return null;
    return buildFigure(query.data, colors);
  }, [query.data, colors]);

  return (
    <div className="space-y-3">
      <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
        IDW power
        <input
          type="range"
          min={1}
          max={4}
          step={0.5}
          value={idwPower}
          onChange={(e) => setIdwPower(Number(e.target.value))}
          className="w-40 accent-accent"
        />
        <span className="text-ink-faint font-normal">
          {idwPower.toFixed(1)} (higher = more locally dominated by the nearest well)
        </span>
      </label>

      {query.isLoading && <div className="h-[520px] rounded-xl bg-surface-sunken animate-pulse" />}

      {query.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Well-seismic tie map unavailable: {errorMessage(query.error)}
        </div>
      )}

      {query.data && (
        <div className="space-y-3">
          <div className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
            {query.data.method_note}
          </div>

          {query.data.warnings.length > 0 && (
            <div className="border border-border-strong bg-surface-sunken text-ink-muted text-xs rounded-xl px-4 py-2.5 leading-relaxed space-y-1">
              <div className="font-semibold text-ink">Wells not shown on this map:</div>
              {query.data.warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          {figure && (
            <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
              <Plot
                data={figure.data}
                layout={figure.layout}
                style={{ width: "100%", height: "560px" }}
                config={{ displaylogo: false, responsive: true }}
              />
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="text-ink-faint border-b border-border">
                  <th className="py-1.5 pr-4 font-semibold">Well</th>
                  <th className="py-1.5 pr-4 font-semibold">Inline</th>
                  <th className="py-1.5 pr-4 font-semibold">Crossline</th>
                  <th className="py-1.5 pr-4 font-semibold">Distance to trace (m)</th>
                  <th className="py-1.5 pr-4 font-semibold">Mean VSH (Pay zone)</th>
                  <th className="py-1.5 pr-4 font-semibold">Pay samples</th>
                </tr>
              </thead>
              <tbody>
                {query.data.wells.map((w) => (
                  <tr key={w.well_id} className="border-b border-border last:border-0">
                    <td className="py-1.5 pr-4 font-semibold text-ink">{w.well_name}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{w.inline}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{w.crossline}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{w.distance_m.toFixed(0)}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{w.mean_vsh_pay.toFixed(3)}</td>
                    <td className="py-1.5 pr-4 text-ink-muted">{w.n_pay_samples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function buildFigure(
  data: {
    inline_axis: number[];
    crossline_axis: number[];
    predicted_vsh: (number | null)[][];
    wells: { well_id: string; well_name: string; inline: number; crossline: number }[];
  },
  colors: ChartColors,
): { data: Data[]; layout: Partial<Layout> } {
  const AXIS_STYLE = axisStyle(colors);
  const heatmap = {
    type: "heatmap",
    x: data.crossline_axis,
    y: data.inline_axis,
    z: data.predicted_vsh,
    colorscale: "Viridis",
    colorbar: { title: { text: "Predicted VSH", font: { size: 10 } }, tickfont: { size: 9 } },
  } as Data;

  const wells = {
    type: "scatter",
    mode: "markers+text",
    x: data.wells.map((w) => w.crossline),
    y: data.wells.map((w) => w.inline),
    text: data.wells.map((w) => w.well_name),
    textposition: "top center",
    textfont: { color: colors.ink, size: 11 },
    marker: { symbol: "triangle-up", size: 12, color: colors.danger, line: { color: colors.surface, width: 1 } },
    hovertemplate: "%{text}<br>Inline %{y}, Crossline %{x}<extra></extra>",
  } as unknown as Data;

  const layout: Partial<Layout> = {
    title: { text: "Predicted VSH", font: { size: 13, color: colors.ink } },
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 40, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: "Crossline" }, ...AXIS_STYLE },
    yaxis: { title: { text: "Inline" }, ...AXIS_STYLE },
    showlegend: false,
  };

  return { data: [heatmap, wells], layout };
}
