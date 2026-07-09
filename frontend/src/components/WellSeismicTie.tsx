import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { getWellSeismicTie, listSeismic, listWells } from "@/api/client";

export default function WellSeismicTie() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const datasetsQuery = useQuery({ queryKey: ["seismic-datasets"], queryFn: listSeismic });

  const [wellId, setWellId] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);

  const tieQuery = useQuery({
    queryKey: ["well-seismic-tie", wellId, datasetId],
    queryFn: () => getWellSeismicTie(wellId!, datasetId!),
    enabled: Boolean(wellId && datasetId),
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5"
          value={wellId ?? ""}
          onChange={(e) => setWellId(e.target.value || null)}
        >
          <option value="">Select well…</option>
          {wellsQuery.data?.map((w) => (
            <option key={w.well_id} value={w.well_id}>{w.well_id}</option>
          ))}
        </select>

        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5"
          value={datasetId ?? ""}
          onChange={(e) => setDatasetId(e.target.value || null)}
        >
          <option value="">Select seismic dataset…</option>
          {datasetsQuery.data?.map((d) => (
            <option key={d.dataset_id} value={d.dataset_id}>{d.dataset_id}</option>
          ))}
        </select>
      </div>

      {tieQuery.isLoading && (
        <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />
      )}

      {tieQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Tie failed: {(tieQuery.error as Error).message}
        </div>
      )}

      {tieQuery.data && (
        <div className="space-y-3">
          {tieQuery.data.geometry_warning && (
            <div className="border border-orange-200 bg-orange-50 text-orange-800 text-xs rounded-xl px-4 py-2">
              {tieQuery.data.geometry_warning}
            </div>
          )}

          <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
            <span
              className={
                tieQuery.data.tie_method === "nearest_trace" ? "text-green-600" : "text-orange-600"
              }
            >
              {tieQuery.data.tie_method === "nearest_trace"
                ? "Nearest-trace (coordinate-based)"
                : "Manual trace override"}
            </span>
            <span>Trace index: {tieQuery.data.trace_index}</span>
            {tieQuery.data.distance_m !== null && (
              <span>Distance: {tieQuery.data.distance_m.toFixed(0)} m</span>
            )}
            <span>Best shift: {tieQuery.data.best_shift_ms.toFixed(1)} ms</span>
            <span className={tieQuery.data.correlation > 0.5 ? "text-green-600" : "text-orange-600"}>
              Correlation: {tieQuery.data.correlation.toFixed(3)}
            </span>
          </div>

          <Plot
            data={[
              {
                x: tieQuery.data.twt_ms,
                y: tieQuery.data.real_trace,
                type: "scatter",
                mode: "lines",
                name: "Real seismic trace",
                line: { color: "#2563eb" },
              },
              {
                x: tieQuery.data.twt_ms,
                y: tieQuery.data.shifted_synthetic,
                type: "scatter",
                mode: "lines",
                name: "Synthetic (shifted)",
                line: { color: "#ea580c" },
              },
            ]}
            layout={{
  autosize: true,
  height: 420,
  xaxis: { title: { text: "Two-Way Time (ms)" } },
  yaxis: { title: { text: "Amplitude" } },
  legend: { orientation: "h" },
  margin: { t: 20 },
}}
            style={{ width: "100%" }}
            config={{ displaylogo: false }}
          />

          <p className="text-xs text-ink-faint">
            Trend correlation from a sonic-integrated synthetic seismogram
            (Ricker wavelet) — no checkshot survey was available, so the
            depth-time relationship comes from the sonic log itself.
          </p>
        </div>
      )}
    </div>
  );
}