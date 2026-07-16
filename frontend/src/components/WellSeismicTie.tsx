import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { getWellSeismicTie, listSeismic, listWells } from "@/api/client";

/** RMS of a numeric array, floored to avoid a divide-by-zero blowup on a
 * degenerate all-zero trace (dead trace, or a synthetic with no overlap).
 * Same fix as SyntheticTraceOverlay.tsx / WellTieView.tsx: the synthetic
 * and the raw SEG-Y trace have no reason to share an amplitude scale, and
 * on a shared axis one routinely dwarfs the other into a flat line, even
 * when the underlying tie correlation (computed on zero-mean/unit-std
 * normalized copies, unaffected by this) is good. Normalized for DISPLAY
 * only -- the API keeps returning raw, unnormalized values. */
function rms(values: number[]): number {
  const meanSq = values.reduce((sum, v) => sum + v * v, 0) / (values.length || 1);
  const r = Math.sqrt(meanSq);
  return r > 1e-12 ? r : 1;
}

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

  const realRms = tieQuery.data ? rms(tieQuery.data.real_trace) : 1;
  const synRms = tieQuery.data ? rms(tieQuery.data.shifted_synthetic) : 1;

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
            <span>
              Best shift: {tieQuery.data.best_shift_ms.toFixed(1)} ms (search ±{tieQuery.data.max_shift_ms.toFixed(0)} ms)
            </span>
            <span className={tieQuery.data.correlation > 0.5 ? "text-green-600" : "text-orange-600"}>
              Correlation: {tieQuery.data.correlation.toFixed(3)}
            </span>
            {tieQuery.data.boundary_pinned && (
              <span className="text-danger">⚠ Shift pinned to search edge — likely spurious, not a genuine tie</span>
            )}
          </div>

          <Plot
            data={[
              {
                x: tieQuery.data.twt_ms,
                y: tieQuery.data.real_trace.map((v) => v / realRms),
                type: "scatter",
                mode: "lines",
                name: "Real seismic trace",
                line: { color: "#2563eb" },
              },
              {
                x: tieQuery.data.twt_ms,
                y: tieQuery.data.shifted_synthetic.map((v) => v / synRms),
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
  yaxis: { title: { text: "Amplitude (RMS-normalized)" } },
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