import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { getAllWellSeismicTies, getWellSeismicTie, listSeismic } from "@/api/client";
import type { WellSeismicTieRow } from "@/api/types";
import { useChartColors, usePlotlyLayout, type ChartColors } from "@/styles/tokens";

/** Correlation quality tiering, matching the pay/reservoir/danger semantic
 * colors used for zone quality elsewhere in the app (styles/tokens.ts) --
 * correlation is a tie-quality signal in the same "good / marginal / poor"
 * shape as a reservoir quality flag. */
function correlationColor(corr: number | null, colors: ChartColors): string {
  if (corr === null) return colors.inkFaint;
  if (corr >= 0.7) return colors.pay;
  if (corr >= 0.4) return colors.reservoir;
  return colors.danger;
}

function fmt(v: number | null, digits = 1): string {
  return v === null ? "—" : v.toFixed(digits);
}

type SortKey = "well_id" | "distance_m" | "best_freq_hz" | "bulk_shift_ms" | "correlation";

export default function WellSeismicTie() {
  const colors = useChartColors();
  const plotlyLayout = usePlotlyLayout();
  const datasetsQuery = useQuery({ queryKey: ["seismic-datasets"], queryFn: listSeismic });
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [selectedWellId, setSelectedWellId] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("correlation");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    if (!datasetId && datasetsQuery.data && datasetsQuery.data.length > 0) {
      setDatasetId(datasetsQuery.data[0].dataset_id);
    }
  }, [datasetsQuery.data, datasetId]);

  const batchQuery = useQuery({
    queryKey: ["well-seismic-tie-batch", datasetId],
    queryFn: () => getAllWellSeismicTies(datasetId!),
    enabled: Boolean(datasetId),
  });

  // Default to the best-correlation well, like the notebook's plot_tie('Z-04')
  // (its own highest-correlation well) -- once, per fresh dataset selection.
  useEffect(() => {
    if (!batchQuery.data) return;
    const tied = batchQuery.data.rows.filter((r) => r.error === null && r.correlation !== null);
    if (tied.length === 0) {
      setSelectedWellId(null);
      return;
    }
    setSelectedWellId((current) => {
      if (current && tied.some((r) => r.well_id === current)) return current;
      const best = tied.reduce((a, b) => (b.correlation! > a.correlation! ? b : a));
      return best.well_id;
    });
  }, [batchQuery.data]);

  const detailQuery = useQuery({
    queryKey: ["well-seismic-tie", selectedWellId, datasetId],
    queryFn: () => getWellSeismicTie(selectedWellId!, datasetId!),
    enabled: Boolean(selectedWellId && datasetId),
  });

  const sortedRows = useMemo(() => {
    const rows = batchQuery.data?.rows ?? [];
    const copy = [...rows];
    copy.sort((a, b) => {
      if (sortKey === "well_id") return a.well_id.localeCompare(b.well_id) * (sortAsc ? 1 : -1);
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return ((av as number) - (bv as number)) * (sortAsc ? 1 : -1);
    });
    return copy;
  }, [batchQuery.data, sortKey, sortAsc]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) setSortAsc((a) => !a);
    else {
      setSortKey(key);
      setSortAsc(key === "well_id");
    }
  }

  const mapWells = (batchQuery.data?.rows ?? []).filter(
    (r): r is WellSeismicTieRow & { well_x: number; well_y: number } =>
      r.well_x !== null && r.well_y !== null,
  );
  const footprint = batchQuery.data?.survey_footprint ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5"
          value={datasetId ?? ""}
          onChange={(e) => {
            setDatasetId(e.target.value || null);
            setSelectedWellId(null);
          }}
        >
          <option value="">Select seismic dataset…</option>
          {datasetsQuery.data?.map((d) => (
            <option key={d.dataset_id} value={d.dataset_id}>
              {d.dataset_id}
            </option>
          ))}
        </select>
      </div>

      {!datasetId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a seismic dataset to compute well ties.
        </div>
      )}

      {batchQuery.isLoading && (
        <div className="h-40 rounded-xl bg-surface-sunken animate-pulse" />
      )}

      {batchQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Batch tie failed: {(batchQuery.error as Error).message}
        </div>
      )}

      {batchQuery.data && (
        <>
          {batchQuery.data.warnings.length > 0 && (
            <details className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5">
              <summary className="cursor-pointer font-semibold">
                {batchQuery.data.warnings.length} well(s) had a tie issue
              </summary>
              <ul className="mt-2 space-y-1 list-disc list-inside">
                {batchQuery.data.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </details>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 items-start">
            <div className="xl:col-span-3 bg-surface border border-border rounded-xl shadow-card overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface-muted border-b border-border">
                    {(
                      [
                        ["well_id", "Well"],
                        ["distance_m", "Dist (m)"],
                        ["best_freq_hz", "Freq (Hz)"],
                        ["bulk_shift_ms", "Shift (ms)"],
                        ["correlation", "Correlation"],
                      ] as [SortKey, string][]
                    ).map(([key, label]) => (
                      <th
                        key={key}
                        onClick={() => toggleSort(key)}
                        className="text-left px-3 py-2.5 font-semibold text-ink-muted cursor-pointer select-none hover:text-accent transition-colors"
                      >
                        {label}
                        {sortKey === key && <span className="ml-1 text-accent">{sortAsc ? "↑" : "↓"}</span>}
                      </th>
                    ))}
                    <th className="text-left px-3 py-2.5 font-semibold text-ink-muted">Trace / IL-XL</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.map((row) => (
                    <tr
                      key={row.well_id}
                      onClick={() => setSelectedWellId(row.well_id)}
                      className={`group border-b border-border last:border-0 cursor-pointer transition-colors ${
                        selectedWellId === row.well_id ? "bg-accent-soft" : "hover:bg-accent-soft/60"
                      }`}
                    >
                      <td className="px-3 py-2.5 font-semibold text-accent-strong">
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-1.5 w-1.5 rounded-full"
                            style={{ backgroundColor: correlationColor(row.correlation, colors) }}
                          />
                          {row.well_id}
                        </span>
                      </td>
                      {row.error ? (
                        <td colSpan={4} className="px-3 py-2.5 text-danger text-xs">
                          {row.error}
                        </td>
                      ) : (
                        <>
                          <td className="px-3 py-2.5 text-ink-muted">{fmt(row.distance_m, 0)}</td>
                          <td className="px-3 py-2.5 text-ink-muted">{fmt(row.best_freq_hz, 1)}</td>
                          <td className="px-3 py-2.5 text-ink-muted">{fmt(row.bulk_shift_ms, 0)}</td>
                          <td className="px-3 py-2.5">
                            <span
                              className="inline-flex px-2 py-0.5 rounded-full text-xs font-semibold"
                              style={{
                                backgroundColor: `${correlationColor(row.correlation, colors)}1A`,
                                color: correlationColor(row.correlation, colors),
                              }}
                            >
                              {fmt(row.correlation, 3)}
                              {row.boundary_pinned && " ⚠"}
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-ink-faint text-xs">
                            #{row.trace_index}
                            {row.inline !== null && row.crossline !== null
                              ? ` · IL ${row.inline}/XL ${row.crossline}`
                              : ""}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                  {sortedRows.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center text-ink-faint">
                        No wells uploaded yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="xl:col-span-2 bg-surface border border-border rounded-xl shadow-card p-2">
              <p className="text-xs font-semibold text-ink-muted px-2 pt-1 pb-2">
                Well locations {footprint.length > 0 ? "vs. survey footprint" : ""}
              </p>
              <Plot
                data={[
                  ...(footprint.length > 0
                    ? [
                        {
                          x: footprint.map((p) => p.x),
                          y: footprint.map((p) => p.y),
                          type: "scattergl" as const,
                          mode: "markers" as const,
                          name: "Survey traces",
                          marker: { color: colors.gridLine, size: 3 },
                          hoverinfo: "skip" as const,
                        },
                      ]
                    : []),
                  {
                    x: mapWells.map((w) => w.well_x),
                    y: mapWells.map((w) => w.well_y),
                    type: "scatter" as const,
                    mode: "text+markers" as const,
                    name: "Wells",
                    text: mapWells.map((w) => w.well_id),
                    textposition: "top center" as const,
                    textfont: { size: 10, color: colors.inkMuted },
                    marker: {
                      color: mapWells.map((w) => correlationColor(w.correlation, colors)),
                      size: 10,
                      line: { color: colors.surface, width: 1 },
                    },
                    hovertext: mapWells.map(
                      (w) =>
                        `${w.well_id}<br>corr=${fmt(w.correlation, 3)}<br>` +
                        (w.inline !== null ? `IL ${w.inline} / XL ${w.crossline}` : `trace #${w.trace_index}`),
                    ),
                    hoverinfo: "text" as const,
                  },
                ]}
                layout={{
                  ...plotlyLayout,
                  autosize: true,
                  height: 420,
                  showlegend: false,
                  xaxis: { ...plotlyLayout.xaxis, title: { text: "X" } },
                  yaxis: { ...plotlyLayout.yaxis, title: { text: "Y" } },
                  margin: { t: 10, r: 10, b: 40, l: 65 },
                }}
                style={{ width: "100%" }}
                config={{ displaylogo: false }}
              />
            </div>
          </div>
        </>
      )}

      {selectedWellId && (
        <div className="space-y-3">
          {detailQuery.isLoading && <div className="h-96 rounded-xl bg-surface-sunken animate-pulse" />}

          {detailQuery.isError && (
            <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
              Tie failed: {(detailQuery.error as Error).message}
            </div>
          )}

          {detailQuery.data && (
            <div className="bg-surface border border-border rounded-xl shadow-card p-4">
              {detailQuery.data.geometry_warning && (
                <div className="border border-orange-200 bg-orange-soft/50 text-orange-strong text-xs rounded-xl px-4 py-2 mb-3">
                  {detailQuery.data.geometry_warning}
                </div>
              )}
              {detailQuery.data.boundary_pinned && (
                <div className="text-danger text-xs mb-2">
                  ⚠ Shift pinned to search edge — likely spurious, not a genuine tie
                </div>
              )}

              <p className="text-sm font-semibold text-ink text-center mb-2">
                {detailQuery.data.well_id} — corr={detailQuery.data.correlation.toFixed(3)}, {detailQuery.data.best_freq_hz.toFixed(0)}
                Hz, pol={detailQuery.data.polarity > 0 ? "+1" : "-1"}, shift=
                {detailQuery.data.bulk_shift_ms >= 0 ? "+" : ""}
                {detailQuery.data.bulk_shift_ms.toFixed(0)}ms
              </p>

              <div className="flex justify-center">
              <Plot
                data={[
                  {
                    x: detailQuery.data.synthetic_amplitude.map((v) => Math.max(v, 0)),
                    y: detailQuery.data.time_ms,
                    type: "scatter",
                    mode: "lines",
                    fill: "tozerox",
                    fillcolor: `${colors.orange}4D`,
                    line: { width: 0 },
                    showlegend: false,
                    hoverinfo: "skip",
                  },
                  {
                    x: detailQuery.data.seismic_amplitude,
                    y: detailQuery.data.time_ms,
                    type: "scatter",
                    mode: "lines",
                    name: "Extracted seismic",
                    line: { color: colors.ink, width: 1.5 },
                  },
                  {
                    x: detailQuery.data.synthetic_amplitude,
                    y: detailQuery.data.time_ms,
                    type: "scatter",
                    mode: "lines",
                    name: "Synthetic",
                    line: { color: colors.orange, width: 1.5 },
                  },
                ]}
                layout={{
                  ...plotlyLayout,
                  autosize: true,
                  height: 560,
                  width: 420,
                  xaxis: {
                    ...plotlyLayout.xaxis,
                    title: { text: "Normalized amplitude" },
                    zeroline: true,
                    zerolinecolor: colors.borderStrong,
                  },
                  yaxis: {
                    ...plotlyLayout.yaxis,
                    title: { text: "Two-Way Time (ms)" },
                    autorange: "reversed",
                  },
                  legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.06, yanchor: "bottom" },
                  margin: { t: 30, r: 20, b: 50, l: 55 },
                }}
                config={{ displaylogo: false }}
              />
              </div>
            </div>
          )}
        </div>
      )}

      <p className="text-xs text-ink-faint">
        Well tie: each well's own DPTM curve (vendor-precomputed when the LAS carries one, else a sonic-integration
        approximation) is jointly searched over Ricker wavelet frequency, polarity, and bulk time shift against the
        nearest real seismic trace, maximizing normalized cross-correlation across the full seismic window — no
        checkshot survey was available.
      </p>
    </div>
  );
}
