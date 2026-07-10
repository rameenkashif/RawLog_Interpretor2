import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getWellTieViz, listWells } from "@/api/client";
import { colors } from "@/styles/tokens";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Synthetic-vs-real trace overlay for a well tie computed directly against
 * the SEG-Y volume (app/services/seismic_processor.py get_well_tie) --
 * distinct from the upload-pipeline tie on the main Seismic page
 * (WellSeismicTie.tsx / /tie/{well_id}), which ties against a manually
 * uploaded+processed dataset instead of this feature's single active
 * volume. A 1D line overlay fits Recharts fine, unlike the 2D sections.
 */
export default function WellTieView() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const [wellId, setWellId] = useState<string | null>(null);
  const [waveletFreqHz, setWaveletFreqHz] = useState(25);

  const tieQuery = useQuery({
    queryKey: ["seismic-viz-well-tie", wellId, waveletFreqHz],
    queryFn: () => getWellTieViz(wellId!, waveletFreqHz),
    enabled: Boolean(wellId),
    retry: false,
  });

  const chartData = tieQuery.data?.twt_ms.map((t, i) => ({
    twt_ms: t,
    synthetic: tieQuery.data!.synthetic[i],
    real: tieQuery.data!.real_trace[i],
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5"
          value={wellId ?? ""}
          onChange={(e) => setWellId(e.target.value || null)}
        >
          <option value="">Select well…</option>
          {wellsQuery.data?.map((w) => (
            <option key={w.well_id} value={w.well_id}>
              {w.well_id}
            </option>
          ))}
        </select>

        <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
          Wavelet frequency (Hz)
          <input
            type="number"
            min={1}
            max={200}
            value={waveletFreqHz}
            onChange={(e) => setWaveletFreqHz(Number(e.target.value) || 25)}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
        </label>
      </div>

      {!wellId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a well to compute its tie against the seismic volume.
        </div>
      )}

      {tieQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {tieQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Tie failed: {errorMessage(tieQuery.error)}
        </div>
      )}

      {tieQuery.data && (
        <div className="space-y-3">
          <div className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
            {tieQuery.data.note}
          </div>

          <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
            <span>
              Nearest inline/crossline: {tieQuery.data.nearest_inline} / {tieQuery.data.nearest_crossline}
            </span>
            <span>
              {tieQuery.data.tie_method === "manual_override"
                ? "Distance: manual override"
                : `Distance: ${tieQuery.data.distance_m?.toFixed(0)} m`}
            </span>
          </div>

          <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
            <ResponsiveContainer width="100%" height={420}>
              <LineChart data={chartData}>
                <CartesianGrid stroke={colors.gridLine} />
                <XAxis
                  dataKey="twt_ms"
                  stroke={colors.borderStrong}
                  tick={{ fill: colors.inkMuted, fontSize: 11 }}
                  label={{ value: "Two-Way Time (ms)", position: "insideBottom", offset: -5, fill: colors.inkMuted }}
                />
                <YAxis
                  stroke={colors.borderStrong}
                  tick={{ fill: colors.inkMuted, fontSize: 11 }}
                  label={{ value: "Amplitude", angle: -90, position: "insideLeft", fill: colors.inkMuted }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }}
                />
                <Legend />
                <Line type="monotone" dataKey="real" name="Real trace" stroke={colors.accent} dot={false} strokeWidth={1.5} />
                <Line
                  type="monotone"
                  dataKey="synthetic"
                  name="Synthetic"
                  stroke={colors.orange}
                  dot={false}
                  strokeWidth={1.5}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
