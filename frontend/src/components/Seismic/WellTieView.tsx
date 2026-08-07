import { useEffect, useState } from "react";
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
import { useChartColors } from "@/styles/tokens";
import { useAppStore } from "@/store/useAppStore";
import { isLowConfidenceTie } from "@/utils/tieConfidence";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/** RMS of a numeric array, floored to avoid a divide-by-zero blowup on a
 * degenerate all-zero trace (dead trace, or a synthetic with no overlap). */
function rms(values: number[]): number {
  const meanSq = values.reduce((sum, v) => sum + v * v, 0) / (values.length || 1);
  const r = Math.sqrt(meanSq);
  return r > 1e-12 ? r : 1;
}

/**
 * Synthetic-vs-real trace overlay for a well tie computed directly against
 * the SEG-Y volume (app/services/seismic_processor.py get_well_tie) --
 * the SAME tie algorithm as the main Seismic page's Well-to-Seismic Tie
 * (WellSeismicTie.tsx / /tie/{well_id}: the well's own DPTM curve jointly
 * searched over Ricker frequency, polarity, and bulk shift against the
 * nearest real trace, via direct_tie_service.resolve_direct_tie), just
 * applied to this feature's single active volume instead of an uploaded
 * dataset. A 1D line overlay fits Recharts fine, unlike the 2D sections.
 *
 * Each curve is independently RMS-normalized for DISPLAY only (see
 * SyntheticTraceOverlay.tsx's identical fix) -- the synthetic and the raw
 * SEG-Y trace have no reason to share an amplitude scale, and on a shared
 * axis one routinely dwarfs the other into a flat line.
 */
export default function WellTieView() {
  const colors = useChartColors();
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const [wellId, setWellId] = useState<string | null>(null);
  // Manual wavelet-frequency override -- null means "auto-optimize over the
  // full frequency grid" (the default), matching WellSeismicTie.tsx.
  const [manualFreqHz, setManualFreqHz] = useState<number | null>(null);
  const [freqDraft, setFreqDraft] = useState<string>("");
  const activeWellId = useAppStore((s) => s.activeWellId);

  // Seed/redirect from the dashboard's shared active well -- a manual pick
  // from the dropdown below still overrides this until it changes again.
  useEffect(() => {
    if (activeWellId) setWellId(activeWellId);
  }, [activeWellId]);

  // A newly selected well starts back at auto-optimize.
  useEffect(() => {
    setManualFreqHz(null);
  }, [wellId]);

  const tieQuery = useQuery({
    queryKey: ["seismic-viz-well-tie", wellId, manualFreqHz],
    queryFn: () => getWellTieViz(wellId!, manualFreqHz ?? undefined),
    enabled: Boolean(wellId),
    retry: false,
  });

  useEffect(() => {
    if (tieQuery.data) setFreqDraft(tieQuery.data.wavelet_freq_hz.toFixed(1));
  }, [tieQuery.data]);

  function applyFreqDraft() {
    const v = parseFloat(freqDraft);
    if (!Number.isNaN(v) && v > 0) setManualFreqHz(v);
  }

  const realRms = tieQuery.data ? rms(tieQuery.data.real_trace) : 1;
  const synRms = tieQuery.data ? rms(tieQuery.data.synthetic) : 1;
  const chartData = tieQuery.data?.twt_ms.map((t, i) => ({
    twt_ms: t,
    synthetic: tieQuery.data!.synthetic[i] / synRms,
    real: tieQuery.data!.real_trace[i] / realRms,
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

        <span className="text-xs font-semibold text-ink-muted">Wavelet frequency</span>
        <input
          type="number"
          step={0.5}
          min={1}
          value={freqDraft}
          onChange={(e) => setFreqDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") applyFreqDraft();
          }}
          className="w-24 text-sm border border-border-strong rounded-lg px-2 py-1"
        />
        <span className="text-xs text-ink-faint">Hz</span>
        <button
          type="button"
          onClick={applyFreqDraft}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-accent text-white hover:opacity-90 transition-opacity"
        >
          Apply
        </button>
        <button
          type="button"
          onClick={() => setManualFreqHz(null)}
          disabled={manualFreqHz === null}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-border-strong text-ink-muted disabled:opacity-40 disabled:cursor-not-allowed hover:bg-surface-sunken transition-colors"
        >
          Auto-optimize
        </button>
        {manualFreqHz !== null && (
          <span className="text-xs text-orange-strong font-medium">Manual override</span>
        )}
      </div>

      {!wellId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a well to compute its tie against the seismic volume.
        </div>
      )}

      {tieQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {tieQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Tie failed: {errorMessage(tieQuery.error)}
        </div>
      )}

      {tieQuery.data && (
        <div className="space-y-3">
          <div className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
            {tieQuery.data.note}
          </div>

          {isLowConfidenceTie(tieQuery.data.correlation, tieQuery.data.boundary_pinned) && (
            <div className="text-danger text-xs">
              ⚠{" "}
              {tieQuery.data.boundary_pinned
                ? "Shift pinned to search edge — likely spurious, not a genuine tie"
                : `Low-confidence tie — correlation ${tieQuery.data.correlation.toFixed(3)} is below the 0.3 threshold`}
            </div>
          )}

          <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
            <span>
              Nearest inline/crossline: {tieQuery.data.nearest_inline} / {tieQuery.data.nearest_crossline}
            </span>
            <span>Distance: {tieQuery.data.distance_m.toFixed(0)} m</span>
            <span>corr={tieQuery.data.correlation.toFixed(3)}</span>
            <span>{tieQuery.data.wavelet_freq_hz.toFixed(0)}Hz</span>
            <span>pol={tieQuery.data.polarity > 0 ? "+1" : "-1"}</span>
            <span>
              shift={tieQuery.data.bulk_shift_ms >= 0 ? "+" : ""}
              {tieQuery.data.bulk_shift_ms.toFixed(0)}ms
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
                  label={{ value: "Amplitude (RMS-normalized)", angle: -90, position: "insideLeft", fill: colors.inkMuted }}
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
