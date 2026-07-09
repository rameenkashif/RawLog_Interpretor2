import { useMemo } from "react";
import type { SyntheticSeismogramResponse } from "@/api/types";

/**
 * Washout/hole-quality QC proxy summary: groups contiguous flagged depth
 * samples into intervals and lists them -- a soft heuristic (NPHI-RHOB
 * crossover / DT spikes), NOT a real caliper substitute, labeled as such.
 */
export default function WashoutSummary({ result }: { result: SyntheticSeismogramResponse }) {
  const intervals = useMemo(() => groupIntervals(result.washout_depth_m, result.washout_flag), [result]);

  return (
    <div className="bg-surface border border-orange/30 bg-orange-soft/30 rounded-xl p-4 shadow-card space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-ink">Washout / Hole-Quality QC Proxy</h4>
        <span className="text-xs font-semibold text-orange-strong bg-orange-soft px-2 py-0.5 rounded-full">
          Heuristic — not real caliper
        </span>
      </div>
      <p className="text-xs text-ink-muted leading-relaxed">
        Flags depth intervals with anomalous NPHI-RHOB crossover or erratic DT spikes as "possible washout /
        unreliable interval." No CALI curve is available for these wells, so this is a soft QC proxy only.
      </p>
      {intervals.length === 0 ? (
        <p className="text-xs text-green-700 font-semibold">No flagged intervals.</p>
      ) : (
        <ul className="text-xs text-ink-muted space-y-1 max-h-40 overflow-y-auto">
          {intervals.map(([start, end], i) => (
            <li key={i} className="font-mono">
              {start.toFixed(1)} – {end.toFixed(1)} m
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function groupIntervals(depths: number[], flags: boolean[]): [number, number][] {
  const intervals: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i < flags.length; i++) {
    if (flags[i] && start === null) {
      start = depths[i];
    } else if (!flags[i] && start !== null) {
      intervals.push([start, depths[i - 1]]);
      start = null;
    }
  }
  if (start !== null) intervals.push([start, depths[depths.length - 1]]);
  return intervals;
}
