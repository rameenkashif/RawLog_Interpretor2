import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getSpectralPetroCorrelation, listWells } from "@/api/client";
import type { SpectralPetroCorrelationWellResult } from "@/api/types";
import { colors } from "@/styles/tokens";

const SWT_MIN_LEVEL = 1;
const SWT_MAX_LEVEL = 6;
const SWT_DEFAULT_LEVEL = 3;

const PROPERTY_LABELS: { key: "vsh" | "phie" | "swe"; label: string }[] = [
  { key: "vsh", label: "VSH" },
  { key: "phie", label: "PHIE" },
  { key: "swe", label: "SWE" },
];

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function fmtR(r: number | null): string {
  return r === null ? "n/a" : r.toFixed(2);
}

/** Auto-generated comparison sentence: for each property, whichever
 * method has the larger |r| is stated as "more diagnostic" -- sign is
 * preserved in the displayed values, but "more strongly" compares
 * magnitude, not raw value. */
function buildSummary(
  subject: string,
  pairs: { label: string; cwtR: number | null; swtR: number | null }[],
): string {
  const clauses = pairs.map(({ label, cwtR, swtR }) => {
    if (cwtR === null && swtR === null) return `insufficient data for ${label}`;
    if (cwtR === null) return `only SWT correlates with ${label} (r=${fmtR(swtR)})`;
    if (swtR === null) return `only CWT correlates with ${label} (r=${fmtR(cwtR)})`;
    const swtStronger = Math.abs(swtR) >= Math.abs(cwtR);
    const [strongName, strongR, weakName, weakR] = swtStronger
      ? ["SWT", swtR, "CWT", cwtR]
      : ["CWT", cwtR, "SWT", swtR];
    return `${strongName} correlates more strongly with ${label} (r=${fmtR(strongR)}) than ${weakName} (r=${fmtR(weakR)})`;
  });
  return `At ${subject}, ${clauses.join("; ")}.`;
}

function LowSampleBadge() {
  return (
    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-orange-soft text-orange-strong border border-orange/30">
      Low sample count (n&lt;20) -- indicative only
    </span>
  );
}

function CorrelationBarChart({ pairs }: { pairs: { label: string; cwtR: number | null; swtR: number | null }[] }) {
  const data = pairs.map((p) => ({ name: p.label, CWT: p.cwtR ?? 0, SWT: p.swtR ?? 0 }));
  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }} barGap={4}>
          <CartesianGrid stroke={colors.gridLine} vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: colors.inkMuted, fontSize: 12 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
          />
          <YAxis
            domain={[-1, 1]}
            tick={{ fill: colors.inkMuted, fontSize: 12 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
            label={{ value: "Pearson r", angle: -90, position: "insideLeft", fill: colors.inkMuted, fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{
              background: colors.surface,
              border: `1px solid ${colors.border}`,
              borderRadius: 10,
              fontSize: 12,
              color: colors.ink,
              boxShadow: "0 4px 12px rgba(15,23,42,0.08)",
            }}
            cursor={{ fill: colors.surfaceSunken }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: colors.inkMuted }} iconType="circle" />
          <Bar dataKey="CWT" fill={colors.accentDeep} radius={[4, 4, 0, 0]} maxBarSize={36} />
          <Bar dataKey="SWT" fill={colors.reservoir} radius={[4, 4, 0, 0]} maxBarSize={36} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function pairsFromWell(well: SpectralPetroCorrelationWellResult) {
  return PROPERTY_LABELS.map(({ key, label }) => ({
    label,
    cwtR: well[key].cwt_r,
    swtR: well[key].swt_r,
  }));
}

function WellResultRow({ well }: { well: SpectralPetroCorrelationWellResult }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs py-1.5 border-b border-border last:border-b-0">
      <span className="font-semibold text-ink">{well.well_id}</span>
      <span className="text-ink-faint">
        inline/crossline {well.nearest_inline}/{well.nearest_crossline}
      </span>
      {PROPERTY_LABELS.map(({ key, label }) => (
        <span key={key} className="text-ink-muted">
          {label} CWT={fmtR(well[key].cwt_r)} SWT={fmtR(well[key].swt_r)}
        </span>
      ))}
      {well.low_sample_warning && <LowSampleBadge />}
    </div>
  );
}

/**
 * "CWT vs SWT -- Petrophysical Correlation": at a matched frequency band
 * (CWT sampled at the requested SWT level's own dyadic band-center
 * frequency, instead of CWT's usual adaptive peak), Pearson-correlates
 * each spectral method's amplitude against VSH/PHIE/SWE over a well's tie
 * interval -- see backend/app/services/spectral_petro_correlation_service.py.
 * Reuses the same well-select + Level-slider control patterns as
 * SyntheticSeismogramPage / SpectralDecompView, and the same grouped-
 * bar-chart pattern as WellsBarChart, just plotting correlation
 * coefficients instead of averaged log values.
 */
export default function SpectralPetroCorrelationView() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const [wellId, setWellId] = useState<string | null>(null);
  const [allWells, setAllWells] = useState(false);
  const [level, setLevel] = useState(SWT_DEFAULT_LEVEL);

  const corrQuery = useQuery({
    queryKey: ["seismic-viz-spectral-petro-correlation", wellId, allWells, level],
    queryFn: () => getSpectralPetroCorrelation({ wellId: wellId ?? undefined, allWells, swtLevel: level }),
    enabled: allWells || Boolean(wellId),
    retry: false,
  });

  const data = corrQuery.data;
  const singleWell = data && data.mode === "single" ? data.wells[0] : null;

  const chartPairs = singleWell
    ? pairsFromWell(singleWell)
    : data?.averages
      ? PROPERTY_LABELS.map(({ key, label }) => ({
          label,
          cwtR: data.averages![key].cwt_r,
          swtR: data.averages![key].swt_r,
        }))
      : null;

  const summarySubject = singleWell
    ? singleWell.well_id
    : data
      ? `the average across ${data.wells.length} well(s)`
      : "";

  return (
    <div className="space-y-3">
      <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
        Compares CWT and SWT at a matched frequency band -- CWT is sampled at the SWT level's own
        band-center frequency instead of its usual adaptive peak, so correlation strength against
        VSH/PHIE/SWE reflects how diagnostic each spectral method is of that property at this well,
        like-for-like rather than an apples-to-oranges comparison.
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5 disabled:opacity-50"
          value={wellId ?? ""}
          onChange={(e) => setWellId(e.target.value || null)}
          disabled={allWells}
        >
          <option value="">Select well…</option>
          {wellsQuery.data?.map((w) => (
            <option key={w.well_id} value={w.well_id}>
              {w.well_id}
            </option>
          ))}
        </select>

        <button
          onClick={() => setAllWells((v) => !v)}
          className={`text-xs font-semibold px-3.5 py-1.5 rounded-full border transition-all ${
            allWells
              ? "bg-brand-gradient text-white border-transparent shadow-card"
              : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
          }`}
        >
          All Wells
        </button>
      </div>

      <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
        Level
        <input
          type="range"
          min={SWT_MIN_LEVEL}
          max={SWT_MAX_LEVEL}
          step={1}
          value={level}
          onChange={(e) => setLevel(Number(e.target.value))}
          className="w-56 accent-accent"
        />
        <input
          type="number"
          min={SWT_MIN_LEVEL}
          max={SWT_MAX_LEVEL}
          step={1}
          value={level}
          onChange={(e) => setLevel(Number(e.target.value))}
          className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
        />
        <span className="text-ink-faint font-normal">
          {data
            ? `Comparing SWT Level ${data.swt_level} (${data.swt_band_hz[0].toFixed(0)}-${data.swt_band_hz[1].toFixed(0)} Hz) against CWT @ ${data.cwt_frequency_hz.toFixed(0)} Hz`
            : `levels ${SWT_MIN_LEVEL}-${SWT_MAX_LEVEL}`}
        </span>
      </label>

      {!allWells && !wellId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a well (or toggle All Wells) to compute the CWT/SWT petrophysical correlation.
        </div>
      )}

      {corrQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {corrQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Correlation failed: {errorMessage(corrQuery.error)}
        </div>
      )}

      {data && chartPairs && (
        <div className="space-y-3">
          {singleWell && (
            <div className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
              <span>
                Nearest inline/crossline: {singleWell.nearest_inline} / {singleWell.nearest_crossline}
              </span>
              <span>
                {singleWell.tie_method === "manual_override"
                  ? "Distance: manual override"
                  : singleWell.distance_m !== null
                    ? `Distance: ${singleWell.distance_m.toFixed(0)} m`
                    : "Distance: n/a"}
              </span>
              {singleWell.low_sample_warning && <LowSampleBadge />}
            </div>
          )}

          <CorrelationBarChart pairs={chartPairs} />

          <p className="text-sm text-ink-muted">{buildSummary(summarySubject, chartPairs)}</p>

          {data.mode === "all_wells" && (
            <div className="bg-surface border border-border rounded-xl p-3 shadow-card">
              <h4 className="text-xs font-semibold text-ink mb-1">Per-well results ({data.wells.length})</h4>
              {data.wells.length === 0 ? (
                <p className="text-xs text-ink-faint">No wells with both a resolvable tie and DT/petrophysical logs.</p>
              ) : (
                data.wells.map((w) => <WellResultRow key={w.well_id} well={w} />)
              )}
              {data.skipped_well_ids.length > 0 && (
                <p className="text-xs text-ink-faint mt-2">
                  Skipped ({data.skipped_well_ids.length}, no resolvable tie or missing logs): {data.skipped_well_ids.join(", ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
