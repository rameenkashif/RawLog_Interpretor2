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
import { getSpectralPetroCorrelation, getSswtPetroCorrelation, listWells } from "@/api/client";
import type {
  SpectralPetroCorrelationWellResult,
  SswtPetroCorrelationWellResult,
} from "@/api/types";
import { colors } from "@/styles/tokens";

type CompareMethod = "swt" | "sswt";

const SWT_MIN_LEVEL = 1;
const SWT_MAX_LEVEL = 6;
const SWT_DEFAULT_LEVEL = 3;
const SSWT_DEFAULT_FREQUENCY_HZ = 30;

const PROPERTY_LABELS: { key: "vsh" | "phie" | "swe"; label: string }[] = [
  { key: "vsh", label: "VSH" },
  { key: "phie", label: "PHIE" },
  { key: "swe", label: "SWE" },
];

/** Normalized shape both comparison modes render into -- "other" is SWT's
 * swt_r or SSWT's sswt_r, whichever mode is active, so the chart/summary/
 * per-well list below don't need to know which one they're looking at. */
type ComparisonPair = { label: string; cwtR: number | null; otherR: number | null };

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
function buildSummary(subject: string, otherLabel: string, pairs: ComparisonPair[]): string {
  const clauses = pairs.map(({ label, cwtR, otherR }) => {
    if (cwtR === null && otherR === null) return `insufficient data for ${label}`;
    if (cwtR === null) return `only ${otherLabel} correlates with ${label} (r=${fmtR(otherR)})`;
    if (otherR === null) return `only CWT correlates with ${label} (r=${fmtR(cwtR)})`;
    const otherStronger = Math.abs(otherR) >= Math.abs(cwtR);
    const [strongName, strongR, weakName, weakR] = otherStronger
      ? [otherLabel, otherR, "CWT", cwtR]
      : ["CWT", cwtR, otherLabel, otherR];
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

function CorrelationBarChart({ otherLabel, pairs }: { otherLabel: string; pairs: ComparisonPair[] }) {
  const data = pairs.map((p) => ({ name: p.label, CWT: p.cwtR ?? 0, [otherLabel]: p.otherR ?? 0 }));
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
          <Bar dataKey={otherLabel} fill={colors.reservoir} radius={[4, 4, 0, 0]} maxBarSize={36} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function swtPairsFromWell(well: SpectralPetroCorrelationWellResult): ComparisonPair[] {
  return PROPERTY_LABELS.map(({ key, label }) => ({
    label,
    cwtR: well[key].cwt_r,
    otherR: well[key].swt_r,
  }));
}

function sswtPairsFromWell(well: SswtPetroCorrelationWellResult): ComparisonPair[] {
  return PROPERTY_LABELS.map(({ key, label }) => ({
    label,
    cwtR: well[key].cwt_r,
    otherR: well[key].sswt_r,
  }));
}

function SwtWellResultRow({ well }: { well: SpectralPetroCorrelationWellResult }) {
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

function SswtWellResultRow({ well }: { well: SswtPetroCorrelationWellResult }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs py-1.5 border-b border-border last:border-b-0">
      <span className="font-semibold text-ink">{well.well_id}</span>
      <span className="text-ink-faint">
        inline/crossline {well.nearest_inline}/{well.nearest_crossline}
      </span>
      {PROPERTY_LABELS.map(({ key, label }) => (
        <span key={key} className="text-ink-muted">
          {label} CWT={fmtR(well[key].cwt_r)} SSWT={fmtR(well[key].sswt_r)}
        </span>
      ))}
      {well.low_sample_warning && <LowSampleBadge />}
    </div>
  );
}

/**
 * "CWT vs SWT" / "CWT vs SSWT" -- Petrophysical Correlation, toggled by
 * comparison method. CWT-vs-SWT matches CWT to the requested SWT level's
 * dyadic band-center frequency (SWT has no continuous frequency axis, see
 * backend/app/services/spectral_petro_correlation_service.get_correlation).
 * CWT-vs-SSWT instead snaps BOTH methods independently to their own
 * nearest bin to a single requested frequency, since SSWT (like CWT) has
 * a continuous -- just much finer-grained -- frequency axis; see that
 * module's get_sswt_correlation. Reuses the same well-select + slider
 * control patterns as SyntheticSeismogramPage / SpectralDecompView, and
 * the same grouped-bar-chart pattern as WellsBarChart either way, just
 * plotting correlation coefficients instead of averaged log values.
 */
export default function SpectralPetroCorrelationView() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const [wellId, setWellId] = useState<string | null>(null);
  const [allWells, setAllWells] = useState(false);
  const [compareMethod, setCompareMethod] = useState<CompareMethod>("swt");
  const [level, setLevel] = useState(SWT_DEFAULT_LEVEL);
  const [frequencyHz, setFrequencyHz] = useState(SSWT_DEFAULT_FREQUENCY_HZ);

  const isSswt = compareMethod === "sswt";
  const enabled = allWells || Boolean(wellId);

  const swtQuery = useQuery({
    queryKey: ["seismic-viz-spectral-petro-correlation", wellId, allWells, level],
    queryFn: () => getSpectralPetroCorrelation({ wellId: wellId ?? undefined, allWells, swtLevel: level }),
    enabled: enabled && !isSswt,
    retry: false,
  });

  const sswtQuery = useQuery({
    queryKey: ["seismic-viz-spectral-petro-correlation-sswt", wellId, allWells, frequencyHz],
    queryFn: () => getSswtPetroCorrelation({ wellId: wellId ?? undefined, allWells, frequencyHz }),
    enabled: enabled && isSswt,
    retry: false,
  });

  const corrQuery = isSswt ? sswtQuery : swtQuery;

  const swtData = swtQuery.data;
  const sswtData = sswtQuery.data;

  const singleSwtWell = swtData && swtData.mode === "single" ? swtData.wells[0] : null;
  const singleSswtWell = sswtData && sswtData.mode === "single" ? sswtData.wells[0] : null;

  const otherLabel = isSswt ? "SSWT" : "SWT";

  const chartPairs: ComparisonPair[] | null = isSswt
    ? singleSswtWell
      ? sswtPairsFromWell(singleSswtWell)
      : sswtData?.averages
        ? PROPERTY_LABELS.map(({ key, label }) => ({
            label,
            cwtR: sswtData.averages![key].cwt_r,
            otherR: sswtData.averages![key].sswt_r,
          }))
        : null
    : singleSwtWell
      ? swtPairsFromWell(singleSwtWell)
      : swtData?.averages
        ? PROPERTY_LABELS.map(({ key, label }) => ({
            label,
            cwtR: swtData.averages![key].cwt_r,
            otherR: swtData.averages![key].swt_r,
          }))
        : null;

  const wellCount = isSswt ? sswtData?.wells.length : swtData?.wells.length;
  const summarySubject = isSswt
    ? singleSswtWell
      ? singleSswtWell.well_id
      : sswtData
        ? `the average across ${wellCount} well(s)`
        : ""
    : singleSwtWell
      ? singleSwtWell.well_id
      : swtData
        ? `the average across ${wellCount} well(s)`
        : "";

  return (
    <div className="space-y-3">
      <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
        {isSswt ? (
          <>
            Compares CWT and SSWT at a matched frequency -- both are independently snapped to their
            own nearest available bin to the same requested frequency, so correlation strength
            against VSH/PHIE/SWE reflects whether SSWT's sharpened, reassigned frequency estimate is
            more diagnostic of that property than the plain CWT it's derived from.
          </>
        ) : (
          <>
            Compares CWT and SWT at a matched frequency band -- CWT is sampled at the SWT level's own
            band-center frequency instead of its usual adaptive peak, so correlation strength against
            VSH/PHIE/SWE reflects how diagnostic each spectral method is of that property at this well,
            like-for-like rather than an apples-to-oranges comparison.
          </>
        )}
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

        <div className="flex gap-1.5">
          {(["swt", "sswt"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setCompareMethod(m)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all uppercase ${
                compareMethod === m
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              vs {m}
            </button>
          ))}
        </div>
      </div>

      {isSswt ? (
        <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
          Frequency (Hz)
          <input
            type="range"
            min={0}
            max={sswtData?.nyquist_hz ?? 250}
            step={0.5}
            value={frequencyHz}
            onChange={(e) => setFrequencyHz(Number(e.target.value))}
            className="w-56 accent-accent"
          />
          <input
            type="number"
            min={0}
            max={sswtData?.nyquist_hz ?? 250}
            step={0.5}
            value={frequencyHz}
            onChange={(e) => setFrequencyHz(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
          <span className="text-ink-faint font-normal">
            {sswtData
              ? `CWT @ ${sswtData.cwt_frequency_hz.toFixed(1)} Hz, SSWT @ ${sswtData.sswt_frequency_hz.toFixed(1)} Hz (nearest bins to ${sswtData.requested_frequency_hz.toFixed(1)} Hz)`
              : "each method snaps independently to its own nearest bin"}
          </span>
        </label>
      ) : (
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
            {swtData
              ? `Comparing SWT Level ${swtData.swt_level} (${swtData.swt_band_hz[0].toFixed(0)}-${swtData.swt_band_hz[1].toFixed(0)} Hz) against CWT @ ${swtData.cwt_frequency_hz.toFixed(0)} Hz`
              : `levels ${SWT_MIN_LEVEL}-${SWT_MAX_LEVEL}`}
          </span>
        </label>
      )}

      {!allWells && !wellId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a well (or toggle All Wells) to compute the CWT/{otherLabel} petrophysical correlation.
        </div>
      )}

      {corrQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {corrQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Correlation failed: {errorMessage(corrQuery.error)}
        </div>
      )}

      {chartPairs && (
        <div className="space-y-3">
          {!isSswt && singleSwtWell && (
            <div className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
              <span>
                Nearest inline/crossline: {singleSwtWell.nearest_inline} / {singleSwtWell.nearest_crossline}
              </span>
              <span>
                {singleSwtWell.tie_method === "manual_override"
                  ? "Distance: manual override"
                  : singleSwtWell.distance_m !== null
                    ? `Distance: ${singleSwtWell.distance_m.toFixed(0)} m`
                    : "Distance: n/a"}
              </span>
              {singleSwtWell.low_sample_warning && <LowSampleBadge />}
            </div>
          )}
          {isSswt && singleSswtWell && (
            <div className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
              <span>
                Nearest inline/crossline: {singleSswtWell.nearest_inline} / {singleSswtWell.nearest_crossline}
              </span>
              <span>
                {singleSswtWell.tie_method === "manual_override"
                  ? "Distance: manual override"
                  : singleSswtWell.distance_m !== null
                    ? `Distance: ${singleSswtWell.distance_m.toFixed(0)} m`
                    : "Distance: n/a"}
              </span>
              {singleSswtWell.low_sample_warning && <LowSampleBadge />}
            </div>
          )}

          <CorrelationBarChart otherLabel={otherLabel} pairs={chartPairs} />

          <p className="text-sm text-ink-muted">{buildSummary(summarySubject, otherLabel, chartPairs)}</p>

          {!isSswt && swtData?.mode === "all_wells" && (
            <div className="bg-surface border border-border rounded-xl p-3 shadow-card">
              <h4 className="text-xs font-semibold text-ink mb-1">Per-well results ({swtData.wells.length})</h4>
              {swtData.wells.length === 0 ? (
                <p className="text-xs text-ink-faint">No wells with both a resolvable tie and DT/petrophysical logs.</p>
              ) : (
                swtData.wells.map((w) => <SwtWellResultRow key={w.well_id} well={w} />)
              )}
              {swtData.skipped_well_ids.length > 0 && (
                <p className="text-xs text-ink-faint mt-2">
                  Skipped ({swtData.skipped_well_ids.length}, no resolvable tie or missing logs): {swtData.skipped_well_ids.join(", ")}
                </p>
              )}
            </div>
          )}
          {isSswt && sswtData?.mode === "all_wells" && (
            <div className="bg-surface border border-border rounded-xl p-3 shadow-card">
              <h4 className="text-xs font-semibold text-ink mb-1">Per-well results ({sswtData.wells.length})</h4>
              {sswtData.wells.length === 0 ? (
                <p className="text-xs text-ink-faint">No wells with both a resolvable tie and DT/petrophysical logs.</p>
              ) : (
                sswtData.wells.map((w) => <SswtWellResultRow key={w.well_id} well={w} />)
              )}
              {sswtData.skipped_well_ids.length > 0 && (
                <p className="text-xs text-ink-faint mt-2">
                  Skipped ({sswtData.skipped_well_ids.length}, no resolvable tie or missing logs): {sswtData.skipped_well_ids.join(", ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
