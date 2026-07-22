import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getSpectralPropertyModel } from "@/api/client";
import type {
  SpectralPropertyMethodResult,
  SpectralPropertyModelResponse,
  SpectralPropertyName,
} from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";
import { Badge } from "@/components/Synthetic/QcBadges";

const PROPERTIES: { key: SpectralPropertyName; label: string }[] = [
  { key: "vsh", label: "VSH" },
  { key: "phie", label: "PHIE" },
  { key: "swe", label: "SWE" },
];

function fmtR2(v: number | null): string {
  return v === null ? "n/a" : v.toFixed(3);
}

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function tooltipStyle(colors: ChartColors) {
  return {
    background: colors.surface,
    border: `1px solid ${colors.border}`,
    borderRadius: 10,
    fontSize: 12,
    color: colors.ink,
    boxShadow: "0 4px 12px rgba(15,23,42,0.08)",
  };
}

/**
 * "Multi-frequency SSWT/CWT -> VSH/PHIE/SWE" prediction, validated with
 * leave-one-well-out cross-validation (see backend/app/services/
 * spectral_property_prediction_service.py). Deliberately a POINT-SOURCE
 * validation view, not a volume-wide map -- that's an explicit follow-up
 * gated on this validating with real, held-out skill first, not
 * in-sample fit. status='insufficient_data' is rendered as a plain
 * informational message, never silently skipped or shown as a normal
 * (empty) result.
 */
export default function SpectralPropertyModelView() {
  const colors = useChartColors();
  const [selectedProperty, setSelectedProperty] = useState<SpectralPropertyName>("vsh");
  const query = useQuery({ queryKey: ["spectral-property-model"], queryFn: getSpectralPropertyModel });
  const data = query.data;

  return (
    <div className="space-y-3">
      <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
        Uses SSWT/CWT amplitude across ALL available frequency bins (not one) as features to
        predict VSH/PHIE/SWE, validated with leave-one-well-out cross-validation across wells
        with a usable tie -- not a random depth-sample split, which would badly overstate
        performance since adjacent depth samples are strongly autocorrelated. This is a
        point-source validation only, not a volume-wide prediction: a good LOOCV R² here is a
        prerequisite for, not the same as, a trustworthy spatial map. Well eligibility and time
        alignment here use a direct nearest-trace tie against the active seismic volume (the
        same resolution + full-window frequency/polarity/shift search the Well-to-Seismic Tie
        page uses), so eligible wells here should track that page's high-confidence wells --
        see README.md for details.
      </div>

      {query.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}
      {query.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load: {errorMessage(query.error)}
        </div>
      )}

      {data && data.status === "insufficient_data" && (
        <div className="border border-orange/30 bg-orange-soft/40 text-orange-strong text-sm rounded-xl px-4 py-4">
          {data.message}
        </div>
      )}

      {data && data.status === "validated" && data.results && (
        <div className="space-y-4">
          <LoocvSummaryChart results={data.results} colors={colors} />

          <div className="flex gap-1.5">
            {PROPERTIES.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSelectedProperty(key)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                  selectedProperty === key
                    ? "bg-brand-gradient text-white border-transparent shadow-card"
                    : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <PropertyDetail
            sswt={data.results[selectedProperty].sswt}
            cwt={data.results[selectedProperty].cwt}
            colors={colors}
          />
        </div>
      )}

      {data && <WellsBreakdown eligible={data.eligible_well_ids} excluded={data.excluded_wells} />}
    </div>
  );
}

function LoocvSummaryChart({
  results,
  colors,
}: {
  results: SpectralPropertyModelResponse["results"];
  colors: ChartColors;
}) {
  if (!results) return null;
  const chartData = PROPERTIES.map(({ key, label }) => ({
    name: label,
    SSWT: results[key].sswt?.loocv_r2 ?? null,
    CWT: results[key].cwt?.loocv_r2 ?? null,
  }));

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <p className="text-xs font-semibold text-ink mb-2">Leave-one-well-out R² by property</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }} barGap={4}>
          <CartesianGrid stroke={colors.gridLine} vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: colors.inkMuted, fontSize: 12 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: colors.inkMuted, fontSize: 12 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
            label={{ value: "R² (held-out)", angle: -90, position: "insideLeft", fill: colors.inkMuted, fontSize: 11 }}
          />
          <Tooltip
            contentStyle={tooltipStyle(colors)}
            cursor={{ fill: colors.surfaceSunken }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(3) : "n/a (too few wells)")}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: colors.inkMuted }} iconType="circle" />
          <Bar dataKey="SSWT" fill={colors.accentDeep} radius={[4, 4, 0, 0]} maxBarSize={36} />
          <Bar dataKey="CWT" fill={colors.reservoir} radius={[4, 4, 0, 0]} maxBarSize={36} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-ink-faint mt-2">
        R² can be negative -- it means the model does WORSE than just predicting the training
        mean, a genuine "no generalizable relationship found" result, not a display error.
      </p>
    </div>
  );
}

function PropertyDetail({
  sswt,
  cwt,
  colors,
}: {
  sswt: SpectralPropertyMethodResult | null;
  cwt: SpectralPropertyMethodResult | null;
  colors: ChartColors;
}) {
  if (!sswt && !cwt) {
    return (
      <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
        Neither method had enough valid samples across wells for this property.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      <FeatureImportanceChart title="SSWT feature importance by frequency" result={sswt} colors={colors} />
      <FeatureImportanceChart title="CWT feature importance by frequency" result={cwt} colors={colors} />
      <PerWellTable title="SSWT per-well (held-out) results" result={sswt} />
      <PerWellTable title="CWT per-well (held-out) results" result={cwt} />
    </div>
  );
}

function FeatureImportanceChart({
  title,
  result,
  colors,
}: {
  title: string;
  result: SpectralPropertyMethodResult | null;
  colors: ChartColors;
}) {
  if (!result) {
    return (
      <div className="bg-surface border border-border rounded-xl p-4 shadow-card text-xs text-ink-faint">
        {title}: unavailable (too few valid samples).
      </div>
    );
  }
  const chartData = result.feature_importance
    .slice()
    .sort((a, b) => a.frequency_hz - b.frequency_hz)
    .map((f) => ({ frequency_hz: f.frequency_hz, importance: f.importance }));

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <p className="text-xs font-semibold text-ink mb-2">
        {title}
        <span className="ml-2 font-normal text-ink-faint">
          (from a model fit on all usable wells -- in-sample, for interpretation only, not the
          validation score above)
        </span>
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={colors.gridLine} vertical={false} />
          <XAxis
            dataKey="frequency_hz"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
            label={{ value: "Frequency (Hz)", position: "insideBottom", offset: -2, fill: colors.inkMuted, fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
          />
          <Tooltip contentStyle={tooltipStyle(colors)} formatter={(v: number) => v.toFixed(4)} />
          <Line type="monotone" dataKey="importance" stroke={colors.accentDeep} strokeWidth={2} dot={{ r: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PerWellTable({ title, result }: { title: string; result: SpectralPropertyMethodResult | null }) {
  if (!result) return null;
  return (
    <div className="bg-surface border border-border rounded-xl shadow-card overflow-hidden">
      <p className="text-xs font-semibold text-ink px-4 pt-3 pb-1">{title}</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-muted border-b border-border">
            <th className="text-left px-4 py-2 font-semibold text-ink-muted text-xs">Well</th>
            <th className="text-left px-4 py-2 font-semibold text-ink-muted text-xs">Held-out R²</th>
            <th className="text-left px-4 py-2 font-semibold text-ink-muted text-xs">Samples</th>
          </tr>
        </thead>
        <tbody>
          {result.per_well.map((row) => (
            <tr key={row.well_id} className="border-b border-border last:border-0">
              <td className="px-4 py-2 font-semibold text-accent-strong text-xs">{row.well_id}</td>
              <td className="px-4 py-2 text-ink-muted text-xs">{fmtR2(row.r2)}</td>
              <td className="px-4 py-2 text-ink-faint text-xs">{row.n_samples}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WellsBreakdown({
  eligible,
  excluded,
}: {
  eligible: string[];
  excluded: SpectralPropertyModelResponse["excluded_wells"];
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {eligible.map((id) => (
        <Badge key={id} tone="green">
          {id} eligible
        </Badge>
      ))}
      {excluded.map((w) => (
        <Badge key={w.well_id} tone="orange">
          {w.well_id} excluded — {w.reason}
        </Badge>
      ))}
    </div>
  );
}
