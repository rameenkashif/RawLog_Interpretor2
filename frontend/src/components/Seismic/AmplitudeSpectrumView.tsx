import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getAmplitudeSpectrum } from "@/api/client";
import type { SurveyInfoResponse } from "@/api/types";
import { useChartColors } from "@/styles/tokens";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Frequency-domain QC: average amplitude spectrum either across the whole
 * volume (a systematic trace sample) or a single inline, with dominant
 * frequency / bandwidth / S/N-proxy stat callouts. A 1D chart, so Recharts
 * (already used elsewhere in the app) fits fine here.
 */
export default function AmplitudeSpectrumView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const colors = useChartColors();
  const [scope, setScope] = useState<"volume" | "inline">("volume");
  const [inlineNumber, setInlineNumber] = useState(surveyInfo.inline_min);

  const query = useQuery({
    queryKey: ["seismic-viz-spectrum", scope, scope === "inline" ? inlineNumber : null],
    queryFn: () => getAmplitudeSpectrum(scope === "inline" ? inlineNumber : null),
  });

  const chartData = query.data?.freq_hz.map((f, i) => ({
    freq_hz: f,
    amplitude: query.data!.amplitude[i],
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(["volume", "inline"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setScope(s)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                scope === s
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {s === "volume" ? "Whole volume" : "Single inline"}
            </button>
          ))}
        </div>

        {scope === "inline" && (
          <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
            Inline
            <input
              type="number"
              min={surveyInfo.inline_min}
              max={surveyInfo.inline_max}
              value={inlineNumber}
              onChange={(e) => setInlineNumber(Number(e.target.value))}
              className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
            />
            <span className="text-ink-faint font-normal">
              ({surveyInfo.inline_min}-{surveyInfo.inline_max})
            </span>
          </label>
        )}
      </div>

      {query.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}
      {query.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load spectrum: {errorMessage(query.error)}
        </div>
      )}

      {query.data && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatTile label="Traces sampled" value={query.data.n_traces_sampled.toLocaleString()} />
            <StatTile label="Dominant Freq" value={`${query.data.dominant_freq_hz.toFixed(1)} Hz`} />
            <StatTile label="Bandwidth (-3dB)" value={`${query.data.bandwidth_hz.toFixed(1)} Hz`} />
            <StatTile
              label="S/N proxy"
              value={query.data.snr_proxy !== null ? query.data.snr_proxy.toFixed(2) : "—"}
            />
          </div>

          <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
            <ResponsiveContainer width="100%" height={340}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="spectrumFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={colors.accent} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={colors.accent} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={colors.gridLine} />
                <XAxis
                  dataKey="freq_hz"
                  stroke={colors.borderStrong}
                  tick={{ fill: colors.inkMuted, fontSize: 11 }}
                  label={{ value: "Frequency (Hz)", position: "insideBottom", offset: -5, fill: colors.inkMuted }}
                />
                <YAxis
                  stroke={colors.borderStrong}
                  tick={{ fill: colors.inkMuted, fontSize: 11 }}
                  label={{ value: "Avg Amplitude", angle: -90, position: "insideLeft", fill: colors.inkMuted }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }}
                  formatter={(v: number) => v.toFixed(3)}
                />
                <Area type="monotone" dataKey="amplitude" stroke={colors.accent} fill="url(#spectrumFill)" strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <p className="text-xs text-ink-faint">
            S/N proxy is an uncalibrated QC signal (mean in-band amplitude / mean out-of-band
            amplitude around the -3dB bandwidth), not a measured signal-to-noise ratio.
          </p>
        </div>
      )}
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface border border-border rounded-xl px-4 py-3 shadow-card">
      <p className="text-xs font-semibold text-ink-faint uppercase tracking-wide">{label}</p>
      <p className="text-xl font-extrabold text-ink mt-1">{value}</p>
    </div>
  );
}
