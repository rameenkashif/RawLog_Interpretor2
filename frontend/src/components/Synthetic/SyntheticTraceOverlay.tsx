import { useState } from "react";
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
import type { SyntheticSeismogramResponse } from "@/api/types";
import { colors } from "@/styles/tokens";

type Domain = "time" | "frequency";

/** Synthetic-vs-real trace overlay + tie quality stats, same chart pattern
 * as the Seismic Visualization module's WellTieView.tsx. Toggles between
 * the time-domain trace overlay and the frequency-domain amplitude
 * spectrum of the same two traces (real_trace / shifted_synthetic) --
 * same underlying convolution result, just two ways to look at it (see
 * synthetic_seismogram_service.py's real_trace_spectrum/synthetic_spectrum,
 * an FFT of the exact arrays plotted in the time-domain view). */
export default function SyntheticTraceOverlay({ result }: { result: SyntheticSeismogramResponse }) {
  const [domain, setDomain] = useState<Domain>("time");

  const timeData = result.seismic_twt_ms.map((t, i) => ({
    x: t,
    synthetic: result.shifted_synthetic[i],
    real: result.real_trace[i],
  }));
  const freqData = result.trace_spectrum_freq_hz.map((f, i) => ({
    x: f,
    synthetic: result.synthetic_spectrum_amplitude[i],
    real: result.real_trace_spectrum_amplitude[i],
  }));
  const chartData = domain === "time" ? timeData : freqData;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
          <span>
            Nearest inline/crossline: {result.nearest_inline} / {result.nearest_crossline}
          </span>
          <span>
            {result.tie_method === "manual_override"
              ? "Distance: manual override"
              : `Distance: ${result.distance_m?.toFixed(0)} m`}
          </span>
          <span>Best shift: {result.best_shift_ms.toFixed(1)} ms</span>
          <span className={result.correlation > 0.5 ? "text-green-600" : "text-orange-600"}>
            Correlation: {result.correlation.toFixed(3)}
          </span>
          {result.polarity === -1 && (
            <span className="text-orange-600">Polarity: reversed</span>
          )}
        </div>

        <div className="flex gap-1.5">
          {(["time", "frequency"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDomain(d)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all uppercase ${
                domain === d
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {result.tie_search_note && (
        <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
          {result.tie_search_note}
        </div>
      )}

      <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid stroke={colors.gridLine} />
            <XAxis
              dataKey="x"
              stroke={colors.borderStrong}
              tick={{ fill: colors.inkMuted, fontSize: 11 }}
              label={{
                value: domain === "time" ? "Two-Way Time (ms)" : "Frequency (Hz)",
                position: "insideBottom",
                offset: -5,
                fill: colors.inkMuted,
              }}
            />
            <YAxis
              stroke={colors.borderStrong}
              tick={{ fill: colors.inkMuted, fontSize: 11 }}
              label={{
                value: domain === "time" ? "Amplitude" : "Spectral Amplitude",
                angle: -90,
                position: "insideLeft",
                fill: colors.inkMuted,
                fontSize: 11,
              }}
            />
            <Tooltip contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }} />
            <Legend />
            <Line type="monotone" dataKey="real" name="Real trace" stroke={colors.accent} dot={false} strokeWidth={1.5} />
            <Line
              type="monotone"
              dataKey="synthetic"
              name="Synthetic (shifted)"
              stroke={colors.orange}
              dot={false}
              strokeWidth={1.5}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
