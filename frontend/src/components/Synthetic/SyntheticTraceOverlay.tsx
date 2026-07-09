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

/** Synthetic-vs-real trace overlay + tie quality stats, same chart pattern
 * as the Seismic Visualization module's WellTieView.tsx. */
export default function SyntheticTraceOverlay({ result }: { result: SyntheticSeismogramResponse }) {
  const chartData = result.seismic_twt_ms.map((t, i) => ({
    twt_ms: t,
    synthetic: result.shifted_synthetic[i],
    real: result.real_trace[i],
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
        <span>
          Nearest inline/crossline: {result.nearest_inline} / {result.nearest_crossline}
        </span>
        <span>Distance: {result.distance_m.toFixed(0)} m</span>
        <span>Best shift: {result.best_shift_ms.toFixed(1)} ms</span>
        <span className={result.correlation > 0.5 ? "text-green-600" : "text-orange-600"}>
          Correlation: {result.correlation.toFixed(3)}
        </span>
      </div>

      <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData}>
            <CartesianGrid stroke={colors.gridLine} />
            <XAxis
              dataKey="twt_ms"
              stroke={colors.borderStrong}
              tick={{ fill: colors.inkMuted, fontSize: 11 }}
              label={{ value: "Two-Way Time (ms)", position: "insideBottom", offset: -5, fill: colors.inkMuted }}
            />
            <YAxis stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 11 }} />
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
