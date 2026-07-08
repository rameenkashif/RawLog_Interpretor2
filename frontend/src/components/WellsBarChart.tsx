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
import type { WellSummary } from "@/api/types";
import { colors } from "@/styles/tokens";

/**
 * Bar/column chart comparing key metrics (avg VSH/PHIE/SWE) across wells.
 * Explicit light styling: white background, light-gray gridlines -- never
 * relies on Recharts' default theme, which can look washed out on white.
 * Blue/orange brand palette, per the product's visual theme.
 */
export default function WellsBarChart({ wells }: { wells: WellSummary[] }) {
  const data = wells.map((w) => ({
    name: w.well_id,
    "Avg VSH": w.avg_vsh !== null ? +(w.avg_vsh * 100).toFixed(1) : 0,
    "Avg PHIE": w.avg_phie !== null ? +(w.avg_phie * 100).toFixed(1) : 0,
    "Avg SWE": w.avg_swe !== null ? +(w.avg_swe * 100).toFixed(1) : 0,
  }));

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-ink">Key Metrics by Well</h3>
        <span className="text-xs text-ink-faint">% of interval</span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
          barGap={4}
        >
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
          <Legend
            wrapperStyle={{ fontSize: 12, color: colors.inkMuted }}
            iconType="circle"
          />
          <Bar
            dataKey="Avg VSH"
            fill={colors.accentDeep}
            radius={[4, 4, 0, 0]}
            maxBarSize={26}
          />
          <Bar
            dataKey="Avg PHIE"
            fill={colors.accent}
            radius={[4, 4, 0, 0]}
            maxBarSize={26}
          />
          <Bar
            dataKey="Avg SWE"
            fill={colors.orange}
            radius={[4, 4, 0, 0]}
            maxBarSize={26}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
