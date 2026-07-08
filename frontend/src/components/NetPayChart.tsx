import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WellSummary } from "@/api/types";
import { colors } from "@/styles/tokens";

/**
 * Horizontal net-pay-thickness comparison across wells -- gives an
 * immediate "which wells matter most" read at the top of the dashboard.
 * Uses `net_pay_thickness` already returned by GET /dashboard/summary
 * (no backend changes needed).
 */
export default function NetPayChart({ wells }: { wells: WellSummary[] }) {
  const data = [...wells]
    .map((w) => ({ name: w.well_id, netPay: w.net_pay_thickness ?? 0 }))
    .sort((a, b) => b.netPay - a.netPay);

  const maxPay = Math.max(...data.map((d) => d.netPay), 1);

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-ink">
          Net Pay Thickness by Well
        </h3>
        <span className="text-xs text-ink-faint">metres</span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
        >
          <CartesianGrid stroke={colors.gridLine} horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: colors.inkMuted, fontSize: 12 }}
            axisLine={{ stroke: colors.borderStrong }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={64}
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
            formatter={(value: number) => [`${value.toFixed(1)} m`, "Net Pay"]}
          />
          <Bar dataKey="netPay" radius={[0, 6, 6, 0]} maxBarSize={22}>
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={d.netPay === maxPay ? colors.orange : colors.accent}
                fillOpacity={0.55 + 0.45 * (d.netPay / maxPay)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
