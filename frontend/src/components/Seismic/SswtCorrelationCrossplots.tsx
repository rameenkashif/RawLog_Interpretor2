import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SswtCorrelationPair, SswtCorrelationScatter } from "@/api/types";
import { useChartColors } from "@/styles/tokens";

type PropertyKey = "vsh" | "phie" | "swe";

const PROPERTIES: { key: PropertyKey; label: string }[] = [
  { key: "vsh", label: "VSH" },
  { key: "phie", label: "PHIE" },
  { key: "swe", label: "SWE" },
];

function fmtR(r: number | null): string {
  return r === null ? "n/a" : r.toFixed(2);
}

/** Min-max normalize to [0, 1] for DISPLAY only -- CWT energy and SSWT
 * amplitude are on independently meaningful, generally different scales
 * (SSWT reassigns coefficients rather than just filtering CWT's), so
 * overlaying them raw on one y-axis would flatten whichever has the
 * smaller range. Same "independently normalize each series so it's
 * visible, not because the raw values are directly comparable" pattern
 * already used for the synthetic-vs-real trace overlays in
 * WellTieView.tsx / SyntheticTraceOverlay.tsx. */
function normalize(values: number[]): number[] {
  if (values.length === 0) return values;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (range <= 1e-12) return values.map(() => 0.5);
  return values.map((v) => (v - min) / range);
}

function buildPoints(
  propertyValues: (number | null)[],
  amplitude: number[],
  normalizedAmplitude: number[],
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i < propertyValues.length; i++) {
    const prop = propertyValues[i];
    if (prop === null || !Number.isFinite(amplitude[i])) continue;
    points.push({ x: prop, y: normalizedAmplitude[i] });
  }
  return points;
}

/**
 * Per-property (VSH/PHIE/SWE) crossplot of the property value against CWT
 * and SSWT amplitude, from the exact same paired samples the Pearson r in
 * the bar chart above was computed from -- so the correlation strength
 * (or lack of it) is visible as point spread, not just a single number.
 */
export default function SswtCorrelationCrossplots({
  scatter,
  vsh,
  phie,
  swe,
}: {
  scatter: SswtCorrelationScatter;
  vsh: SswtCorrelationPair;
  phie: SswtCorrelationPair;
  swe: SswtCorrelationPair;
}) {
  const colors = useChartColors();
  const pairs: Record<PropertyKey, SswtCorrelationPair> = { vsh, phie, swe };

  const cwtNorm = normalize(scatter.cwt_amplitude);
  const sswtNorm = normalize(scatter.sswt_amplitude);

  return (
    <div className="space-y-3">
      <p className="text-xs text-ink-faint">
        Each point is one depth sample in the well's tie interval. Amplitude is independently
        normalized per method (0-1) for display, since CWT energy and SSWT's reassigned
        amplitude are on different scales -- only the correlation shape/spread is comparable,
        not the raw axis values.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {PROPERTIES.map(({ key, label }) => {
          const pair = pairs[key];
          const cwtPoints = buildPoints(scatter[key], scatter.cwt_amplitude, cwtNorm);
          const sswtPoints = buildPoints(scatter[key], scatter.sswt_amplitude, sswtNorm);
          return (
            <div key={key} className="bg-surface border border-border rounded-xl p-3 shadow-card">
              <p className="text-xs font-semibold text-ink mb-1">
                {label} vs. amplitude
                <span className="ml-2 font-normal text-ink-faint">
                  CWT r={fmtR(pair.cwt_r)}, SSWT r={fmtR(pair.sswt_r)}
                </span>
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid stroke={colors.gridLine} />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name={label}
                    tick={{ fill: colors.inkMuted, fontSize: 11 }}
                    axisLine={{ stroke: colors.borderStrong }}
                    tickLine={false}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Normalized amplitude"
                    domain={[0, 1]}
                    tick={{ fill: colors.inkMuted, fontSize: 11 }}
                    axisLine={{ stroke: colors.borderStrong }}
                    tickLine={false}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3", stroke: colors.borderStrong }}
                    contentStyle={{
                      background: colors.surface,
                      border: `1px solid ${colors.border}`,
                      borderRadius: 10,
                      fontSize: 12,
                      color: colors.ink,
                      boxShadow: "0 4px 12px rgba(15,23,42,0.08)",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: colors.inkMuted }} iconType="circle" />
                  <Scatter name="CWT" data={cwtPoints} fill={colors.accentDeep} opacity={0.7} />
                  <Scatter name="SSWT" data={sswtPoints} fill={colors.reservoir} opacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}
