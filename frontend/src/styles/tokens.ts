/**
 * tokens.ts
 * ---------
 * Light-theme design tokens shared across chart libraries (Plotly, Recharts)
 * which can't read Tailwind classes directly and need raw hex values.
 *
 * These MUST mirror tailwind.config.js `theme.extend.colors`. Brand palette
 * is blue (primary) + orange (secondary) on a white/near-white surface --
 * there is no dark-mode equivalent by design, per the light-mode UI
 * requirement.
 */

export const colors = {
  surface: "#FFFFFF",
  surfaceMuted: "#F8FAFC",
  surfaceSunken: "#EEF2F8",
  border: "#E2E8F0",
  borderStrong: "#CBD5E1",
  ink: "#0F172A",
  inkMuted: "#475569",
  inkFaint: "#94A3B8",
  accent: "#2563EB",
  accentSoft: "#EFF6FF",
  accentStrong: "#1D4ED8",
  accentDeep: "#1E3A8A",
  orange: "#F97316",
  orangeSoft: "#FFF3E8",
  orangeStrong: "#C2410C",
  pay: "#16A34A",
  reservoir: "#F59E0B",
  nonReservoir: "#94A3B8",
  danger: "#DC2626",
  gridLine: "#EDF1F7",
};

/** Zone code -> color, matching backend app/petrophysics.py ZONE_* constants.
 * These stay semantic (green/amber/gray) rather than brand blue/orange,
 * since they encode a standard petrophysical interpretation meaning
 * (pay / reservoir non-pay / non-reservoir) that shouldn't be reskinned.
 */
export const zoneColors: Record<number, string> = {
  1: colors.pay,
  2: colors.reservoir,
  3: colors.nonReservoir,
};

export const zoneLabels: Record<number, string> = {
  1: "Pay",
  2: "Reservoir (non-pay)",
  3: "Non-reservoir",
};

/** Alternating blue/orange palette for multi-series charts (bar charts, etc.) */
export const brandSeriesColors = [
  colors.accent,
  colors.orange,
  colors.accentDeep,
  colors.reservoir,
];

/**
 * Base Plotly layout shared by every chart in the app. Plotly defaults to a
 * dark-friendly template in some contexts, so every chart must spread this
 * in explicitly rather than relying on Plotly's own default template.
 */
export const plotlyLightLayout = {
  paper_bgcolor: colors.surface,
  plot_bgcolor: colors.surface,
  font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
  xaxis: {
    gridcolor: colors.gridLine,
    zerolinecolor: colors.border,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  },
  yaxis: {
    gridcolor: colors.gridLine,
    zerolinecolor: colors.border,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  },
  margin: { t: 40, r: 20, b: 40, l: 50 },
};
