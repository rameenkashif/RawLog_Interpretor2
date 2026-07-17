/**
 * tokens.ts
 * ---------
 * Theme-reactive design tokens for chart libraries (Plotly, Recharts) which
 * can't read CSS custom properties or Tailwind classes and need raw hex
 * values passed at render time.
 *
 * lightColors/darkColors MUST mirror the RGB-triplet custom properties in
 * src/styles/index.css (:root / :root.dark) and the color scale in
 * tailwind.config.js -- there is no single source of truth shared at build
 * time between CSS and these hex literals, so keep all three in sync by
 * hand when the palette changes. DOM/Tailwind-styled elements get dark mode
 * for free via the CSS variables; only chart code needs the hooks below.
 */

import { useTheme } from "@/theme/ThemeContext";

export const lightColors = {
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

export const darkColors: typeof lightColors = {
  surface: "#0B1220",
  surfaceMuted: "#111827",
  surfaceSunken: "#1E293B",
  border: "#1E293B",
  borderStrong: "#334155",
  ink: "#E2E8F0",
  inkMuted: "#94A3B8",
  inkFaint: "#64748B",
  accent: "#3B82F6",
  accentSoft: "#1E293B",
  accentStrong: "#60A5FA",
  accentDeep: "#1D4ED8",
  orange: "#FB923C",
  orangeSoft: "#3F2611",
  orangeStrong: "#FDBA74",
  pay: "#4ADE80",
  reservoir: "#FBBF24",
  nonReservoir: "#64748B",
  danger: "#F87171",
  gridLine: "#1E293B",
};

export type ChartColors = typeof lightColors;

/** Current theme's chart color palette -- use inside any component that
 * passes literal colors to Plotly/Recharts. */
export function useChartColors(): ChartColors {
  const { theme } = useTheme();
  return theme === "dark" ? darkColors : lightColors;
}

/** Zone code -> color, matching backend app/petrophysics.py ZONE_* constants.
 * These stay semantic (green/amber/gray) rather than brand blue/orange,
 * since they encode a standard petrophysical interpretation meaning
 * (pay / reservoir non-pay / non-reservoir) that shouldn't be reskinned. */
export function useZoneColors(): Record<number, string> {
  const colors = useChartColors();
  return { 1: colors.pay, 2: colors.reservoir, 3: colors.nonReservoir };
}

export const zoneLabels: Record<number, string> = {
  1: "Pay",
  2: "Reservoir (non-pay)",
  3: "Non-reservoir",
};

/** Alternating blue/orange palette for multi-series charts (bar charts, etc.) */
export function useBrandSeriesColors(): string[] {
  const colors = useChartColors();
  return [colors.accent, colors.orange, colors.accentDeep, colors.reservoir];
}

/**
 * Base Plotly layout shared by every chart in the app. Plotly defaults to a
 * dark-friendly template in some contexts, so every chart must spread this
 * in explicitly rather than relying on Plotly's own default template.
 */
export function usePlotlyLayout() {
  const colors = useChartColors();
  return {
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
}
