/**
 * tokens.ts
 * ---------
 * Light-theme design tokens shared across chart libraries (Plotly, Recharts)
 * which can't read Tailwind classes directly and need raw hex values.
 *
 * These MUST mirror tailwind.config.js `theme.extend.colors`. There is no
 * dark-mode equivalent by design -- see the project's non-negotiable
 * light-mode UI requirement.
 */

export const colors = {
  surface: "#FFFFFF",
  surfaceMuted: "#F8F9FB",
  surfaceSunken: "#F1F3F6",
  border: "#E4E7EC",
  borderStrong: "#D0D5DD",
  ink: "#1A1A1A",
  inkMuted: "#4B5563",
  inkFaint: "#8A94A6",
  accent: "#2563EB",
  accentSoft: "#EFF4FF",
  accentStrong: "#1D4ED8",
  pay: "#16A34A",
  reservoir: "#F59E0B",
  nonReservoir: "#94A3B8",
  danger: "#DC2626",
  gridLine: "#EAECF0",
};

/** Zone code -> color, matching backend app/petrophysics.py ZONE_* constants. */
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
