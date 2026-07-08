/** @type {import('tailwindcss').Config} */
export default {
  // IMPORTANT: darkMode is intentionally left at its default ('media') OFF by
  // never being referenced -- we do not use Tailwind's `dark:` variant
  // anywhere in this app. The UI is light-mode only, per product requirement.
  darkMode: ["class"], // only ever toggled manually if a future feature needs it; default state is always light
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Light-theme design tokens (see src/styles/tokens.ts for the JS/TS mirror).
        // Brand palette: blue (primary) + orange (secondary), on a white/near-white
        // surface -- never a dark background.
        surface: {
          DEFAULT: "#FFFFFF",
          muted: "#F8FAFC",
          sunken: "#EEF2F8",
        },
        border: {
          DEFAULT: "#E2E8F0",
          strong: "#CBD5E1",
        },
        ink: {
          DEFAULT: "#0F172A",
          muted: "#475569",
          faint: "#94A3B8",
        },
        accent: {
          DEFAULT: "#2563EB",
          soft: "#EFF6FF",
          strong: "#1D4ED8",
          deep: "#1E3A8A",
        },
        orange: {
          DEFAULT: "#F97316",
          soft: "#FFF3E8",
          strong: "#C2410C",
        },
        pay: "#16A34A",
        reservoir: "#F59E0B",
        nonreservoir: "#94A3B8",
        danger: "#DC2626",
      },
      backgroundImage: {
        "brand-gradient":
          "linear-gradient(135deg, #2563EB 0%, #1D4ED8 55%, #F97316 130%)",
        "brand-gradient-soft":
          "linear-gradient(135deg, #EFF6FF 0%, #FFF3E8 100%)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.06)",
        "card-hover":
          "0 4px 12px rgba(15, 23, 42, 0.08), 0 2px 4px rgba(15, 23, 42, 0.06)",
      },
    },
  },
  plugins: [],
};
