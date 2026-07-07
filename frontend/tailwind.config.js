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
        // Light-theme design tokens (see src/styles/tokens.ts for the JS/TS mirror)
        surface: {
          DEFAULT: "#FFFFFF",
          muted: "#F8F9FB",
          sunken: "#F1F3F6",
        },
        border: {
          DEFAULT: "#E4E7EC",
          strong: "#D0D5DD",
        },
        ink: {
          DEFAULT: "#1A1A1A",
          muted: "#4B5563",
          faint: "#8A94A6",
        },
        accent: {
          DEFAULT: "#2563EB",
          soft: "#EFF4FF",
          strong: "#1D4ED8",
        },
        pay: "#16A34A",
        reservoir: "#F59E0B",
        nonreservoir: "#94A3B8",
        danger: "#DC2626",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
