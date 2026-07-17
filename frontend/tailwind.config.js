/** @type {import('tailwindcss').Config} */

// Every entry below resolves an RGB triplet CSS custom property (defined in
// src/styles/index.css under :root / :root.dark) rather than a flat hex
// value, so every existing class name (bg-surface, text-ink-muted, etc.)
// stays unchanged in every component but flips automatically when the
// `dark` class toggles on <html> (see src/theme/ThemeContext.tsx). The
// `rgb(var(...) / <alpha-value>)` form is required (not `rgb(var(...))`) so
// Tailwind's opacity modifiers (e.g. `border-danger/30`) keep working.
function withOpacity(variable) {
  return `rgb(var(${variable}) / <alpha-value>)`;
}

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: withOpacity("--color-surface"),
          muted: withOpacity("--color-surface-muted"),
          sunken: withOpacity("--color-surface-sunken"),
        },
        border: {
          DEFAULT: withOpacity("--color-border"),
          strong: withOpacity("--color-border-strong"),
        },
        ink: {
          DEFAULT: withOpacity("--color-ink"),
          muted: withOpacity("--color-ink-muted"),
          faint: withOpacity("--color-ink-faint"),
        },
        accent: {
          DEFAULT: withOpacity("--color-accent"),
          soft: withOpacity("--color-accent-soft"),
          strong: withOpacity("--color-accent-strong"),
          deep: withOpacity("--color-accent-deep"),
        },
        orange: {
          DEFAULT: withOpacity("--color-orange"),
          soft: withOpacity("--color-orange-soft"),
          strong: withOpacity("--color-orange-strong"),
        },
        pay: withOpacity("--color-pay"),
        reservoir: withOpacity("--color-reservoir"),
        nonreservoir: withOpacity("--color-nonreservoir"),
        danger: {
          DEFAULT: withOpacity("--color-danger"),
          soft: withOpacity("--color-danger-soft"),
        },
        success: {
          DEFAULT: withOpacity("--color-success"),
          soft: withOpacity("--color-success-soft"),
        },
      },
      backgroundImage: {
        "brand-gradient":
          "linear-gradient(135deg, #2563EB 0%, #1D4ED8 55%, #F97316 130%)",
        "brand-gradient-soft":
          "linear-gradient(135deg, rgb(var(--color-accent) / 0.08) 0%, rgb(var(--color-orange) / 0.08) 100%)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
      },
    },
  },
  plugins: [],
};
