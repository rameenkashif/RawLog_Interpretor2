import { useTheme } from "@/theme/ThemeContext";

/** Sun/moon icon toggle, lives in the shared app header (App.tsx) so it's
 * visible on every page. */
export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-border text-ink-muted hover:text-accent hover:border-accent transition-colors"
    >
      {isDark ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="h-[18px] w-[18px]">
          <circle cx="12" cy="12" r="4.2" strokeLinecap="round" strokeLinejoin="round" />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 2.5v2M12 19.5v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2.5 12h2M19.5 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} className="h-[18px] w-[18px]">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"
          />
        </svg>
      )}
    </button>
  );
}
