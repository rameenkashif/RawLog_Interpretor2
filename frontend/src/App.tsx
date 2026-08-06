import { NavLink, Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import WellDetailPage from "./pages/WellDetailPage";
import SeismicPage from "./pages/SeismicPage";
import SyntheticSeismogramPage from "./pages/SyntheticSeismogramPage";
import ErrorBoundary from "./components/ErrorBoundary";
import ThemeToggle from "./components/ThemeToggle";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
    isActive ? "bg-accent text-white shadow-card" : "text-ink-muted hover:bg-surface-sunken"
  }`;

/**
 * App shell: slim top nav (brand, page nav, dark-mode toggle, vendor logos)
 * + routed pages. Colors flow entirely through CSS variables (see
 * src/styles/index.css / tailwind.config.js), so light/dark is a single
 * `dark` class toggle on <html> (src/theme/ThemeContext.tsx) -- no
 * per-page theme wiring needed here beyond the toggle button itself.
 * Brand accent: a fixed blue-to-orange gradient strip under the header,
 * echoed sparingly elsewhere (buttons, active states), kept identical in
 * both themes as the one constant visual signature.
 */
export default function App() {
  return (
    <div className="min-h-screen bg-surface text-ink">
      <header className="sticky top-0 z-30 bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/80 shadow-card">
        <div className="mx-auto max-w-[1600px] px-5 py-2.5 flex items-center justify-between gap-6">
          <div className="flex items-center gap-6 min-w-0">
            <div className="flex items-center gap-2 shrink-0">
              <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-brand-gradient text-white font-bold text-xs shadow-card">
                RC
              </span>
              <span className="font-bold text-base tracking-tight text-ink whitespace-nowrap">
                RawReservoir<span className="text-accent">Classifier</span>
              </span>
            </div>
            <nav className="flex gap-1">
              <NavLink to="/" end className={navLinkClass}>
                Dashboard
              </NavLink>
              <NavLink to="/seismic" className={navLinkClass}>
                Seismic
              </NavLink>
              <NavLink to="/synthetic" className={navLinkClass}>
                Synthetic Seismogram
              </NavLink>
            </nav>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <ThemeToggle />
            {/* Partner/vendor logos: GeoGraphix first, then LMKR. GeoGraphix's
                source PNG has an opaque white plate baked in (not just a
                transparent background) -- wrapped in a white chip so it
                reads as an intentional badge rather than a stray white box
                in dark mode; LMKR's artwork has no white fill and sits
                directly on the header in either theme. */}
            <div className="flex items-center gap-2 pl-3 border-l border-border">
              <span className="inline-flex items-center rounded-md bg-white px-1.5 py-1">
                <img
                  src="/logos/geographix-logo.png"
                  alt="GeoGraphix"
                  className="h-5 w-auto object-contain"
                />
              </span>
              <img
                src="/logos/lmkr-logo.png"
                alt="LMKR"
                className="h-8 w-auto object-contain"
              />
            </div>
          </div>
        </div>
        <div className="h-[3px] bg-brand-gradient" />
      </header>

      <main className="mx-auto max-w-[1600px] px-5 py-5">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/wells/:wellId" element={<WellDetailPage />} />
            <Route path="/seismic" element={<SeismicPage />} />
            <Route path="/synthetic" element={<SyntheticSeismogramPage />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}
