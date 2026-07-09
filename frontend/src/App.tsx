import { NavLink, Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import WellDetailPage from "./pages/WellDetailPage";
import SeismicPage from "./pages/SeismicPage";
import SyntheticSeismogramPage from "./pages/SyntheticSeismogramPage";
import ErrorBoundary from "./components/ErrorBoundary";

/**
 * App shell: light-themed top nav + routed pages. Every page below renders
 * on the shared white/near-white background -- there is no dark variant.
 * Brand accent: a subtle blue-to-orange gradient strip under the header,
 * echoed sparingly elsewhere (buttons, active states) to keep the palette
 * cohesive without ever going dark.
 */
export default function App() {
  return (
    <div className="min-h-screen bg-surface text-ink">
      <header className="sticky top-0 z-30 bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/80 shadow-card">
        <div className="mx-auto max-w-[1600px] px-6 py-3.5 flex items-center justify-between gap-8">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2.5">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-brand-gradient text-white font-bold text-sm shadow-card">
                RC
              </span>
              <span className="font-bold text-lg tracking-tight text-ink">
                RawReservoir<span className="text-accent">Classifier</span>
              </span>
            </div>
            <nav className="flex gap-1">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-sm font-medium transition-all ${
                    isActive
                      ? "bg-accent text-white shadow-card"
                      : "text-ink-muted hover:bg-surface-sunken"
                  }`
                }
              >
                Dashboard
              </NavLink>
              <NavLink
                to="/seismic"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-sm font-medium transition-all ${
                    isActive
                      ? "bg-accent text-white shadow-card"
                      : "text-ink-muted hover:bg-surface-sunken"
                  }`
                }
              >
                Seismic
              </NavLink>
              <NavLink
                to="/synthetic"
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-sm font-medium transition-all ${
                    isActive
                      ? "bg-accent text-white shadow-card"
                      : "text-ink-muted hover:bg-surface-sunken"
                  }`
                }
              >
                Synthetic Seismogram
              </NavLink>
            </nav>
          </div>

          {/* Partner/vendor logos, top-right, horizontal: GeoGraphix first, then LMKR. */}
          <div className="flex items-center gap-2 shrink-0 pl-4 border-l border-border">
            <img
              src="/logos/geographix-logo.png"
              alt="GeoGraphix"
              className="h-20 w-auto object-contain"
            />
            <img
              src="/logos/lmkr-logo.png"
              alt="LMKR"
              className="h-24 w-auto object-contain"
            />
          </div>
        </div>
        <div className="h-[3px] bg-brand-gradient" />
      </header>

      <main className="mx-auto max-w-[1600px] px-6 py-6">
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
