import { NavLink, Route, Routes } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import WellDetailPage from "./pages/WellDetailPage";

/**
 * App shell: light-themed top nav + routed pages. Every page below renders
 * on the shared white/near-white background -- there is no dark variant.
 */
export default function App() {
  return (
    <div className="min-h-screen bg-surface text-ink">
      <header className="border-b border-border bg-surface sticky top-0 z-30">
        <div className="mx-auto max-w-[2000px] px-6 py-3 flex items-center justify-between gap-8">
          <div className="flex items-center gap-8">
            <span className="font-semibold text-lg tracking-tight">
              RawReservoir<span className="text-accent">Classifier</span>
            </span>
            <nav className="flex gap-1">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-accent-soft text-accent-strong"
                      : "text-ink-muted hover:bg-surface-sunken"
                  }`
                }
              >
                Dashboard
              </NavLink>
            </nav>
          </div>

          {/* Partner/vendor logos, top-right, horizontal: GeoGraphix first, then LMKR. */}
          <div className="flex items-center gap-2 shrink-0">
            <img
  src="/logos/geographix-logo.png"
  alt="GeoGraphix"
  className="h-28 w-auto object-contain"
/>
<img
  src="/logos/lmkr-logo.png"
  alt="LMKR"
  className="h-20 w-auto object-contain"
/>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-6 py-6">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/wells/:wellId" element={<WellDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}