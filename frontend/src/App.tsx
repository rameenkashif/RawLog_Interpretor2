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
        <div className="mx-auto max-w-[1600px] px-6 py-3 flex items-center gap-8">
          <span className="font-semibold text-lg tracking-tight">
            Petro<span className="text-accent">Interp</span>
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
