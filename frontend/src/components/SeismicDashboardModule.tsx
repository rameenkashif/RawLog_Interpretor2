import { Link } from "react-router-dom";
import type { SeismicSummary } from "@/api/types";

function fmtPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

/**
 * Compact "Seismic Data" module for the dashboard (section: at-a-glance
 * overview). Shows dataset count and average proxy values already returned
 * by GET /dashboard/summary, with a link through to the full Seismic page
 * for the raw section display and detailed attribute plots.
 */
export default function SeismicDashboardModule({
  nDatasets,
  datasets,
}: {
  nDatasets: number;
  datasets: SeismicSummary[];
}) {
  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-ink">Seismic Data</h3>
        <Link
          to="/seismic"
          className="text-xs font-semibold text-accent-strong hover:underline"
        >
          Open module →
        </Link>
      </div>

      {nDatasets === 0 ? (
        <p className="text-xs text-ink-faint">
          No SEG-Y datasets uploaded yet. Visit the Seismic module to add one.
        </p>
      ) : (
        <div className="space-y-3">
          <p className="text-xs text-ink-muted">
            <span className="font-semibold text-ink">{nDatasets}</span> dataset
            {nDatasets !== 1 ? "s" : ""} processed
          </p>
          <div className="grid grid-cols-3 gap-3">
            {(() => {
              const avg = (key: keyof SeismicSummary) => {
                const values = datasets
                  .map((d) => d[key] as number | null)
                  .filter((v): v is number => v !== null);
                return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
              };
              return (
                <>
                  <MiniStat label="Avg VSH Proxy" value={fmtPct(avg("avg_vsh_proxy"))} />
                  <MiniStat label="Avg PHIE Proxy" value={fmtPct(avg("avg_phie_proxy"))} />
                  <MiniStat label="Avg SWE Proxy" value={fmtPct(avg("avg_swe_proxy"))} />
                </>
              );
            })()}
          </div>
          <p className="text-[11px] text-orange-strong bg-orange-soft rounded-md px-2 py-1.5 leading-relaxed">
            Seismic proxies are uncalibrated amplitude heuristics -- not measured shale volume,
            porosity, or water saturation. See the Seismic module for details.
          </p>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center bg-surface-muted rounded-lg py-2">
      <p className="text-sm font-bold text-ink">{value}</p>
      <p className="text-[10px] text-ink-faint uppercase tracking-wide mt-0.5">{label}</p>
    </div>
  );
}
