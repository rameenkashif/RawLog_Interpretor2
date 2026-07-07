import type { WellZonesResponse } from "@/api/types";
import { zoneColors } from "@/styles/tokens";

function fmtPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

/** Per-zone thickness and average PHIE/SWE/VSH table (section 7). */
export default function ZoneSummaryTable({ zones }: { zones: WellZonesResponse }) {
  return (
    <div className="bg-surface border border-border rounded-lg shadow-sm overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-muted border-b border-border">
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Zone</th>
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Thickness (m)</th>
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Samples</th>
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Avg PHIE</th>
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Avg SWE</th>
            <th className="text-left px-4 py-2.5 font-medium text-ink-muted">Avg VSH</th>
          </tr>
        </thead>
        <tbody>
          {zones.zones.map((z) => (
            <tr key={z.zone_code} className="border-b border-border last:border-0">
              <td className="px-4 py-2.5 font-medium flex items-center gap-2">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: zoneColors[z.zone_code] }}
                />
                {z.zone_label}
              </td>
              <td className="px-4 py-2.5 text-ink-muted">{z.thickness.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{z.n_samples}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmtPct(z.avg_phie)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmtPct(z.avg_swe)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmtPct(z.avg_vsh)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
