import type { DashboardSummary } from "@/api/types";

function formatPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

function formatNum(v: number, digits = 0): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** Field-wide summary cards (section 6): well count, footage, avg VSH/PHIE/SWE. */
export default function SummaryCards({ summary }: { summary: DashboardSummary }) {
  const cards = [
    { label: "Wells", value: formatNum(summary.n_wells) },
    { label: "Total Footage Logged", value: `${formatNum(summary.total_footage)} m` },
    { label: "Avg VSH", value: formatPct(summary.avg_vsh) },
    { label: "Avg PHIE", value: formatPct(summary.avg_phie) },
    { label: "Avg SWE", value: formatPct(summary.avg_swe) },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className="bg-surface border border-border rounded-lg px-4 py-3 shadow-sm"
        >
          <p className="text-xs font-medium text-ink-faint uppercase tracking-wide">{c.label}</p>
          <p className="text-2xl font-semibold text-ink mt-1">{c.value}</p>
        </div>
      ))}
    </div>
  );
}
