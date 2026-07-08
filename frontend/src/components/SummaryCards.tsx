import type { DashboardSummary } from "@/api/types";

function formatPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

function formatNum(v: number, digits = 0): string {
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

const ICONS: Record<string, JSX.Element> = {
  wells: (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 3v18M7 7l5-4 5 4M5 12h14M5 17h14"
    />
  ),
  footage: (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3 17l6-10 4 6 3-4 5 8M3 20h18"
    />
  ),
  vsh: (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M4 6h16M4 12h16M4 18h10"
    />
  ),
  phie: (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 3c3 4 6 7.5 6 11a6 6 0 1 1-12 0c0-3.5 3-7 6-11z"
    />
  ),
  swe: (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 4c2.5 3 5 6.5 5 9.5A5 5 0 0 1 7 13.5C7 10.5 9.5 7 12 4z"
    />
  ),
};

interface CardDef {
  key: keyof typeof ICONS;
  label: string;
  value: string;
  accent: "accent" | "orange";
}

/** Field-wide summary cards (section 6): well count, footage, avg VSH/PHIE/SWE. */
export default function SummaryCards({
  summary,
}: {
  summary: DashboardSummary;
}) {
  const cards: CardDef[] = [
    {
      key: "wells",
      label: "Wells",
      value: formatNum(summary.n_wells),
      accent: "accent",
    },
    {
      key: "footage",
      label: "Total Footage Logged",
      value: `${formatNum(summary.total_footage)} m`,
      accent: "orange",
    },
    {
      key: "vsh",
      label: "Avg VSH",
      value: formatPct(summary.avg_vsh),
      accent: "accent",
    },
    {
      key: "phie",
      label: "Avg PHIE",
      value: formatPct(summary.avg_phie),
      accent: "orange",
    },
    {
      key: "swe",
      label: "Avg SWE",
      value: formatPct(summary.avg_swe),
      accent: "accent",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className="group relative overflow-hidden bg-surface border border-border rounded-xl px-4 py-4 shadow-card hover:shadow-card-hover transition-shadow"
        >
          <div
            className={`absolute inset-x-0 top-0 h-1 ${
              c.accent === "accent" ? "bg-accent" : "bg-orange"
            }`}
          />
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-ink-faint uppercase tracking-wide">
              {c.label}
            </p>
            <span
              className={`inline-flex h-8 w-8 items-center justify-center rounded-lg ${
                c.accent === "accent"
                  ? "bg-accent-soft text-accent"
                  : "bg-orange-soft text-orange"
              }`}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                className="h-[18px] w-[18px]"
              >
                {ICONS[c.key]}
              </svg>
            </span>
          </div>
          <p className="text-2xl font-extrabold text-ink tracking-tight">
            {c.value}
          </p>
        </div>
      ))}
    </div>
  );
}
