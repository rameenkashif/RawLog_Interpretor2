import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { WellSummary } from "@/api/types";

type SortKey = keyof Pick<
  WellSummary,
  "well_id" | "start_depth" | "stop_depth" | "avg_phie" | "avg_swe" | "avg_vsh" | "net_pay_thickness"
>;

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: "well_id", label: "Well" },
  { key: "start_depth", label: "Start (m)" },
  { key: "stop_depth", label: "Stop (m)" },
  { key: "avg_vsh", label: "Avg VSH" },
  { key: "avg_phie", label: "Avg PHIE" },
  { key: "avg_swe", label: "Avg SWE" },
  { key: "net_pay_thickness", label: "Net Pay (m)" },
];

function fmt(v: number | null, pct = false): string {
  if (v === null) return "—";
  return pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(1);
}

/** Sortable table of all wells (section 6). Clicking a row navigates to the single-well view. */
export default function WellTable({ wells }: { wells: WellSummary[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("well_id");
  const [asc, setAsc] = useState(true);
  const navigate = useNavigate();

  const sorted = useMemo(() => {
    const copy = [...wells];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null) return 1;
      if (bv === null) return -1;
      if (typeof av === "string" || typeof bv === "string") {
        return String(av).localeCompare(String(bv)) * (asc ? 1 : -1);
      }
      return ((av as number) - (bv as number)) * (asc ? 1 : -1);
    });
    return copy;
  }, [wells, sortKey, asc]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) setAsc((a) => !a);
    else {
      setSortKey(key);
      setAsc(true);
    }
  }

  return (
    <div className="bg-surface border border-border rounded-lg shadow-sm overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-muted border-b border-border">
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                onClick={() => toggleSort(col.key)}
                className="text-left px-4 py-2.5 font-medium text-ink-muted cursor-pointer select-none hover:text-ink"
              >
                {col.label}
                {sortKey === col.key && <span className="ml-1 text-accent">{asc ? "↑" : "↓"}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((w) => (
            <tr
              key={w.well_id}
              onClick={() => navigate(`/wells/${w.well_id}`)}
              className="border-b border-border last:border-0 hover:bg-accent-soft cursor-pointer transition-colors"
            >
              <td className="px-4 py-2.5 font-medium text-accent-strong">{w.well_id}</td>
              <td className="px-4 py-2.5 text-ink-muted">{w.start_depth.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{w.stop_depth.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmt(w.avg_vsh, true)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmt(w.avg_phie, true)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmt(w.avg_swe, true)}</td>
              <td className="px-4 py-2.5 text-ink-muted">{fmt(w.net_pay_thickness)}</td>
            </tr>
          ))}
          {sorted.length === 0 && (
            <tr>
              <td colSpan={COLUMNS.length} className="px-4 py-8 text-center text-ink-faint">
                No wells uploaded yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
