import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { WellSummary } from "@/api/types";

type SortKey = keyof Pick<
  WellSummary,
  | "well_id"
  | "start_depth"
  | "stop_depth"
  | "avg_phie"
  | "avg_swe"
  | "avg_vsh"
  | "net_pay_thickness"
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

/** Small color-coded quality badge based on SWE (lower Sw = better hydrocarbon saturation). */
function SweBadge({ swe }: { swe: number | null }) {
  if (swe === null) return <span className="text-ink-faint">—</span>;
  const pct = swe * 100;
  const style =
    swe < 0.4
      ? "bg-emerald-50 text-emerald-700"
      : swe < 0.65
        ? "bg-orange-soft text-orange-strong"
        : "bg-slate-100 text-ink-muted";
  return (
    <span
      className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${style}`}
    >
      {pct.toFixed(1)}%
    </span>
  );
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
    <div className="bg-surface border border-border rounded-xl shadow-card overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-muted border-b border-border">
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                onClick={() => toggleSort(col.key)}
                className="text-left px-4 py-3 font-semibold text-ink-muted cursor-pointer select-none hover:text-accent transition-colors"
              >
                {col.label}
                {sortKey === col.key && (
                  <span className="ml-1 text-accent">{asc ? "↑" : "↓"}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((w) => (
            <tr
              key={w.well_id}
              onClick={() => navigate(`/wells/${w.well_id}`)}
              className="group border-b border-border last:border-0 hover:bg-accent-soft/60 cursor-pointer transition-colors"
            >
              <td className="px-4 py-3 font-semibold text-accent-strong">
                <span className="inline-flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-accent group-hover:bg-orange transition-colors" />
                  {w.well_id}
                </span>
              </td>
              <td className="px-4 py-3 text-ink-muted">
                {w.start_depth.toFixed(1)}
              </td>
              <td className="px-4 py-3 text-ink-muted">
                {w.stop_depth.toFixed(1)}
              </td>
              <td className="px-4 py-3 text-ink-muted">
                {fmt(w.avg_vsh, true)}
              </td>
              <td className="px-4 py-3 text-ink-muted">
                {fmt(w.avg_phie, true)}
              </td>
              <td className="px-4 py-3">
                <SweBadge swe={w.avg_swe} />
              </td>
              <td className="px-4 py-3 text-ink-muted font-medium">
                {fmt(w.net_pay_thickness)}
              </td>
            </tr>
          ))}
          {sorted.length === 0 && (
            <tr>
              <td
                colSpan={COLUMNS.length}
                className="px-4 py-10 text-center text-ink-faint"
              >
                No wells uploaded yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
