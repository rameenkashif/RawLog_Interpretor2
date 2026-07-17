import type { ReactNode } from "react";
import type { SyntheticSeismogramResponse } from "@/api/types";

function Badge({ tone, children }: { tone: "orange" | "accent" | "green" | "danger"; children: ReactNode }) {
  const toneClasses = {
    orange: "border-orange/30 bg-orange-soft text-orange-strong",
    accent: "border-accent/30 bg-accent-soft text-accent-strong",
    green: "border-success/30 bg-success-soft text-success",
    danger: "border-danger/40 bg-danger-soft text-danger",
  }[tone];
  return (
    <span className={`text-xs font-semibold px-3 py-1.5 rounded-full border ${toneClasses}`}>{children}</span>
  );
}

/**
 * Prominent QC/caveat badges -- a geophysicist needs to see these before
 * trusting the tie, not have them buried in logs. Covers: vertical-well
 * assumption (no deviation survey), no-checkshot time-depth caveat,
 * coordinate unit standardization status, washout interval count, the
 * time-depth datum plausibility check, and the bulk-shift
 * boundary-pinned reliability flag -- a correlation number alone can't
 * distinguish a genuine tie from a spurious match against noise pinned
 * to the edge of the search window.
 */
export default function QcBadges({ result }: { result: SyntheticSeismogramResponse }) {
  const washoutCount = result.washout_flag.filter(Boolean).length;
  const unit = result.well_header.coordinate_unit_detected;

  return (
    <div className="flex flex-wrap gap-2">
      <Badge tone="orange">Vertical assumption — no deviation survey</Badge>
      <Badge tone="orange">No checkshot — sonic-integration time-depth only</Badge>
      {unit === "feet" && result.well_header.unit_conversion_applied && (
        <Badge tone="accent">
          Coordinates converted ft→m (TD/STOP ratio {result.well_header.td_stop_ratio?.toFixed(2)})
        </Badge>
      )}
      {unit === "meters" && <Badge tone="green">Coordinates already in meters</Badge>}
      {unit === null && <Badge tone="orange">Coordinate unit unvalidated (no TD/STOP to check)</Badge>}
      <Badge tone={washoutCount > 0 ? "orange" : "green"}>
        {washoutCount > 0
          ? `${washoutCount} depth sample(s) flagged: possible washout / unreliable interval`
          : "No washout intervals flagged"}
      </Badge>
      <Badge tone={result.datum_check.plausible ? "green" : "danger"}>
        {result.datum_check.plausible
          ? `Datum check OK (delay-implied depth ${result.datum_check.implied_depth_m.toFixed(0)}m vs. logged top ${result.datum_check.logged_top_depth_m.toFixed(0)}m)`
          : `Datum check FAILED — delay-implied depth ${result.datum_check.implied_depth_m.toFixed(0)}m is far from logged top ${result.datum_check.logged_top_depth_m.toFixed(0)}m; time-depth anchor may not be meaningful`}
      </Badge>
      <Badge tone={result.boundary_pinned ? "danger" : "green"}>
        {result.boundary_pinned
          ? `Shift pinned to search edge (±${result.max_shift_ms.toFixed(0)}ms) — likely a spurious match, not a genuine tie`
          : `Bulk shift converged within ±${result.max_shift_ms.toFixed(0)}ms search range`}
      </Badge>
    </div>
  );
}
