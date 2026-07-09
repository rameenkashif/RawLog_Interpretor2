import type { ReactNode } from "react";
import type { SyntheticSeismogramResponse } from "@/api/types";

function Badge({ tone, children }: { tone: "orange" | "accent" | "green"; children: ReactNode }) {
  const toneClasses = {
    orange: "border-orange/30 bg-orange-soft text-orange-strong",
    accent: "border-accent/30 bg-accent-soft text-accent-strong",
    green: "border-green-200 bg-green-50 text-green-700",
  }[tone];
  return (
    <span className={`text-xs font-semibold px-3 py-1.5 rounded-full border ${toneClasses}`}>{children}</span>
  );
}

/**
 * Prominent QC/caveat badges -- a geophysicist needs to see these before
 * trusting the tie, not have them buried in logs. Covers: vertical-well
 * assumption (no deviation survey), no-checkshot time-depth caveat,
 * coordinate unit standardization status, and washout interval count.
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
    </div>
  );
}
