import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getWellCurves, getWellZones } from "@/api/client";
import LogTrackViewer from "@/components/LogTrackViewer";
import CrossplotBuilder from "@/components/CrossplotBuilder";
import ZoneSummaryTable from "@/components/ZoneSummaryTable";
import ExportButtons from "@/components/ExportButtons";
import ChatPanel from "@/components/ChatPanel";

/** Single-well view (section 7): log tracks, crossplots, zone table, export, chat. */
export default function WellDetailPage() {
  const { wellId } = useParams<{ wellId: string }>();

  const curvesQuery = useQuery({
    queryKey: ["well-curves", wellId],
    queryFn: () => getWellCurves(wellId!),
    enabled: Boolean(wellId),
  });

  const zonesQuery = useQuery({
    queryKey: ["well-zones", wellId],
    queryFn: () => getWellZones(wellId!),
    enabled: Boolean(wellId),
  });

  if (!wellId) return null;

  return (
    <div className="pb-24 space-y-6">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-6 py-5">
        <div className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative flex items-center justify-between flex-wrap gap-4">
          <div>
            <Link
              to="/"
              className="text-xs font-medium text-accent-strong hover:underline inline-flex items-center gap-1"
            >
              ← Back to dashboard
            </Link>
            <h1 className="text-2xl font-extrabold text-ink tracking-tight mt-1">
              {wellId}
            </h1>
            <p className="text-sm text-ink-muted">
              Single-well petrophysical interpretation
            </p>
          </div>
          <ExportButtons wellId={wellId} />
        </div>
      </div>

      {curvesQuery.isLoading && (
        <div className="h-[720px] rounded-xl bg-surface-sunken animate-pulse" />
      )}
      {curvesQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Failed to load curves: {(curvesQuery.error as Error).message}
        </div>
      )}
      {curvesQuery.data && (
        <section>
          <h2 className="text-sm font-semibold text-ink mb-2">Log Tracks</h2>
          <LogTrackViewer curves={curvesQuery.data} />
        </section>
      )}

      {zonesQuery.data && (
        <section>
          <h2 className="text-sm font-semibold text-ink mb-2">Zone Summary</h2>
          <ZoneSummaryTable zones={zonesQuery.data} />
        </section>
      )}

      {curvesQuery.data && (
        <section>
          <h2 className="text-sm font-semibold text-ink mb-2">Crossplots</h2>
          <CrossplotBuilder
            wellId={wellId}
            curveNames={curvesQuery.data.curve_names}
          />
        </section>
      )}

      <ChatPanel
        scope={wellId}
        wellId={wellId}
        title={`${wellId} Assistant`}
        subtitle="Ask about this well's curves and zonation"
      />
    </div>
  );
}
