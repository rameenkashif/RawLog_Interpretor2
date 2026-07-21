import { useQuery } from "@tanstack/react-query";
import { getDashboardSummary } from "@/api/client";
import SummaryCards from "@/components/SummaryCards";
import WellsBarChart from "@/components/WellsBarChart";
import NetPayChart from "@/components/NetPayChart";
import WellTable from "@/components/WellTable";
import UploadWells from "@/components/UploadWells";
import DashboardUpload from "@/components/DashboardUpload";
import SeismicDashboardModule from "@/components/SeismicDashboardModule";
import ChatPanel from "@/components/ChatPanel";

/** Multi-well dashboard (section 6): field-wide summary, comparison charts, wells table, upload, chat. */
export default function DashboardPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: getDashboardSummary,
  });

  return (
    <div className="pb-12 space-y-4">
      {/* Hero banner -- gives an immediate field-level orientation before the detail below. */}
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-5 py-4">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-accent/10 blur-2xl" />
        <div className="absolute right-24 bottom-0 h-24 w-24 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative flex items-center justify-between flex-wrap gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-accent-strong mb-1">
              Field Overview
            </p>
            <h1 className="text-xl font-extrabold text-ink tracking-tight">
              Field Dashboard
            </h1>
            <p className="text-sm text-ink-muted mt-1 max-w-xl">
              Cross-well petrophysical summary across all processed wells --
              VSH, PHIE, SWE, and net pay computed from raw LAS logs.
            </p>
          </div>
          {data && (
            <div className="flex gap-6 text-right">
              <div>
                <p className="text-3xl font-extrabold text-accent-strong">
                  {data.n_wells}
                </p>
                <p className="text-xs text-ink-faint uppercase tracking-wide">
                  Wells Online
                </p>
              </div>
              <div>
                <p className="text-3xl font-extrabold text-orange-strong">
                  {data.total_footage.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  m
                </p>
                <p className="text-xs text-ink-faint uppercase tracking-wide">
                  Total Footage
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {isLoading && <LoadingState />}
      {isError && <ErrorState message={(error as Error).message} />}

      {data && (
        <div className="space-y-4">
          <SummaryCards summary={data} />

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            <WellsBarChart wells={data.wells} />
            <NetPayChart wells={data.wells} />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 items-start">
            <div className="xl:col-span-2">
              <h2 className="text-sm font-semibold text-ink mb-1.5">All Wells</h2>
              <WellTable wells={data.wells} />
            </div>
            <div className="xl:col-span-1 space-y-3">
              <div>
                <h2 className="text-sm font-semibold text-ink mb-1.5">
                  Add Data
                </h2>
                <div className="space-y-3">
                  <DashboardUpload />
                  <UploadWells />
                </div>
              </div>
              <SeismicDashboardModule
                nDatasets={data.n_seismic_datasets}
                datasets={data.seismic_datasets}
              />
            </div>
          </div>
        </div>
      )}

      <ChatPanel
        scope="dashboard"
        wellId={null}
        title="Field Assistant"
        subtitle="Ask cross-well petrophysics questions"
      />
    </div>
  );
}

function LoadingState() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="h-24 rounded-xl bg-surface-sunken animate-pulse"
        />
      ))}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
      Failed to load dashboard: {message}
    </div>
  );
}
