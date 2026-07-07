import { useQuery } from "@tanstack/react-query";
import { getDashboardSummary } from "@/api/client";
import SummaryCards from "@/components/SummaryCards";
import WellsBarChart from "@/components/WellsBarChart";
import WellTable from "@/components/WellTable";
import UploadWells from "@/components/UploadWells";
import ChatPanel from "@/components/ChatPanel";

/** Multi-well dashboard (section 6): field-wide summary, comparison chart, wells table, upload, chat. */
export default function DashboardPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: getDashboardSummary,
  });

  return (
    <div className="pb-24">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-ink">Field Dashboard</h1>
        <p className="text-sm text-ink-faint">
          Cross-well petrophysical summary across all processed wells.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="lg:col-span-2">
          <UploadWells />
        </div>
      </div>

      {isLoading && <LoadingState />}
      {isError && <ErrorState message={(error as Error).message} />}

      {data && (
        <div className="space-y-6">
          <SummaryCards summary={data} />
          <WellsBarChart wells={data.wells} />
          <div>
            <h2 className="text-sm font-semibold text-ink mb-2">All Wells</h2>
            <WellTable wells={data.wells} />
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
        <div key={i} className="h-20 rounded-lg bg-surface-sunken animate-pulse" />
      ))}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-lg px-4 py-3">
      Failed to load dashboard: {message}
    </div>
  );
}
