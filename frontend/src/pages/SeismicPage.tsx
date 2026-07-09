import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getSeismicAttributes, getSeismicExportUrl, getSeismicSection, listSeismic } from "@/api/client";
import SeismicUpload from "@/components/SeismicUpload";
import SeismicSection from "@/components/SeismicSection";
import SeismicAttributesChart from "@/components/SeismicAttributesChart";
import WellSeismicTie from "@/components/WellSeismicTie";
import SeismicPanel from "@/components/Seismic/SeismicPanel";

function fmtPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

/**
 * Seismic module (SEG-Y): upload, dataset picker, raw amplitude section
 * display, and computed attribute trends including the heuristic
 * VSH/PHIE/SWE seismic proxies. Linked from the dashboard's "Seismic Data"
 * summary card.
 */
export default function SeismicPage() {
  const datasetsQuery = useQuery({ queryKey: ["seismic-datasets"], queryFn: listSeismic });
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedId && datasetsQuery.data && datasetsQuery.data.length > 0) {
      setSelectedId(datasetsQuery.data[0].dataset_id);
    }
  }, [datasetsQuery.data, selectedId]);

  const sectionQuery = useQuery({
    queryKey: ["seismic-section", selectedId],
    queryFn: () => getSeismicSection(selectedId!),
    enabled: Boolean(selectedId),
  });

  const attributesQuery = useQuery({
    queryKey: ["seismic-attributes", selectedId],
    queryFn: () => getSeismicAttributes(selectedId!),
    enabled: Boolean(selectedId),
  });

  const selectedSummary = datasetsQuery.data?.find((d) => d.dataset_id === selectedId);

  return (
    <div className="pb-24 space-y-6">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-6 py-6">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative flex items-center justify-between flex-wrap gap-4">
          <div>
            <Link to="/" className="text-xs font-medium text-accent-strong hover:underline">
              ← Back to dashboard
            </Link>
            <p className="text-xs font-semibold uppercase tracking-wider text-accent-strong mb-1 mt-1">
              Seismic Module
            </p>
            <h1 className="text-2xl font-extrabold text-ink tracking-tight">Seismic Data</h1>
            <p className="text-sm text-ink-muted mt-1 max-w-xl">
              Raw SEG-Y amplitude sections and derived attributes, including uncalibrated
              VSH/PHIE/SWE seismic proxies for lateral trend screening away from well control.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 items-start">
        <div className="xl:col-span-2">
          <h2 className="text-sm font-semibold text-ink mb-2">Datasets</h2>
          {datasetsQuery.isLoading && (
            <div className="h-16 rounded-xl bg-surface-sunken animate-pulse" />
          )}
          {datasetsQuery.data && datasetsQuery.data.length === 0 && (
            <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
              No seismic datasets uploaded yet. Upload a SEG-Y file to get started.
            </div>
          )}
          {datasetsQuery.data && datasetsQuery.data.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {datasetsQuery.data.map((d) => (
                <button
                  key={d.dataset_id}
                  onClick={() => setSelectedId(d.dataset_id)}
                  className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                    selectedId === d.dataset_id
                      ? "bg-brand-gradient text-white border-transparent shadow-card"
                      : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
                  }`}
                >
                  {d.dataset_id}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="xl:col-span-1">
          <SeismicUpload />
        </div>
      </div>

      {selectedSummary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <SummaryTile label="Traces" value={selectedSummary.n_traces.toLocaleString()} />
          <SummaryTile
            label="Duration"
            value={`${selectedSummary.duration_ms.toFixed(0)} ms`}
          />
          <SummaryTile label="Avg VSH Proxy" value={fmtPct(selectedSummary.avg_vsh_proxy)} />
          <SummaryTile label="Avg PHIE Proxy" value={fmtPct(selectedSummary.avg_phie_proxy)} />
          <SummaryTile label="Avg SWE Proxy" value={fmtPct(selectedSummary.avg_swe_proxy)} />
        </div>
      )}

      {selectedId && (
        <div className="flex justify-end">
          
            <a
              href={getSeismicExportUrl(selectedId)}
            className="text-xs font-semibold px-3.5 py-1.5 rounded-full border border-accent/30 bg-accent-soft text-accent-strong hover:bg-accent hover:text-white transition-colors"
          >
            Export Attributes CSV
          </a>
        </div>
      )}

      {sectionQuery.isLoading && (
        <div className="h-[520px] rounded-xl bg-surface-sunken animate-pulse" />
      )}
      {sectionQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Failed to load seismic section: {(sectionQuery.error as Error).message}
        </div>
      )}
      {sectionQuery.data && (
        <section>
          <h2 className="text-sm font-semibold text-ink mb-2">Raw Amplitude Section</h2>
          <SeismicSection section={sectionQuery.data} />
        </section>
      )}

      {attributesQuery.data && (
        <section>
          <h2 className="text-sm font-semibold text-ink mb-2">Computed Attributes</h2>
          <SeismicAttributesChart attributes={attributesQuery.data} />
        </section>
      )}

      <section>
        <h2 className="text-sm font-semibold text-ink mb-2">Well-to-Seismic Tie</h2>
        <WellSeismicTie />
      </section>

      <section className="border-t border-border pt-6">
        <SeismicPanel />
      </section>
    </div>
  );
}

function SummaryTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface border border-border rounded-xl px-4 py-3 shadow-card">
      <p className="text-xs font-semibold text-ink-faint uppercase tracking-wide">{label}</p>
      <p className="text-xl font-extrabold text-ink mt-1">{value}</p>
    </div>
  );
}