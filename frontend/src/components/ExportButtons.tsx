import { getExportUrl } from "@/api/client";

/**
 * Export buttons (section 7): CSV/LAS download of interpreted curves.
 * PNG export of any chart is handled by Plotly's built-in camera icon in
 * the chart's mode bar (config.displaylogo=false leaves that control
 * visible) -- no extra plumbing needed for that part.
 */
export default function ExportButtons({ wellId }: { wellId: string }) {
  return (
    <div className="flex gap-2">
      <a
        href={getExportUrl(wellId, "csv")}
        className="text-xs font-medium px-3 py-1.5 rounded-md border border-border-strong bg-surface text-ink-muted hover:bg-surface-sunken transition-colors"
      >
        Export CSV
      </a>
      <a
        href={getExportUrl(wellId, "las")}
        className="text-xs font-medium px-3 py-1.5 rounded-md border border-border-strong bg-surface text-ink-muted hover:bg-surface-sunken transition-colors"
      >
        Export LAS
      </a>
    </div>
  );
}
