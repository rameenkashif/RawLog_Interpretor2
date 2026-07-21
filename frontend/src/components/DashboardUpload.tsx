import { useRef, useState, type RefObject } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { getDashboardUploadStatus, uploadDashboard } from "@/api/client";
import type { DashboardUploadStatusResponse } from "@/api/types";
import { Badge } from "@/components/Synthetic/QcBadges";
import { useAppStore } from "@/store/useAppStore";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

// Poll every 1.5s while processing, give up (not forever) if it's taking
// unreasonably long -- a stuck background task shouldn't poll silently
// forever in the browser.
const POLL_INTERVAL_MS = 1500;
const MAX_POLL_MS = 5 * 60 * 1000;

/**
 * Dashboard-level combined upload: a well (LAS) + its corresponding
 * seismic data (SEG-Y) together, auto-processed in the background (tie +
 * synthetic seismogram + spectral summary -- see backend/app/services/
 * dashboard_upload_service.py). On completion, sets the shared "active
 * well/dataset" (useAppStore) and invalidates the queries every page reads
 * from, so Wells/Dashboard, Seismic, and Synthetic Seismogram all reflect
 * the new data without manual re-selection.
 *
 * Additive alongside UploadWells/SeismicUpload -- those remain for
 * standalone single-file-type uploads; this is for the common "I have a
 * new well and its seismic" case.
 */
export default function DashboardUpload() {
  const lasInputRef = useRef<HTMLInputElement>(null);
  const segyInputRef = useRef<HTMLInputElement>(null);
  const [lasFile, setLasFile] = useState<File | null>(null);
  const [segyFile, setSegyFile] = useState<File | null>(null);
  const [pollingWellId, setPollingWellId] = useState<string | null>(null);
  const [pollStartedAt, setPollStartedAt] = useState<number | null>(null);
  const [settled, setSettled] = useState<DashboardUploadStatusResponse | null>(null);

  const queryClient = useQueryClient();
  const setActiveWell = useAppStore((s) => s.setActiveWell);

  const uploadMutation = useMutation({
    mutationFn: () => uploadDashboard(lasFile!, segyFile!),
    onSuccess: (data) => {
      setActiveWell(data.well_id, null);
      setSettled(null);
      setPollingWellId(data.well_id);
      setPollStartedAt(Date.now());
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
      queryClient.invalidateQueries({ queryKey: ["wells"] });
    },
  });

  const timedOut = pollStartedAt !== null && Date.now() - pollStartedAt > MAX_POLL_MS;

  const statusQuery = useQuery({
    queryKey: ["dashboard-upload-status", pollingWellId],
    queryFn: () => getDashboardUploadStatus(pollingWellId!),
    enabled: Boolean(pollingWellId) && !timedOut,
    refetchInterval: (query) => {
      const data = query.state.data as DashboardUploadStatusResponse | undefined;
      return data && data.status !== "processing" ? false : POLL_INTERVAL_MS;
    },
  });

  const status = statusQuery.data;
  if (status && status.status !== "processing" && settled?.updated_at !== status.updated_at) {
    // Reached a terminal state for the first time -- sync the shared
    // active well/dataset and let every page's data refetch.
    setSettled(status);
    setActiveWell(status.well_id, status.dataset_id ?? null);
    queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    queryClient.invalidateQueries({ queryKey: ["wells"] });
    queryClient.invalidateQueries({ queryKey: ["seismic-datasets"] });
  }

  function handleSubmit() {
    if (!lasFile || !segyFile) return;
    uploadMutation.mutate();
  }

  function reset() {
    setLasFile(null);
    setSegyFile(null);
    setPollingWellId(null);
    setPollStartedAt(null);
    setSettled(null);
    uploadMutation.reset();
    if (lasInputRef.current) lasInputRef.current.value = "";
    if (segyInputRef.current) segyInputRef.current.value = "";
  }

  const isBusy = uploadMutation.isPending || (Boolean(pollingWellId) && status?.status === "processing");

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-ink">Upload Well + Seismic</h3>
        <span className="inline-flex items-center gap-1 text-xs text-accent-strong bg-accent-soft px-2 py-0.5 rounded-full font-medium">
          combined
        </span>
      </div>
      <p className="text-xs text-ink-faint mb-3">
        Upload a well and its seismic together -- the tie, synthetic seismogram, and spectral
        summary are computed automatically in the background, and Wells, Seismic, and Synthetic
        Seismogram all switch to the new data once it's ready.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
        <FilePicker
          inputRef={lasInputRef}
          label="Well log"
          accept=".las"
          file={lasFile}
          onChange={setLasFile}
          disabled={isBusy}
        />
        <FilePicker
          inputRef={segyInputRef}
          label="Seismic volume"
          accept=".sgy,.segy"
          file={segyFile}
          onChange={setSegyFile}
          disabled={isBusy}
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={handleSubmit}
          disabled={!lasFile || !segyFile || isBusy}
          className="text-xs font-semibold px-3.5 py-1.5 rounded-full bg-brand-gradient text-white disabled:opacity-40 shadow-card transition-transform hover:scale-[1.02]"
        >
          Upload &amp; Process
        </button>
        {(uploadMutation.isSuccess || uploadMutation.isError) && (
          <button
            onClick={reset}
            className="text-xs font-semibold px-3 py-1.5 rounded-full border border-border-strong text-ink-muted hover:border-accent hover:text-accent"
          >
            Reset
          </button>
        )}
      </div>

      {uploadMutation.isError && (
        <p className="text-xs text-danger mt-2">Upload failed: {errorMessage(uploadMutation.error)}</p>
      )}

      {pollingWellId && (
        <div className="mt-3">
          <StatusBanner wellId={pollingWellId} status={status} timedOut={timedOut} />
        </div>
      )}
    </div>
  );
}

function FilePicker({
  inputRef,
  label,
  accept,
  file,
  onChange,
  disabled,
}: {
  inputRef: RefObject<HTMLInputElement>;
  label: string;
  accept: string;
  file: File | null;
  onChange: (f: File | null) => void;
  disabled: boolean;
}) {
  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      className={`cursor-pointer border-2 border-dashed rounded-lg px-3 py-4 text-center transition-all ${
        disabled
          ? "opacity-50 cursor-not-allowed border-border-strong"
          : "border-border-strong bg-surface-muted hover:border-accent hover:bg-accent-soft/50"
      }`}
    >
      <p className="text-xs font-semibold text-ink-muted">{label}</p>
      <p className="text-xs text-ink-faint mt-0.5 truncate">
        {file ? file.name : `Click to choose ${accept}`}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        disabled={disabled}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

function StatusBanner({
  wellId,
  status,
  timedOut,
}: {
  wellId: string;
  status: DashboardUploadStatusResponse | undefined;
  timedOut: boolean;
}) {
  if (timedOut) {
    return (
      <Badge tone="orange">
        {wellId} is taking longer than expected -- check back later or refresh.
      </Badge>
    );
  }
  if (!status || status.status === "processing") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-accent font-medium">
        <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
        Processing {wellId}: seismic upload, tie, synthetic seismogram, spectral summary…
      </span>
    );
  }
  if (status.status === "failed") {
    return <Badge tone="danger">{wellId} failed: {status.error ?? "unknown error"}</Badge>;
  }

  // status === "ready" -- the pipeline ran, but individual sub-results may
  // still be flagged low-confidence/unavailable; never collapse that into
  // a single "success" badge.
  const lowConfidence = status.tie_low_confidence || status.synthetic_low_confidence;
  return (
    <div className="flex flex-wrap gap-2">
      <Badge tone={lowConfidence ? "danger" : "green"}>
        {wellId} ready
        {status.tie_available && ` — tie correlation ${status.tie_correlation?.toFixed(3)}`}
        {status.tie_low_confidence && " (low confidence)"}
      </Badge>
      {!status.tie_available && (
        <Badge tone="orange">No tie: {status.tie_error ?? "unavailable"}</Badge>
      )}
      {status.synthetic_available && status.synthetic_low_confidence && (
        <Badge tone="danger">
          Synthetic seismogram low confidence (correlation {status.synthetic_correlation?.toFixed(3)})
        </Badge>
      )}
      {status.stale && (
        <Badge tone="orange">A newer upload has since replaced the active seismic volume</Badge>
      )}
    </div>
  );
}
