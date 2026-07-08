import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { uploadSeismic } from "@/api/client";

/** Drag/select SEG-Y file(s) upload widget, mirroring UploadWells.tsx. */
export default function SeismicUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: uploadSeismic,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
      queryClient.invalidateQueries({ queryKey: ["seismic-datasets"] });
    },
  });

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    mutation.mutate(Array.from(fileList));
  }

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-ink">Upload Seismic Data</h3>
        <span className="inline-flex items-center gap-1 text-xs text-accent-strong bg-accent-soft px-2 py-0.5 rounded-full font-medium">
          .sgy / .segy
        </span>
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer border-2 border-dashed rounded-lg px-4 py-8 text-center transition-all ${
          dragActive
            ? "border-accent bg-accent-soft scale-[1.01]"
            : "border-border-strong bg-surface-muted hover:border-orange hover:bg-orange-soft/50"
        }`}
      >
        <div className="mx-auto mb-2 h-10 w-10 rounded-full bg-brand-gradient flex items-center justify-center">
          <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth={2} className="h-5 w-5">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 12h3l2-7 4 14 2-7h7"
            />
          </svg>
        </div>
        <p className="text-sm text-ink-muted font-medium">
          Drop raw <code className="font-mono text-accent-strong">.sgy</code> files here, or
          click to browse
        </p>
        <p className="text-xs text-ink-faint mt-1">
          Parsed with segyio; RMS amplitude, envelope, and VSH/PHIE/SWE proxies computed
          automatically
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".sgy,.segy"
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {mutation.isPending && (
        <p className="text-xs text-accent mt-2 flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
          Uploading and processing…
        </p>
      )}

      {mutation.isSuccess && mutation.data.uploaded.length > 0 && (
        <p className="text-xs text-pay mt-2 font-medium">
          Processed: {mutation.data.uploaded.map((s) => s.dataset_id).join(", ")}
        </p>
      )}

      {mutation.isSuccess && mutation.data.errors.length > 0 && (
        <div className="text-xs text-danger mt-2 space-y-0.5">
          {mutation.data.errors.map((e, i) => (
            <p key={i}>{e}</p>
          ))}
        </div>
      )}

      {mutation.isError && (
        <p className="text-xs text-danger mt-2">
          Upload failed: {(mutation.error as Error).message}
        </p>
      )}
    </div>
  );
}
