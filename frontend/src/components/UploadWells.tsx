import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { uploadWells } from "@/api/client";

/** Drag/select LAS file(s) upload widget for the dashboard (section 4 upload endpoint). */
export default function UploadWells() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: uploadWells,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
      queryClient.invalidateQueries({ queryKey: ["wells"] });
    },
  });

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    mutation.mutate(Array.from(fileList));
  }

  return (
    <div className="bg-surface border border-border rounded-lg p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-ink mb-2">Upload Well Logs</h3>
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
        className={`cursor-pointer border-2 border-dashed rounded-md px-4 py-8 text-center transition-colors ${
          dragActive ? "border-accent bg-accent-soft" : "border-border-strong bg-surface-muted"
        }`}
      >
        <p className="text-sm text-ink-muted">
          Drop raw <code className="font-mono">.las</code> files here, or click to browse
        </p>
        <p className="text-xs text-ink-faint mt-1">
          Requires curves: DEPT, GR, RESISTIVITY, RHOB, NPHI, DT
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".las"
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {mutation.isPending && <p className="text-xs text-accent mt-2">Uploading and processing…</p>}

      {mutation.isSuccess && mutation.data.uploaded.length > 0 && (
        <p className="text-xs text-pay mt-2">
          Processed: {mutation.data.uploaded.map((w) => w.well_id).join(", ")}
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
