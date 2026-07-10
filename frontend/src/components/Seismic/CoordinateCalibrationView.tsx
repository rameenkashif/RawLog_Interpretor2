import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import {
  deleteCoordinateOverride,
  getCoordinateCalibrationReport,
  listCoordinateOverrides,
  recalibrateCoordinates,
  saveCoordinateOverride,
} from "@/api/client";
import type { WellCalibrationReportItem } from "@/api/types";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function TrustBadge({ well }: { well: WellCalibrationReportItem }) {
  if (well.has_manual_override) {
    return (
      <span className="text-xs font-semibold px-2.5 py-1 rounded-full border border-accent/30 bg-accent-soft text-accent-strong">
        Manual override
      </span>
    );
  }
  if (well.trustworthy) {
    return (
      <span className="text-xs font-semibold px-2.5 py-1 rounded-full border border-green-200 bg-green-50 text-green-700">
        Trustworthy
      </span>
    );
  }
  return (
    <span className="text-xs font-semibold px-2.5 py-1 rounded-full border border-red-300 bg-red-50 text-danger">
      Unresolved{well.is_extrapolated ? " — extrapolated" : " — outside bin tolerance"}
    </span>
  );
}

/**
 * "Coordinate Calibration" view: surfaces the well<->seismic per-axis
 * linear coordinate fit's diagnostics (coordinate_calibration_service.py)
 * -- residual distance vs. survey bin spacing, extrapolation flag, and
 * whether each well is part of the calibration baseline -- plus a manual
 * tie-point override table, since fixes #4/#5 explicitly require this
 * NOT be a silent pass/fail: only a well flagged trustworthy (or with a
 * manual override) should have downstream tie/prediction workflows run
 * on it, and a user needs to see why before trusting a tie.
 */
export default function CoordinateCalibrationView() {
  const queryClient = useQueryClient();
  const [overrideDrafts, setOverrideDrafts] = useState<Record<string, { inline: string; crossline: string; note: string }>>(
    {},
  );

  const reportQuery = useQuery({
    queryKey: ["coordinate-calibration-report"],
    queryFn: getCoordinateCalibrationReport,
    retry: false,
  });

  const overridesQuery = useQuery({
    queryKey: ["coordinate-overrides"],
    queryFn: listCoordinateOverrides,
  });

  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: ["coordinate-calibration-report"] });
    queryClient.invalidateQueries({ queryKey: ["coordinate-overrides"] });
  };

  const recalibrateMutation = useMutation({
    mutationFn: () => recalibrateCoordinates(null),
    onSuccess: invalidateAll,
  });

  const saveOverrideMutation = useMutation({
    mutationFn: ({ wellId, inline, crossline, note }: { wellId: string; inline: number; crossline: number; note: string }) =>
      saveCoordinateOverride(wellId, { inline, crossline, note }),
    onSuccess: invalidateAll,
  });

  const deleteOverrideMutation = useMutation({
    mutationFn: (wellId: string) => deleteCoordinateOverride(wellId),
    onSuccess: invalidateAll,
  });

  const overrideByWellId = new Map((overridesQuery.data ?? []).map((o) => [o.well_id, o]));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-ink-muted max-w-2xl leading-relaxed">
          Well header X/Y and SEG-Y trace coordinates are on different, unknown coordinate reference
          systems -- there is no real reprojection available, so this fits a 2-point-per-axis linear
          approximation instead. Only trust a well flagged <span className="font-semibold">trustworthy</span>,
          or one with a manual override; treat any other well&apos;s tie as unresolved.
        </p>
        <button
          onClick={() => recalibrateMutation.mutate()}
          disabled={recalibrateMutation.isPending}
          className="text-xs font-semibold px-3.5 py-1.5 rounded-full border border-border-strong bg-surface text-ink-muted hover:border-accent hover:text-accent transition-all disabled:opacity-50"
        >
          {recalibrateMutation.isPending ? "Recalibrating…" : "Recalibrate from all wells"}
        </button>
      </div>

      {reportQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {reportQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Calibration unavailable: {errorMessage(reportQuery.error)}
        </div>
      )}

      {reportQuery.data && (
        <>
          <div className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
            {reportQuery.data.method_note}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="text-ink-faint border-b border-border">
                  <th className="py-1.5 pr-4 font-semibold">Well</th>
                  <th className="py-1.5 pr-4 font-semibold">Status</th>
                  <th className="py-1.5 pr-4 font-semibold">Tied inline/crossline</th>
                  <th className="py-1.5 pr-4 font-semibold">Residual (m)</th>
                  <th className="py-1.5 pr-4 font-semibold">In calibration baseline</th>
                  <th className="py-1.5 pr-4 font-semibold">Manual override</th>
                </tr>
              </thead>
              <tbody>
                {reportQuery.data.wells.map((w) => {
                  const draft = overrideDrafts[w.well_id] ?? {
                    inline: String(w.override_inline ?? w.nearest_inline),
                    crossline: String(w.override_crossline ?? w.nearest_crossline),
                    note: "",
                  };
                  const existingOverride = overrideByWellId.get(w.well_id);
                  return (
                    <tr key={w.well_id} className="border-b border-border last:border-0 align-top">
                      <td className="py-2 pr-4 font-semibold text-ink">{w.well_name}</td>
                      <td className="py-2 pr-4">
                        <TrustBadge well={w} />
                      </td>
                      <td className="py-2 pr-4 text-ink-muted">
                        {w.nearest_inline} / {w.nearest_crossline}
                      </td>
                      <td className="py-2 pr-4 text-ink-muted">{w.nearest_trace_distance_m.toFixed(0)}</td>
                      <td className="py-2 pr-4 text-ink-muted">{w.used_in_calibration ? "Yes" : "No"}</td>
                      <td className="py-2 pr-4">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <input
                            type="number"
                            value={draft.inline}
                            onChange={(e) =>
                              setOverrideDrafts((prev) => ({
                                ...prev,
                                [w.well_id]: { ...draft, inline: e.target.value },
                              }))
                            }
                            className="w-16 text-xs border border-border-strong rounded-lg px-1.5 py-1"
                            placeholder="inline"
                          />
                          <input
                            type="number"
                            value={draft.crossline}
                            onChange={(e) =>
                              setOverrideDrafts((prev) => ({
                                ...prev,
                                [w.well_id]: { ...draft, crossline: e.target.value },
                              }))
                            }
                            className="w-16 text-xs border border-border-strong rounded-lg px-1.5 py-1"
                            placeholder="crossline"
                          />
                          <button
                            onClick={() =>
                              saveOverrideMutation.mutate({
                                wellId: w.well_id,
                                inline: Number(draft.inline),
                                crossline: Number(draft.crossline),
                                note: draft.note,
                              })
                            }
                            disabled={saveOverrideMutation.isPending}
                            className="text-xs font-semibold px-2 py-1 rounded-lg bg-brand-gradient text-white disabled:opacity-50"
                          >
                            Save
                          </button>
                          {existingOverride && (
                            <button
                              onClick={() => deleteOverrideMutation.mutate(w.well_id)}
                              disabled={deleteOverrideMutation.isPending}
                              className="text-xs font-semibold px-2 py-1 rounded-lg border border-border-strong text-ink-muted hover:border-danger hover:text-danger disabled:opacity-50"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
