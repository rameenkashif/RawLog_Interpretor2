import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { getPrediction, getPredictionImageUrl, listWells } from "@/api/client";
import type { PredictionMethod, PredictionTarget } from "@/api/types";
import { Badge } from "@/components/Synthetic/QcBadges";
import { useAppStore } from "@/store/useAppStore";

const TARGETS: { key: PredictionTarget; label: string }[] = [
  { key: "vsh", label: "VSH" },
  { key: "phie", label: "PHIE" },
  { key: "swe", label: "SWE" },
];
const METHODS: { key: PredictionMethod; label: string }[] = [
  { key: "cwt", label: "CWT" },
  { key: "sswt", label: "SSWT" },
];

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function fmtR2(v: number | null): string {
  return v === null ? "n/a" : v.toFixed(3);
}

/**
 * Prediction page: blind-well VSH/PHIE/SWE from CWT/SSWT amplitude via
 * Ridge regression, plotted as a true-vs-predicted inline section --
 * ported from a user-supplied reference pipeline (see backend/app/
 * services/prediction_pipeline_service.py's docstring for exactly what
 * was kept vs. changed from that reference: only the well->trace tie
 * uses this app's own validated direct-tie method instead of the
 * reference script's hand-fit coordinate anchors).
 *
 * Deliberately a separate page/model from the Seismic panel's Spectral
 * Property Prediction tab (RandomForest + full leave-one-well-out) --
 * this one holds out ONE named blind well at a time with Ridge
 * regression, matching the reference pipeline's own approach.
 */
export default function PredictionPage() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const activeWellId = useAppStore((s) => s.activeWellId);

  const [blindWellId, setBlindWellId] = useState<string | null>(null);
  const [target, setTarget] = useState<PredictionTarget>("vsh");
  const [method, setMethod] = useState<PredictionMethod>("cwt");

  // Seed from the dashboard's shared active well, same convention as the
  // other well-scoped pages -- a manual pick below still overrides this.
  useEffect(() => {
    if (activeWellId) setBlindWellId(activeWellId);
    else if (!blindWellId && wellsQuery.data && wellsQuery.data.length > 0) {
      setBlindWellId(wellsQuery.data[0].well_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeWellId, wellsQuery.data]);

  const predictionQuery = useQuery({
    queryKey: ["prediction", blindWellId, target, method],
    queryFn: () => getPrediction(blindWellId!, target, method),
    enabled: Boolean(blindWellId),
  });

  const data = predictionQuery.data;

  return (
    <div className="pb-12 space-y-4">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-5 py-4">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative">
          <Link to="/" className="text-xs font-medium text-accent-strong hover:underline">
            ← Back to dashboard
          </Link>
          <p className="text-xs font-semibold uppercase tracking-wider text-accent-strong mb-1 mt-1">
            Prediction
          </p>
          <h1 className="text-xl font-extrabold text-ink tracking-tight">Blind-Well Property Prediction</h1>
          <p className="text-sm text-ink-muted mt-1 max-w-2xl">
            Holds one well out entirely, trains a Ridge regression on CWT/SSWT amplitude from every
            other well, and predicts the held-out (blind) well's VSH/PHIE/SWE -- never trained on its
            own data. Well location is resolved via the same direct nearest-trace tie validated on the
            Well-to-Seismic Tie page, not a hand-fit coordinate transform.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
          Blind well
          <select
            value={blindWellId ?? ""}
            onChange={(e) => setBlindWellId(e.target.value || null)}
            className="text-xs border border-border-strong rounded-lg px-2 py-1"
          >
            <option value="">Select well…</option>
            {wellsQuery.data?.map((w) => (
              <option key={w.well_id} value={w.well_id}>
                {w.well_id}
              </option>
            ))}
          </select>
        </label>

        <div className="flex gap-1.5">
          {TARGETS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setTarget(key)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                target === key
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex gap-1.5">
          {METHODS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setMethod(key)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                method === key
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {!blindWellId && (
        <div className="bg-surface border border-border rounded-xl p-6 text-center text-sm text-ink-faint shadow-card">
          Select a well to hold out and predict blind.
        </div>
      )}

      {predictionQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}
      {predictionQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load: {errorMessage(predictionQuery.error)}
        </div>
      )}

      {data && data.status === "insufficient_data" && (
        <div className="border border-orange/30 bg-orange-soft/40 text-orange-strong text-sm rounded-xl px-4 py-4">
          {data.message}
        </div>
      )}

      {data && data.status === "validated" && data.result && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge tone="accent">
              Tied: IL {data.inline_number} / XL {data.crossline_number} (r={data.tie_correlation?.toFixed(3)})
            </Badge>
            <Badge tone={data.result.r2 !== null && data.result.r2 > 0 ? "green" : "orange"}>
              Blind R² = {fmtR2(data.result.r2)}
            </Badge>
            <Badge tone="accent">
              Trained on {data.n_train_wells} well(s), {data.result.n_train_samples} samples
            </Badge>
            <Badge tone="accent">{data.result.n_test_samples} blind samples</Badge>
          </div>

          <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
            <img
              key={`${blindWellId}-${target}-${method}`}
              src={getPredictionImageUrl(blindWellId!, target, method)}
              alt={`${blindWellId} true vs predicted ${target.toUpperCase()}`}
              className="w-full rounded-lg"
            />
            <p className="text-xs text-ink-faint px-2 pb-2 pt-1">
              Left: {blindWellId}'s real (logged) {target.toUpperCase()}, painted as a colored strip at
              its tied crossline. Right: the model's blind prediction at the same location -- this
              well's own data never appeared in training.
            </p>
          </div>
        </div>
      )}

      {data && data.excluded_wells.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-ink-muted">Excluded from training</p>
          <div className="flex flex-wrap gap-2">
            {data.excluded_wells.map((w) => (
              <Badge key={w.well_id} tone="orange">
                {w.well_id} — {w.reason}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
