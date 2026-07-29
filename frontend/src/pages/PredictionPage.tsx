import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import {
  getPrediction,
  getPredictionFrequencyMapUrl,
  getPredictionImageUrl,
  getPredictionInlineMapsUrl,
  getPredictionLoocvHeatmapUrl,
  getSectionImageUrl,
  listWells,
} from "@/api/client";
import type { PredictionMethod, PredictionResponse, PredictionTarget } from "@/api/types";
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
 * Prediction page: SEG-Y+LAS -> frequency maps -> well tie -> CWT/SSWT ->
 * blind-well VSH/PHIE/SWE prediction -> inline display, ported from a
 * user-supplied reference pipeline (see backend/app/services/
 * prediction_pipeline_service.py's docstring for exactly what was kept
 * vs. changed: only the well->trace tie uses this app's own validated
 * direct-tie method instead of the reference script's hand-fit
 * coordinate anchors -- everything else, including the Ridge regression
 * model and full leave-one-well-out heatmap, mirrors the reference
 * pipeline).
 *
 * Deliberately a separate page/model from the Seismic panel's Spectral
 * Property Prediction tab (RandomForest).
 */
export default function PredictionPage() {
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const activeWellId = useAppStore((s) => s.activeWellId);

  const [blindWellId, setBlindWellId] = useState<string | null>(null);
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

  // One query per property (a fixed set of exactly 3, so three explicit
  // hooks instead of a dynamic useQueries -- all share the current
  // method toggle).
  const vshQuery = useQuery({
    queryKey: ["prediction", blindWellId, "vsh", method],
    queryFn: () => getPrediction(blindWellId!, "vsh", method),
    enabled: Boolean(blindWellId),
  });
  const phieQuery = useQuery({
    queryKey: ["prediction", blindWellId, "phie", method],
    queryFn: () => getPrediction(blindWellId!, "phie", method),
    enabled: Boolean(blindWellId),
  });
  const sweQuery = useQuery({
    queryKey: ["prediction", blindWellId, "swe", method],
    queryFn: () => getPrediction(blindWellId!, "swe", method),
    enabled: Boolean(blindWellId),
  });
  const queriesByTarget: Record<PredictionTarget, typeof vshQuery> = {
    vsh: vshQuery,
    phie: phieQuery,
    swe: sweQuery,
  };

  // Tie/exclusion info is identical across the three queries (it only
  // depends on the tie, not the target) -- just use whichever loaded
  // first that reached "validated".
  const tieInfo = [vshQuery.data, phieQuery.data, sweQuery.data].find((d) => d?.status === "validated");
  const excludedWells =
    vshQuery.data?.excluded_wells ?? phieQuery.data?.excluded_wells ?? sweQuery.data?.excluded_wells ?? [];

  return (
    <div className="pb-12 space-y-5">
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

      {excludedWells.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-ink-muted">Excluded from training</p>
          <div className="flex flex-wrap gap-2">
            {excludedWells.map((w) => (
              <Badge key={w.well_id} tone="orange">
                {w.well_id} — {w.reason}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {blindWellId && (
        <>
          {/* 1. Frequency maps */}
          <Section title="Frequency map">
            <img
              key={`freq-${blindWellId}`}
              src={getPredictionFrequencyMapUrl(blindWellId)}
              alt={`${blindWellId} frequency map`}
              className="w-full rounded-lg"
            />
          </Section>

          {/* 2. Time-domain inline section */}
          {tieInfo?.inline_number != null && (
            <Section title={`Time-domain inline section (Inline ${tieInfo.inline_number})`}>
              <img
                key={`section-${tieInfo.inline_number}`}
                src={getSectionImageUrl("inline", tieInfo.inline_number)}
                alt={`Inline ${tieInfo.inline_number} section`}
                className="w-full rounded-lg"
              />
            </Section>
          )}

          {/* 3. Well tie for the selected well */}
          {tieInfo && (
            <Section title={`Well tie -- ${blindWellId}`}>
              <div className="flex flex-wrap gap-2 px-1 pb-1">
                <Badge tone="accent">
                  IL {tieInfo.inline_number} / XL {tieInfo.crossline_number}
                </Badge>
                <Badge tone={tieInfo.tie_correlation !== null && tieInfo.tie_correlation >= 0.5 ? "green" : "orange"}>
                  Correlation = {tieInfo.tie_correlation?.toFixed(3) ?? "n/a"}
                </Badge>
                <Badge tone="accent">Best freq = {tieInfo.tie_best_freq_hz?.toFixed(1) ?? "n/a"} Hz</Badge>
                <Badge tone="accent">
                  Polarity = {tieInfo.tie_polarity === 1 ? "normal" : tieInfo.tie_polarity === -1 ? "reversed" : "n/a"}
                </Badge>
                <Badge tone="accent">Bulk shift = {tieInfo.tie_bulk_shift_ms?.toFixed(1) ?? "n/a"} ms</Badge>
              </div>
            </Section>
          )}

          {/* 4. True vs predicted for the blind well, per property */}
          <div className="space-y-4">
            <p className="text-sm font-bold text-ink">True vs. predicted -- {blindWellId} (blind)</p>
            {TARGETS.map(({ key, label }) => (
              <PropertySection
                key={key}
                target={key}
                label={label}
                method={method}
                blindWellId={blindWellId}
                query={queriesByTarget[key]}
              />
            ))}
          </div>

          {/* 5. Full LOOCV heatmap */}
          <Section title="Full leave-one-well-out R² (every well held out in turn)">
            <img
              key="full-loocv"
              src={getPredictionLoocvHeatmapUrl()}
              alt="Full LOOCV R2 heatmap"
              className="mx-auto max-w-sm rounded-lg"
            />
          </Section>

          {/* 6. Real vs predicted VSH/PHIE/SWE painted across the whole inline */}
          <Section title={`Real vs. predicted VSH / PHIE / SWE across Inline ${tieInfo?.inline_number ?? ""}`}>
            <p className="text-xs text-orange-strong bg-orange-soft/40 border border-orange/30 rounded-lg px-3 py-2 mb-2">
              Exploratory, not a validated spatial map: blind R² for this pipeline has been near zero
              or negative for most property/method combinations (see the LOOCV heatmap above) -- a
              confidently colored map does not mean the colors are trustworthy. A real property value
              only exists at the well itself; away from it, the right-hand panels are the model's
              estimate only.
            </p>
            <img
              key={`inline-maps-${blindWellId}-${method}`}
              src={getPredictionInlineMapsUrl(blindWellId, method)}
              alt={`${blindWellId} real vs predicted VSH/PHIE/SWE inline maps`}
              className="w-full rounded-lg"
            />
          </Section>
        </>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="bg-surface border border-border rounded-xl p-3 shadow-card space-y-2">
      <p className="text-xs font-semibold text-ink">{title}</p>
      {children}
    </div>
  );
}

function PropertySection({
  target,
  label,
  method,
  blindWellId,
  query,
}: {
  target: PredictionTarget;
  label: string;
  method: PredictionMethod;
  blindWellId: string;
  query: { data?: PredictionResponse; isLoading: boolean; isError: boolean; error: unknown };
}) {
  const data = query.data;

  return (
    <div className="space-y-2">
      <p className="text-xs font-bold text-ink-muted">{label}</p>

      {query.isLoading && <div className="h-48 rounded-xl bg-surface-sunken animate-pulse" />}
      {query.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load {label}: {errorMessage(query.error)}
        </div>
      )}

      {data && data.status === "insufficient_data" && (
        <div className="border border-orange/30 bg-orange-soft/40 text-orange-strong text-sm rounded-xl px-4 py-4">
          {data.message}
        </div>
      )}

      {data && data.status === "validated" && data.result && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <Badge tone={data.result.r2 !== null && data.result.r2 > 0 ? "green" : "orange"}>
              Blind R² = {fmtR2(data.result.r2)}
            </Badge>
            <Badge tone="accent">
              Trained on {data.n_train_wells} well(s), {data.result.n_train_samples} samples
            </Badge>
            <Badge tone="accent">{data.result.n_test_samples} blind samples</Badge>
          </div>

          <div className="bg-surface-sunken border border-border rounded-xl p-2">
            <img
              key={`${blindWellId}-${target}-${method}`}
              src={getPredictionImageUrl(blindWellId, target, method)}
              alt={`${blindWellId} true vs predicted ${label}`}
              className="w-full rounded-lg"
            />
            <p className="text-xs text-ink-faint px-2 pb-1 pt-1">
              Left: {blindWellId}'s real (logged) {label}, painted as a colored strip at its tied
              crossline. Right: the model's blind prediction at the same location -- this well's own
              data never appeared in training.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
