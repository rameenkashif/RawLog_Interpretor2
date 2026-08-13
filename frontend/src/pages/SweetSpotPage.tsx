import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { getSurveyInfo, trainSweetSpot } from "@/api/client";
import type { SweetSpotPropertyName, SweetSpotTargetResult } from "@/api/types";
import SweetSpotSectionView, { SWEET_SPOT_PROPERTIES } from "@/components/SweetSpot/SweetSpotSectionView";
import ChatPanel from "@/components/ChatPanel";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

function r2Tone(r2: number | null): "green" | "orange" | "danger" | "muted" {
  if (r2 === null) return "muted";
  if (r2 >= 0.3) return "green";
  if (r2 >= 0) return "orange";
  return "danger";
}

function MetricBadge({ label, value, tone }: { label: string; value: string; tone: "green" | "orange" | "danger" | "muted" }) {
  const toneClasses = {
    green: "border-success/30 bg-success-soft text-success",
    orange: "border-orange/30 bg-orange-soft text-orange-strong",
    danger: "border-danger/40 bg-danger-soft text-danger",
    muted: "border-border-strong bg-surface-sunken text-ink-faint",
  }[tone];
  return (
    <span className={`inline-flex flex-col items-start px-2.5 py-1 rounded-lg border text-[11px] font-semibold ${toneClasses}`}>
      <span className="text-[9px] font-medium uppercase tracking-wide opacity-80">{label}</span>
      <span className="text-xs">{value}</span>
    </span>
  );
}

function TargetDiagnosticChip({
  propertyKey,
  label,
  result,
  active,
  onSelect,
}: {
  propertyKey: SweetSpotPropertyName;
  label: string;
  result: SweetSpotTargetResult | undefined;
  active: boolean;
  onSelect: (key: SweetSpotPropertyName) => void;
}) {
  const fmt = (v: number | null) => (v !== null ? v.toFixed(3) : "—");
  return (
    <button
      onClick={() => onSelect(propertyKey)}
      className={`text-left rounded-xl border p-3 space-y-2 transition-all ${
        active ? "border-accent shadow-card bg-accent-soft/40" : "border-border bg-surface hover:border-accent/50"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-ink">{label}</span>
        {result?.model_name && <span className="text-[10px] font-mono text-ink-faint">{result.model_name}</span>}
      </div>
      {!result || result.status !== "validated" ? (
        <p className="text-[11px] text-orange-strong">{result?.message ?? "Not trained"}</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          <MetricBadge label="CV R²" value={fmt(result.cv_r2)} tone={r2Tone(result.cv_r2)} />
          <MetricBadge label="Blind R²" value={fmt(result.blind_well_r2)} tone={r2Tone(result.blind_well_r2)} />
          {result.facies_alpha !== null && (
            <MetricBadge label="Facies α" value={result.facies_alpha.toFixed(2)} tone="muted" />
          )}
        </div>
      )}
    </button>
  );
}

/**
 * Sweet-Spot Prediction: the V11-style 2-stage cascaded ML pipeline
 * (phase-rotated well tie, curated 22-attribute seismic feature engine,
 * dynamic calibration, LOGO-CV model selection across an 11-template
 * pool with facies modulation, and a genuine blind-well holdout) --
 * train once, then pick an inline "sweet spot" and see any of
 * AI/DT/PHIT/GR/RHOB/VSH/PHIE/SWE painted across the whole section with
 * live hover readout, not just validated at the blind well's own trace.
 */
export default function SweetSpotPage() {
  const [triggered, setTriggered] = useState(false);
  const [property, setProperty] = useState<SweetSpotPropertyName>("gr");

  const surveyQuery = useQuery({ queryKey: ["survey-info"], queryFn: getSurveyInfo });

  const trainQuery = useQuery({
    queryKey: ["sweet-spot-train"],
    queryFn: () => trainSweetSpot(),
    enabled: triggered,
    retry: false,
  });

  const trained = trainQuery.data?.status === "validated";

  return (
    <div className="pb-12 space-y-4">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-5 py-4">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative">
          <Link to="/" className="text-xs font-medium text-accent-strong hover:underline">
            ← Back to dashboard
          </Link>
          <p className="text-xs font-semibold uppercase tracking-wider text-accent-strong mb-1 mt-1">
            Sweet-Spot Prediction
          </p>
          <h1 className="text-xl font-extrabold text-ink tracking-tight">
            2-Stage Cascaded Property Prediction Across the Section
          </h1>
          <p className="text-sm text-ink-muted mt-1 max-w-3xl">
            Trains Stage 1 (AI, DT, PHIT) and Stage 2 (GR, RHOB, VSH, PHIE, SWE -- using Stage 1's own
            out-of-fold predictions as cascade features) on every well except a held-out blind well, with
            per-well dynamic calibration and leave-one-group-out model selection across an 11-template pool
            (including facies-modulated blends for GR/VSH/PHIE/SWE). Once trained, pick an inline "sweet spot"
            below and hover the section to read the predicted property at any point.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => setTriggered(true)}
          disabled={trainQuery.isFetching}
          className="text-xs font-semibold px-4 py-2 rounded-full bg-brand-gradient text-white shadow-card disabled:opacity-50"
        >
          {trainQuery.isFetching ? "Training…" : triggered ? "Re-train" : "Train models"}
        </button>
        <span className="text-xs text-ink-faint">
          Trains an 11-template pool per target with leave-one-group-out cross-validation -- this can take
          several minutes.
        </span>
      </div>

      {trainQuery.isFetching && (
        <div className="h-24 rounded-xl bg-surface-sunken animate-pulse flex items-center justify-center text-xs text-ink-faint">
          Building the 22-feature curated set, running LOGO-CV across the model pool for every target…
        </div>
      )}

      {trainQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Training failed: {errorMessage(trainQuery.error)}
        </div>
      )}

      {trainQuery.data && trainQuery.data.status !== "validated" && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          {trainQuery.data.message ?? trainQuery.data.status}
        </div>
      )}

      {trainQuery.data && trainQuery.data.status === "validated" && (
        <>
          <div className="flex flex-wrap items-center gap-4 text-xs font-semibold text-ink-muted">
            <span>
              Blind well: <span className="text-ink">{trainQuery.data.blind_well_id}</span>
            </span>
            <span>Training wells: {trainQuery.data.training_well_ids.join(", ")}</span>
          </div>

          {trainQuery.data.excluded_wells.length > 0 && (
            <details className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5">
              <summary className="cursor-pointer font-semibold">
                {trainQuery.data.excluded_wells.length} well(s) excluded
              </summary>
              <ul className="mt-2 space-y-1 list-disc list-inside">
                {trainQuery.data.excluded_wells.map((w, i) => (
                  <li key={i}>
                    {w.well_id}: {w.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {SWEET_SPOT_PROPERTIES.map(({ key, label }) => (
              <TargetDiagnosticChip
                key={key}
                propertyKey={key}
                label={label}
                result={trainQuery.data!.results?.[key]}
                active={property === key}
                onSelect={setProperty}
              />
            ))}
          </div>
        </>
      )}

      {surveyQuery.data && (
        <div className="bg-surface border border-border rounded-xl p-4 shadow-card space-y-2">
          <h2 className="text-sm font-semibold text-ink">
            Predicted {SWEET_SPOT_PROPERTIES.find((p) => p.key === property)?.label} across the sweet spot
          </h2>
          <SweetSpotSectionView
            surveyInfo={surveyQuery.data}
            blindWellId={trainQuery.data?.blind_well_id ?? "Z-02_RAW"}
            property={property}
            trained={trained}
          />
        </div>
      )}

      <ChatPanel
        scope="dashboard"
        wellId={null}
        title="Sweet-Spot Assistant"
        subtitle="Ask about this cascade and what the mapped predictions mean"
      />
    </div>
  );
}
