import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AxiosError } from "axios";
import Plot from "react-plotly.js";
import { getBlindWellPrediction } from "@/api/client";
import type { BlindWellPropertyResult, SpectralPropertyName } from "@/api/types";
import { useChartColors, usePlotlyLayout } from "@/styles/tokens";
import ChatPanel from "@/components/ChatPanel";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

const PROPERTY_LABELS: Record<SpectralPropertyName, string> = {
  vsh: "VSH (Shale Volume)",
  phie: "PHIE (Effective Porosity)",
  swe: "SWE (Water Saturation)",
};

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
    <span className={`inline-flex flex-col items-start px-3 py-1.5 rounded-xl border text-xs font-semibold ${toneClasses}`}>
      <span className="text-[10px] font-medium uppercase tracking-wide opacity-80">{label}</span>
      <span className="text-sm">{value}</span>
    </span>
  );
}

function PropertyPanel({ property, result }: { property: SpectralPropertyName; result: BlindWellPropertyResult }) {
  const colors = useChartColors();
  const plotlyLayout = usePlotlyLayout();
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  if (result.status !== "validated") {
    return (
      <div className="bg-surface border border-border rounded-xl p-4 shadow-card space-y-2">
        <h3 className="text-sm font-semibold text-ink">{PROPERTY_LABELS[property]}</h3>
        <div className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-3 py-2">
          {result.message ?? result.status}
        </div>
      </div>
    );
  }

  const chartData = result.depth_m.map((d, i) => ({
    depth: d,
    real: result.y_true[i],
    pred: result.y_pred[i],
  }));
  const rmseFmt = result.blind_well_rmse !== null ? result.blind_well_rmse.toFixed(3) : "—";

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card space-y-3">
      <h3 className="text-sm font-semibold text-ink">{PROPERTY_LABELS[property]}</h3>

      <div className="flex flex-wrap gap-2">
        <MetricBadge label="Decision gate R²" value={result.decision_gate_r2 !== null ? result.decision_gate_r2.toFixed(3) : "—"} tone={r2Tone(result.decision_gate_r2)} />
        <MetricBadge label="Stack LOOCV R²" value={result.stack_loocv_r2 !== null ? result.stack_loocv_r2.toFixed(3) : "—"} tone={r2Tone(result.stack_loocv_r2)} />
        <MetricBadge label="Blind well R²" value={result.blind_well_r2 !== null ? result.blind_well_r2.toFixed(3) : "—"} tone={r2Tone(result.blind_well_r2)} />
        <MetricBadge label="Blind well RMSE" value={rmseFmt} tone="muted" />
      </div>

      <p className="text-xs text-ink-faint">
        {result.n_training_samples.toLocaleString()} training samples (neighborhood-expanded) ·{" "}
        {result.n_blind_samples} blind-well samples
      </p>

      <div className="border border-border rounded-lg p-1.5">
        <Plot
          data={[
            {
              x: chartData.map((d) => d.real),
              y: chartData.map((d) => d.depth),
              type: "scatter",
              mode: "lines+markers",
              name: "Logged (real)",
              line: { color: colors.accent, width: 1.5 },
              marker: { size: 4 },
            },
            {
              x: chartData.map((d) => d.pred),
              y: chartData.map((d) => d.depth),
              type: "scatter",
              mode: "lines+markers",
              name: "Predicted",
              line: { color: colors.orange, width: 1.5 },
              marker: { size: 4 },
            },
          ]}
          layout={{
            ...plotlyLayout,
            autosize: true,
            height: 420,
            xaxis: { ...plotlyLayout.xaxis, title: { text: property.toUpperCase() } },
            yaxis: { ...plotlyLayout.yaxis, title: { text: "Depth (m)" }, autorange: "reversed" },
            legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.08, yanchor: "bottom" },
            margin: { t: 30, r: 15, b: 45, l: 55 },
          }}
          style={{ width: "100%" }}
          useResizeHandler
          config={{ displaylogo: false }}
        />
      </div>

      <div>
        <button
          onClick={() => setShowDiagnostics((v) => !v)}
          className="text-xs font-semibold text-accent-strong hover:underline"
        >
          {showDiagnostics ? "Hide" : "Show"} selected features & stability diagnostics
        </button>
        {showDiagnostics && (
          <div className="mt-2 space-y-2">
            <div className="flex flex-wrap gap-1.5">
              {result.selected_features.map((f) => (
                <span key={f} className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-accent-soft text-accent-strong border border-accent/30">
                  {f}
                </span>
              ))}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px] text-left">
                <thead>
                  <tr className="text-ink-faint border-b border-border">
                    <th className="py-1 pr-3 font-semibold">Feature</th>
                    <th className="py-1 pr-3 font-semibold">Pooled corr.</th>
                    <th className="py-1 pr-3 font-semibold">LOWO-stable</th>
                    <th className="py-1 pr-3 font-semibold">Selected</th>
                  </tr>
                </thead>
                <tbody>
                  {result.feature_diagnostics.map((d) => (
                    <tr key={d.feature} className="border-b border-border last:border-0">
                      <td className="py-1 pr-3 font-mono text-ink-muted">{d.feature}</td>
                      <td className="py-1 pr-3 text-ink-muted">{d.pooled_corr !== null ? d.pooled_corr.toFixed(3) : "—"}</td>
                      <td className="py-1 pr-3">{d.stable ? "✓" : "—"}</td>
                      <td className="py-1 pr-3">{d.selected ? <span className="text-accent-strong font-semibold">✓</span> : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * "Prediction" page: neighborhood-expanded CWT/SSWT + instantaneous-
 * attribute feature bank, leave-one-well-out-stable feature selection,
 * and a 4-base-learner stacking ensemble (XGBoost/ExtraTrees/
 * RandomForest/Ridge -> Ridge meta-learner), validated by holding ONE
 * well out ENTIRELY -- see blind_well_prediction_service.py's module
 * docstring for the full pipeline. The blind well defaults to the
 * backend's own default (Z-02) and isn't yet selectable here -- kept
 * deliberately single-well for now rather than rotating through all 7,
 * per the initial ask.
 */
export default function PredictionPage() {
  const [triggered, setTriggered] = useState(false);
  const query = useQuery({
    queryKey: ["blind-well-prediction"],
    queryFn: () => getBlindWellPrediction(),
    enabled: triggered,
    retry: false,
  });

  return (
    <div className="pb-12 space-y-4">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-brand-gradient-soft px-5 py-4">
        <div className="absolute -right-10 -top-10 h-40 w-40 rounded-full bg-orange/10 blur-2xl" />
        <div className="relative">
          <Link to="/" className="text-xs font-medium text-accent-strong hover:underline">
            ← Back to dashboard
          </Link>
          <p className="text-xs font-semibold uppercase tracking-wider text-accent-strong mb-1 mt-1">
            Blind-Well Prediction
          </p>
          <h1 className="text-xl font-extrabold text-ink tracking-tight">
            Neighborhood CWT/SSWT Stacking &amp; Blind-Well Validation
          </h1>
          <p className="text-sm text-ink-muted mt-1 max-w-3xl">
            Trains a stacked XGBoost/ExtraTrees/RandomForest/Ridge ensemble on every well EXCEPT the blind well
            (spatially expanded to each training well's trace neighborhood for more samples), then predicts the
            blind well's VSH/PHIE/SWE using only its own tied trace -- it never participates in feature selection or
            model training. A cheap single-feature "decision gate" R² is reported alongside the full stack's, so a
            weak or negative result doesn't get dressed up by ensemble complexity.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => setTriggered(true)}
          disabled={query.isFetching}
          className="text-xs font-semibold px-4 py-2 rounded-full bg-brand-gradient text-white shadow-card disabled:opacity-50"
        >
          {query.isFetching ? "Running…" : triggered ? "Re-run" : "Run blind-well prediction"}
        </button>
        <span className="text-xs text-ink-faint">
          Neighborhood-expanded CWT/SSWT feature extraction across several training wells -- this can take a minute.
        </span>
      </div>

      {query.isFetching && (
        <div className="h-40 rounded-xl bg-surface-sunken animate-pulse flex items-center justify-center text-xs text-ink-faint">
          Extracting neighborhood spectral features and training the stack…
        </div>
      )}

      {query.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Prediction failed: {errorMessage(query.error)}
        </div>
      )}

      {query.data && query.data.status !== "validated" && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          {query.data.message ?? query.data.status}
        </div>
      )}

      {query.data && query.data.status === "validated" && (
        <>
          <div className="flex flex-wrap items-center gap-4 text-xs font-semibold text-ink-muted">
            <span>
              Blind well: <span className="text-ink">{query.data.blind_well_id}</span>
            </span>
            <span>Training wells: {query.data.training_well_ids.join(", ")}</span>
            <span>Neighborhood radius: {query.data.neighborhood_radius_m?.toFixed(0)} m</span>
          </div>

          {query.data.excluded_wells.length > 0 && (
            <details className="border border-orange/30 bg-orange-soft/30 text-orange-strong text-xs rounded-xl px-4 py-2.5">
              <summary className="cursor-pointer font-semibold">
                {query.data.excluded_wells.length} well(s) excluded
              </summary>
              <ul className="mt-2 space-y-1 list-disc list-inside">
                {query.data.excluded_wells.map((w, i) => (
                  <li key={i}>
                    {w.well_id}: {w.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}

          {query.data.results && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {(["vsh", "phie", "swe"] as const).map((property) => (
                <PropertyPanel key={property} property={property} result={query.data!.results![property]} />
              ))}
            </div>
          )}
        </>
      )}

      <ChatPanel
        scope="dashboard"
        wellId={null}
        title="Prediction Assistant"
        subtitle="Ask about this blind-well validation and what the numbers mean"
      />
    </div>
  );
}
