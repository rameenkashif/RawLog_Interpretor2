import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { generateSyntheticSeismogram, getSyntheticExportUrl, listWells } from "@/api/client";
import type { DensityMethod, WaveletMethod } from "@/api/types";
import QcBadges from "@/components/Synthetic/QcBadges";
import AcousticImpedanceChart from "@/components/Synthetic/AcousticImpedanceChart";
import WaveletView from "@/components/Synthetic/WaveletView";
import SyntheticTraceOverlay from "@/components/Synthetic/SyntheticTraceOverlay";
import WashoutSummary from "@/components/Synthetic/WashoutSummary";
import StretchSqueezeControls from "@/components/Synthetic/StretchSqueezeControls";
import ChatPanel from "@/components/ChatPanel";
import { useAppStore } from "@/store/useAppStore";

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Synthetic Seismogram / well-tie module: unit-standardized well header QC,
 * density estimation (real RHOB / calibrated Gardner / rock-physics),
 * acoustic impedance + reflectivity, wavelet extraction/generation with
 * spectra, synthetic-vs-real trace tie, washout QC proxy, and persisted
 * manual stretch/squeeze -- reuses the same well-selection pattern and
 * light-mode UI conventions as the rest of the dashboard.
 */
export default function SyntheticSeismogramPage() {
  const queryClient = useQueryClient();
  const wellsQuery = useQuery({ queryKey: ["wells"], queryFn: listWells });
  const activeWellId = useAppStore((s) => s.activeWellId);

  const [wellId, setWellId] = useState<string | null>(null);
  // Seed/redirect from the dashboard's shared "active well" (e.g. right
  // after a combined upload) -- a manual pick from the dropdown below
  // still overrides this until the active well changes again.
  useEffect(() => {
    if (activeWellId) setWellId(activeWellId);
  }, [activeWellId]);

  const [waveletMethod, setWaveletMethod] = useState<WaveletMethod>("statistical");
  const [waveletFreqHz, setWaveletFreqHz] = useState(25);
  const [densityMethod, setDensityMethod] = useState<DensityMethod>("rhob");
  const [autoOptimizeTie, setAutoOptimizeTie] = useState(false);

  const queryKey = ["synthetic-generate", wellId, waveletMethod, waveletFreqHz, densityMethod, autoOptimizeTie];
  const genQuery = useQuery({
    queryKey,
    queryFn: () =>
      generateSyntheticSeismogram(wellId!, {
        waveletMethod,
        waveletFreqHz,
        densityMethod,
        autoOptimizeTie,
      }),
    enabled: Boolean(wellId),
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
            Synthetic Seismogram Module
          </p>
          <h1 className="text-xl font-extrabold text-ink tracking-tight">Synthetic Seismogram &amp; Well Tie</h1>
          <p className="text-sm text-ink-muted mt-1 max-w-2xl">
            Density → acoustic impedance → reflectivity → wavelet convolution → synthetic trace, tied against the
            nearest real seismic trace. Unit-standardized well header (X/Y/KB/TD), selectable density/wavelet, and
            manual stretch/squeeze since no checkshot survey is available.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select
          className="text-sm border border-border-strong rounded-lg px-3 py-1.5"
          value={wellId ?? ""}
          onChange={(e) => setWellId(e.target.value || null)}
        >
          <option value="">Select well…</option>
          {wellsQuery.data?.map((w) => (
            <option key={w.well_id} value={w.well_id}>
              {w.well_id}
            </option>
          ))}
        </select>

        <div className="flex gap-1.5">
          {(["statistical", "ricker"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setWaveletMethod(m)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all capitalize ${
                waveletMethod === m
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {m} wavelet
            </button>
          ))}
        </div>

        {waveletMethod === "ricker" && (
          <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
            Freq (Hz)
            <input
              type="number"
              min={1}
              max={200}
              value={waveletFreqHz}
              onChange={(e) => setWaveletFreqHz(Number(e.target.value) || 25)}
              disabled={autoOptimizeTie}
              title={autoOptimizeTie ? "Ignored -- auto-optimize searches frequency instead" : undefined}
              className="w-16 text-xs border border-border-strong rounded-lg px-2 py-1 disabled:opacity-50"
            />
          </label>
        )}

        <button
          onClick={() => setAutoOptimizeTie((v) => !v)}
          title="Search wavelet frequency (ricker) and polarity, not just shift position, keeping whichever combination maximizes correlation -- more expensive, off by default"
          className={`text-xs font-semibold px-3.5 py-1.5 rounded-full border transition-all ${
            autoOptimizeTie
              ? "bg-brand-gradient text-white border-transparent shadow-card"
              : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
          }`}
        >
          Auto-optimize tie
        </button>

        <div className="flex gap-1.5">
          {(["rhob", "gardner", "rock_physics"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setDensityMethod(m)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all ${
                densityMethod === m
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {m === "rhob" ? "Real RHOB" : m === "gardner" ? "Gardner" : "Rock-physics"}
            </button>
          ))}
        </div>

        {wellId && (
          <a
            href={getSyntheticExportUrl(wellId, { waveletMethod, waveletFreqHz, densityMethod, autoOptimizeTie })}
            className="ml-auto text-xs font-semibold px-3.5 py-1.5 rounded-full border border-accent/30 bg-accent-soft text-accent-strong hover:bg-accent hover:text-white transition-colors"
          >
            Export CSV
          </a>
        )}
      </div>

      {!wellId && (
        <div className="bg-surface border border-border rounded-xl p-8 text-center text-sm text-ink-faint shadow-card">
          Select a well to generate its synthetic seismogram and well tie.
        </div>
      )}

      {genQuery.isLoading && <div className="h-64 rounded-xl bg-surface-sunken animate-pulse" />}

      {genQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to generate synthetic seismogram: {errorMessage(genQuery.error)}
        </div>
      )}

      {genQuery.data && (
        <div className="space-y-4">
          <QcBadges result={genQuery.data} />

          <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5">
            {genQuery.data.density_note}
            {genQuery.data.gardner_coefficients && (
              <span className="ml-1">
                (a={genQuery.data.gardner_coefficients.a.toFixed(4)}, b=
                {genQuery.data.gardner_coefficients.b.toFixed(4)})
              </span>
            )}
          </div>

          <section>
            <h2 className="text-sm font-semibold text-ink mb-1.5">Acoustic Impedance &amp; Reflectivity</h2>
            <AcousticImpedanceChart result={genQuery.data} />
          </section>

          <section>
            <h2 className="text-sm font-semibold text-ink mb-1.5">Wavelet</h2>
            <WaveletView result={genQuery.data} />
          </section>

          <section>
            <h2 className="text-sm font-semibold text-ink mb-1.5">Synthetic vs. Real Trace</h2>
            <SyntheticTraceOverlay result={genQuery.data} />
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <WashoutSummary result={genQuery.data} />
            <StretchSqueezeControls
              wellId={wellId!}
              appliedPoints={genQuery.data.applied_tie_points}
              waveletMethod={waveletMethod}
              waveletFreqHz={waveletFreqHz}
              onApplied={() => queryClient.invalidateQueries({ queryKey })}
            />
          </section>
        </div>
      )}

      <ChatPanel
        scope="synthetic"
        wellId={wellId}
        title="Synthetic Seismogram Assistant"
        subtitle="Ask about this well's tie, wavelet, and QC flags"
      />
    </div>
  );
}
