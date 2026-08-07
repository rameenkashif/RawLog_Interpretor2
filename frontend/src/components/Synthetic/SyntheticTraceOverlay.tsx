import { useState } from "react";
import { getSyntheticImageUrl } from "@/api/client";
import type { SyntheticSeismogramResponse } from "@/api/types";

type Domain = "time" | "frequency";

/** Synthetic-vs-real trace overlay + tie quality stats, rendered
 * server-side (Matplotlib) via
 * synthetic_seismogram_service.render_trace_overlay_image. Toggles between
 * the time-domain trace overlay and the frequency-domain amplitude
 * spectrum of the same two traces (real_trace / shifted_synthetic) --
 * same underlying convolution result, just two ways to look at it (see
 * synthetic_seismogram_service.py's real_trace_spectrum/synthetic_spectrum,
 * an FFT of the exact arrays plotted in the time-domain view). Each curve
 * is independently RMS-normalized for DISPLAY only on the backend -- the
 * raw, unnormalized values are what's actually returned by the API (CSV
 * export, any future tool) so nothing downstream of this component is
 * affected. */
export default function SyntheticTraceOverlay({ result }: { result: SyntheticSeismogramResponse }) {
  const [domain, setDomain] = useState<Domain>("time");

  const overlaySrc = getSyntheticImageUrl(result.well_id, "trace-overlay", {
    waveletMethod: result.wavelet_method,
    waveletFreqHz: result.wavelet_freq_hz,
    densityMethod: result.density_method,
    autoOptimizeTie: result.auto_optimize_tie,
    domain,
  });
  const sectionSrc = getSyntheticImageUrl(result.well_id, "section", {
    waveletMethod: result.wavelet_method,
    waveletFreqHz: result.wavelet_freq_hz,
    densityMethod: result.density_method,
    autoOptimizeTie: result.auto_optimize_tie,
  });

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-4 text-xs font-semibold text-ink-muted">
          <span>
            Nearest inline/crossline: {result.nearest_inline} / {result.nearest_crossline}
          </span>
          <span>Distance: {result.distance_m?.toFixed(0)} m</span>
          <span>Best shift: {result.best_shift_ms.toFixed(1)} ms</span>
          <span className={result.correlation > 0.5 ? "text-success" : "text-orange-strong"}>
            Correlation: {result.correlation.toFixed(3)}
          </span>
          {result.polarity === -1 && (
            <span className="text-orange-strong">Polarity: reversed</span>
          )}
        </div>

        <div className="flex gap-1.5">
          {(["time", "frequency"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDomain(d)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all uppercase ${
                domain === d
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {result.tie_search_note && (
        <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
          {result.tie_search_note}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 items-start">
        <div className="lg:col-span-2 bg-surface border border-border rounded-xl p-4 shadow-card">
          <img
            src={overlaySrc}
            alt={`${result.well_id} synthetic vs. real trace, ${domain} domain`}
            className="w-full h-auto"
          />
        </div>
        <div className="lg:col-span-3 bg-surface border border-border rounded-xl p-4 shadow-card">
          <img
            src={sectionSrc}
            alt={`${result.well_id} inline section with synthetic overlay`}
            className="w-full h-auto"
          />
        </div>
      </div>
    </div>
  );
}
