import { getSyntheticImageUrl } from "@/api/client";
import type { SyntheticSeismogramResponse } from "@/api/types";

/**
 * Acoustic impedance (depth domain) and reflectivity (depth + time domain)
 * -- two side-by-side depth tracks like a classic log display, rendered
 * server-side (Matplotlib) via synthetic_seismogram_service.
 * render_impedance_image so every chart on this page comes from the same
 * static-image convention (see WaveletView.tsx / SyntheticTraceOverlay.tsx).
 */
export default function AcousticImpedanceChart({ result }: { result: SyntheticSeismogramResponse }) {
  const src = getSyntheticImageUrl(result.well_id, "impedance", {
    waveletMethod: result.wavelet_method,
    waveletFreqHz: result.wavelet_freq_hz,
    densityMethod: result.density_method,
    autoOptimizeTie: result.auto_optimize_tie,
  });

  return (
    <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
      <img src={src} alt={`${result.well_id} acoustic impedance and reflectivity`} className="w-full h-auto" />
    </div>
  );
}
