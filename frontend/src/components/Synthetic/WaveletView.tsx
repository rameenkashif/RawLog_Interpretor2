import { getSyntheticImageUrl } from "@/api/client";
import type { SyntheticSeismogramResponse } from "@/api/types";

/**
 * Wavelet QC display: time-domain amplitude plus amplitude/phase spectra,
 * so a geophysicist can check an extracted-or-Ricker wavelet's phase
 * behavior before trusting the synthetic it produces. Rendered server-side
 * (Matplotlib) via synthetic_seismogram_service.render_wavelet_image.
 */
export default function WaveletView({ result }: { result: SyntheticSeismogramResponse }) {
  const src = getSyntheticImageUrl(result.well_id, "wavelet", {
    waveletMethod: result.wavelet_method,
    waveletFreqHz: result.wavelet_freq_hz,
    densityMethod: result.density_method,
    autoOptimizeTie: result.auto_optimize_tie,
  });

  return (
    <div className="bg-surface border border-border rounded-xl p-3 shadow-card">
      <img src={src} alt={`${result.well_id} wavelet and spectra`} className="w-full h-auto" />
    </div>
  );
}
