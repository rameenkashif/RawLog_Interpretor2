import type { ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SyntheticSeismogramResponse } from "@/api/types";
import { useChartColors } from "@/styles/tokens";

/**
 * Wavelet QC display: time-domain amplitude plus amplitude/phase spectra,
 * so a geophysicist can check an extracted-or-Ricker wavelet's phase
 * behavior before trusting the synthetic it produces.
 */
export default function WaveletView({ result }: { result: SyntheticSeismogramResponse }) {
  const colors = useChartColors();
  const waveletData = result.wavelet_t_ms.map((t, i) => ({ t_ms: t, amplitude: result.wavelet_amplitude[i] }));
  const spectrumData = result.wavelet_spectrum_freq_hz.map((f, i) => ({
    freq_hz: f,
    amplitude: result.wavelet_spectrum_amplitude[i],
    phase_deg: result.wavelet_spectrum_phase_deg[i],
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <ChartCard title={`Wavelet (${result.wavelet_method})`}>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={waveletData}>
            <CartesianGrid stroke={colors.gridLine} />
            <XAxis dataKey="t_ms" stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 10 }} />
            <YAxis stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }} />
            <Line type="monotone" dataKey="amplitude" stroke={colors.accent} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Amplitude Spectrum">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={spectrumData}>
            <CartesianGrid stroke={colors.gridLine} />
            <XAxis dataKey="freq_hz" stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 10 }} />
            <YAxis stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }} />
            <Line type="monotone" dataKey="amplitude" stroke={colors.accent} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Phase Spectrum">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={spectrumData}>
            <CartesianGrid stroke={colors.gridLine} />
            <XAxis dataKey="freq_hz" stroke={colors.borderStrong} tick={{ fill: colors.inkMuted, fontSize: 10 }} />
            <YAxis
              stroke={colors.borderStrong}
              tick={{ fill: colors.inkMuted, fontSize: 10 }}
              label={{ value: "deg", angle: -90, position: "insideLeft", fill: colors.inkMuted, fontSize: 10 }}
            />
            <Tooltip contentStyle={{ backgroundColor: colors.surface, border: `1px solid ${colors.border}` }} />
            <Line type="monotone" dataKey="phase_deg" stroke={colors.orange} dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="bg-surface border border-border rounded-xl p-3 shadow-card">
      <h4 className="text-xs font-semibold text-ink-muted mb-1">{title}</h4>
      {children}
    </div>
  );
}
