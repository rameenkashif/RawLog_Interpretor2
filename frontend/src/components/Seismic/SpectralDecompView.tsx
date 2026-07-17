import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { AxiosError } from "axios";
import { getSpectralDecompositionTrace, getSpectralFrequencySlice, getSpectralSwtSlice } from "@/api/client";
import type {
  SpectralFrequencySliceResponse,
  SpectralMethod,
  SpectralSwtSliceResponse,
  SurveyInfoResponse,
  SwtWavelet,
} from "@/api/types";
import { useChartColors, type ChartColors } from "@/styles/tokens";
import TraceScalogramView from "./TraceScalogramView";

function axisStyle(colors: ChartColors) {
  return {
    gridcolor: colors.gridLine,
    linecolor: colors.borderStrong,
    tickfont: { color: colors.inkMuted },
  };
}

const DEFAULT_BAND_MIN_HZ = 5;
const DEFAULT_BAND_MAX_HZ = 80;
const SLIDER_DEBOUNCE_MS = 180;

const SWT_MIN_LEVEL = 1;
const SWT_MAX_LEVEL = 6;
const SWT_DEFAULT_LEVEL = 3;
const SWT_WAVELETS: { value: SwtWavelet; label: string }[] = [
  { value: "sym8", label: "Symlet (sym8)" },
  { value: "coif3", label: "Coiflet (coif3)" },
];

/**
 * Spectral decomposition: energy at a single, user-selected frequency (or,
 * for SWT, decomposition level) across an inline section, instead of a
 * flat per-trace amplitude or a single averaged spectrum (see
 * AmplitudeSpectrumView) -- this is what surfaces thin-bed tuning and
 * stratigraphic features (channels, thin reservoir layers) that a plain
 * amplitude section or flat spectrum can't show, since tuning shows up as
 * brightening at specific frequencies at specific times, not as a change
 * in overall amplitude.
 *
 * Reuses the same Plotly heatmap pattern as SeismicSectionView -- same
 * data shape convention (time x position), just colored by energy at the
 * selected frequency/level instead of raw amplitude. STFT/CWT expose a
 * continuous frequency slider defaulting to the typical useful seismic
 * band (5-80 Hz); SWT instead exposes a discrete Level (1-6) slider plus a
 * wavelet-family picker, since a stationary wavelet transform decomposes
 * into a fixed set of dyadic (octave) bands rather than a continuous
 * frequency axis -- see seismic_processor.py's _decompose_swt. Dragging
 * either control re-fetches the single-slice amplitude (fast -- see
 * seismic_processor.py's per-inline cache) debounced so it doesn't fire on
 * every pixel of movement.
 */
export default function SpectralDecompView({ surveyInfo }: { surveyInfo: SurveyInfoResponse }) {
  const colors = useChartColors();
  const [inlineNumber, setInlineNumber] = useState(surveyInfo.inline_min);
  const [method, setMethod] = useState<SpectralMethod>("stft");
  const [frequencyHz, setFrequencyHz] = useState(
    Math.min(30, surveyInfo.sample_interval_ms > 0 ? 1000 / (2 * surveyInfo.sample_interval_ms) : 30),
  );
  const [debouncedFrequencyHz, setDebouncedFrequencyHz] = useState(frequencyHz);
  const [level, setLevel] = useState(SWT_DEFAULT_LEVEL);
  const [debouncedLevel, setDebouncedLevel] = useState(level);
  const [wavelet, setWavelet] = useState<SwtWavelet>("sym8");

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedFrequencyHz(frequencyHz), SLIDER_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [frequencyHz]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedLevel(level), SLIDER_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [level]);

  const nyquistHz = surveyInfo.sample_interval_ms > 0 ? 1000 / (2 * surveyInfo.sample_interval_ms) : 250;
  const isSwt = method === "swt";

  const sliceQuery = useQuery<SpectralFrequencySliceResponse | SpectralSwtSliceResponse>({
    queryKey: isSwt
      ? ["seismic-viz-spectral-swt-slice", inlineNumber, debouncedLevel, wavelet]
      : ["seismic-viz-spectral-slice", inlineNumber, method, debouncedFrequencyHz],
    queryFn: () =>
      isSwt
        ? getSpectralSwtSlice(inlineNumber, debouncedLevel, wavelet)
        : getSpectralFrequencySlice(inlineNumber, method, debouncedFrequencyHz),
  });

  const figure = useMemo(() => {
    if (!sliceQuery.data) return null;
    return buildFigure(sliceQuery.data.crossline_axis, sliceQuery.data.time_ms, sliceQuery.data.amplitude, colors);
  }, [sliceQuery.data, colors]);

  const swtData = isSwt && sliceQuery.data && "band_hz" in sliceQuery.data ? sliceQuery.data : null;

  return (
    <div className="space-y-3">
      <div className="border border-accent/30 bg-accent-soft/40 text-accent-strong text-xs rounded-xl px-4 py-2.5 leading-relaxed">
        {isSwt ? (
          <>
            SWT highlights shift-invariant discontinuities and edges in the trace — useful for
            pinpointing thin-bed boundaries directly, as a contrast to the frequency-tuning view
            shown by STFT/CWT.
          </>
        ) : (
          <>
            Energy at the selected frequency, across the section — thin layers tend to show up as
            tuning/brightening at specific frequencies rather than as a change in overall
            amplitude. Compare this against the plain amplitude section to spot features a flat
            view hides.
          </>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(["stft", "cwt", "swt"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMethod(m)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-all uppercase ${
                method === m
                  ? "bg-brand-gradient text-white border-transparent shadow-card"
                  : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {isSwt && (
          <select
            value={wavelet}
            onChange={(e) => setWavelet(e.target.value as SwtWavelet)}
            className="text-xs font-semibold border border-border-strong rounded-lg px-2.5 py-1.5 text-ink-muted"
          >
            {SWT_WAVELETS.map((w) => (
              <option key={w.value} value={w.value}>
                {w.label}
              </option>
            ))}
          </select>
        )}

        <label className="flex items-center gap-2 text-xs font-semibold text-ink-muted">
          Inline
          <input
            type="number"
            min={surveyInfo.inline_min}
            max={surveyInfo.inline_max}
            value={inlineNumber}
            onChange={(e) => setInlineNumber(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
          <span className="text-ink-faint font-normal">
            ({surveyInfo.inline_min}-{surveyInfo.inline_max})
          </span>
        </label>
      </div>

      {isSwt ? (
        <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
          Level
          <input
            type="range"
            min={SWT_MIN_LEVEL}
            max={SWT_MAX_LEVEL}
            step={1}
            value={level}
            onChange={(e) => setLevel(Number(e.target.value))}
            className="w-56 accent-accent"
          />
          <input
            type="number"
            min={SWT_MIN_LEVEL}
            max={SWT_MAX_LEVEL}
            step={1}
            value={level}
            onChange={(e) => setLevel(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
          <span className="text-ink-faint font-normal">
            {swtData
              ? `Level ${swtData.level} ≈ ${swtData.band_hz[0].toFixed(0)}-${swtData.band_hz[1].toFixed(0)} Hz`
              : `levels ${SWT_MIN_LEVEL}-${SWT_MAX_LEVEL}, Nyquist ${nyquistHz.toFixed(0)} Hz`}
          </span>
        </label>
      ) : (
        <label className="flex flex-wrap items-center gap-3 text-xs font-semibold text-ink-muted">
          Frequency (Hz)
          <input
            type="range"
            min={0}
            max={nyquistHz}
            step={0.5}
            value={frequencyHz}
            onChange={(e) => setFrequencyHz(Number(e.target.value))}
            className="w-56 accent-accent"
          />
          <input
            type="number"
            min={0}
            max={nyquistHz}
            step={0.5}
            value={frequencyHz}
            onChange={(e) => setFrequencyHz(Number(e.target.value))}
            className="w-20 text-xs border border-border-strong rounded-lg px-2 py-1"
          />
          <span className="text-ink-faint font-normal">
            typical band {DEFAULT_BAND_MIN_HZ}-{DEFAULT_BAND_MAX_HZ} Hz, Nyquist {nyquistHz.toFixed(0)} Hz
          </span>
          {sliceQuery.data && "frequency_hz" in sliceQuery.data && (
            <span className="text-ink-faint font-normal">
              actual bin: {sliceQuery.data.frequency_hz.toFixed(1)} Hz
            </span>
          )}
        </label>
      )}

      {sliceQuery.isLoading && <div className="h-[480px] rounded-xl bg-surface-sunken animate-pulse" />}
      {sliceQuery.isError && (
        <div className="border border-danger/30 bg-danger-soft text-danger text-sm rounded-xl px-4 py-3">
          Failed to load spectral decomposition: {errorMessage(sliceQuery.error)}
        </div>
      )}
      {figure && (
        <div className="bg-surface border border-border rounded-xl p-2 shadow-card">
          <Plot
            data={figure.data}
            layout={figure.layout}
            style={{ width: "100%", height: "480px" }}
            config={{ displaylogo: false, responsive: true }}
          />
        </div>
      )}

      {method === "cwt" && <TraceScalogramView surveyInfo={surveyInfo} />}
    </div>
  );
}

function errorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

/**
 * Shared heatmap builder for every (n_time, n_x) energy/amplitude map in
 * this view: the inline-section slices (x=crossline) above, and the
 * single-trace scalogram (x=frequency) below -- same data shape
 * convention (z is time-major, matching Plotly's z[row]=y-axis
 * requirement when y=timeMs), just a different x-axis and its label.
 * Exported so other trace-level views (e.g. a future well-tie scalogram)
 * can reuse it instead of duplicating the heatmap styling.
 */
export function buildFigure(
  xAxis: number[],
  timeMs: number[],
  energy: number[][], // (n_time, n_x)
  colors: ChartColors,
  xAxisLabel: string = "Crossline",
): { data: Data[]; layout: Partial<Layout> } {
  const AXIS_STYLE = axisStyle(colors);
  let maxVal = 1e-6;
  for (const row of energy) {
    for (const value of row) {
      if (value > maxVal) maxVal = value;
    }
  }

  const trace = {
    type: "heatmap",
    x: xAxis,
    y: timeMs,
    z: energy,
    zmin: 0,
    zmax: maxVal,
    colorscale: "Viridis",
    colorbar: { title: { text: "Energy", font: { size: 10 } }, tickfont: { size: 9 } },
  } as Data;

  const layout: Partial<Layout> = {
    paper_bgcolor: colors.surface,
    plot_bgcolor: colors.surface,
    font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    xaxis: { title: { text: xAxisLabel }, ...AXIS_STYLE },
    yaxis: { title: { text: "Two-Way Time (ms)" }, autorange: "reversed", ...AXIS_STYLE },
  };

  return { data: [trace], layout };
}
