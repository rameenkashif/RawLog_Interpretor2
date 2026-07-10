import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getSurveyInfo } from "@/api/client";
import SeismicSectionView from "./SeismicSectionView";
import TimeSliceView from "./TimeSliceView";
import WellTieView from "./WellTieView";
import WellZoneTieMapView from "./WellZoneTieMapView";
import AmplitudeSpectrumView from "./AmplitudeSpectrumView";
import SpectralDecompView from "./SpectralDecompView";

const TABS = [
  { id: "section", label: "Inline / Crossline Section" },
  { id: "timeslice", label: "Time Slice" },
  { id: "welltie", label: "Well Tie" },
  { id: "wellzonetiemap", label: "Well-Seismic Tie" },
  { id: "spectrum", label: "Amplitude Spectrum" },
  { id: "spectral", label: "Spectral Decomposition" },
] as const;

type TabId = (typeof TABS)[number]["id"];

/**
 * "Seismic Visualization" feature: tabbed container for the interpretation
 * displays computed directly from the raw SEG-Y volume
 * (app/services/seismic_processor.py) -- inline/crossline sections, time
 * slices, well ties, amplitude spectra, and spectral decomposition.
 * survey-info is fetched once here and passed down so each tab's sliders
 * are bounded by the actual volume geometry rather than hardcoded ranges.
 */
export default function SeismicPanel() {
  const [activeTab, setActiveTab] = useState<TabId>("section");
  const surveyInfoQuery = useQuery({ queryKey: ["seismic-viz-survey-info"], queryFn: getSurveyInfo });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">Seismic Visualization</h2>
          <p className="text-xs text-ink-muted mt-0.5">
            Inline/crossline sections, time slices, well ties, and amplitude spectra read
            directly from the loaded SEG-Y volume.
          </p>
        </div>
        {surveyInfoQuery.data && (
          <span className="text-xs font-semibold text-ink-faint bg-surface-sunken px-3 py-1 rounded-full">
            {surveyInfoQuery.data.source_filename} · {surveyInfoQuery.data.n_traces.toLocaleString()} traces ·{" "}
            {surveyInfoQuery.data.n_inlines}×{surveyInfoQuery.data.n_crosslines} grid
          </span>
        )}
      </div>

      {surveyInfoQuery.isLoading && (
        <div className="h-16 rounded-xl bg-surface-sunken animate-pulse" />
      )}

      {surveyInfoQuery.isError && (
        <div className="border border-red-200 bg-red-50 text-danger text-sm rounded-xl px-4 py-3">
          Seismic volume unavailable: {(surveyInfoQuery.error as Error).message}. Drop a .sgy/.segy
          file into backend/data/seismic_raw/ and reload.
        </div>
      )}

      {surveyInfoQuery.data && (
        <>
          <div className="flex flex-wrap gap-2 border-b border-border pb-3">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`text-xs font-semibold px-3.5 py-1.5 rounded-full border transition-all ${
                  activeTab === tab.id
                    ? "bg-brand-gradient text-white border-transparent shadow-card"
                    : "bg-surface text-ink-muted border-border-strong hover:border-accent hover:text-accent"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div>
            {activeTab === "section" && <SeismicSectionView surveyInfo={surveyInfoQuery.data} />}
            {activeTab === "timeslice" && <TimeSliceView surveyInfo={surveyInfoQuery.data} />}
            {activeTab === "welltie" && <WellTieView />}
            {activeTab === "wellzonetiemap" && <WellZoneTieMapView />}
            {activeTab === "spectrum" && <AmplitudeSpectrumView surveyInfo={surveyInfoQuery.data} />}
            {activeTab === "spectral" && <SpectralDecompView surveyInfo={surveyInfoQuery.data} />}
          </div>
        </>
      )}
    </div>
  );
}
