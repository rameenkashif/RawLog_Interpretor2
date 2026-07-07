import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { getCrossplot } from "@/api/client";
import { colors, zoneColors, zoneLabels } from "@/styles/tokens";

interface CrossplotBuilderProps {
  wellId: string;
  curveNames: string[];
}

interface Preset {
  label: string;
  x: string;
  y: string;
  color: string | null;
  logX?: boolean;
  logY?: boolean;
  reverseY?: boolean;
  description: string;
}

const PRESETS: Preset[] = [
  {
    label: "Neutron-Density",
    x: "NPHI",
    y: "RHOB",
    color: "VSH",
    description: "Classic lithology/porosity crossplot, colored by shale volume.",
  },
  {
    label: "Pickett Plot",
    x: "RESISTIVITY",
    y: "PHIE",
    color: "ZONES",
    logX: true,
    logY: true,
    description: "log(Rt) vs log(PHIE) for Archie Sw QC -- points trending toward the origin along an Sw=const line indicate consistent saturation.",
  },
  {
    label: "PHIE vs PERM_TIXIER",
    x: "PHIE",
    y: "PERM_TIXIER",
    color: "ZONES",
    logY: true,
    description: "Porosity-permeability transform (Tixier). Log-scale permeability axis.",
  },
  {
    label: "VSH vs Depth",
    x: "VSH",
    y: "DEPT",
    color: "ZONES",
    reverseY: true,
    description: "Shale volume trend with depth.",
  },
  {
    label: "PHIE vs Depth",
    x: "PHIE",
    y: "DEPT",
    color: "ZONES",
    reverseY: true,
    description: "Effective porosity trend with depth.",
  },
];

/**
 * Generic "pick any two curves + optional color-by third curve" crossplot
 * builder, pre-loaded with the minimum required preset set from section 7.
 * Backed by GET /wells/{well_id}/crossplot so the same endpoint powers both
 * the presets and free-form exploration.
 */
export default function CrossplotBuilder({ wellId, curveNames }: CrossplotBuilderProps) {
  const [xCurve, setXCurve] = useState("NPHI");
  const [yCurve, setYCurve] = useState("RHOB");
  const [colorCurve, setColorCurve] = useState<string | null>("VSH");
  const [logX, setLogX] = useState(false);
  const [logY, setLogY] = useState(false);
  const [reverseY, setReverseY] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>("Neutron-Density");

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["crossplot", wellId, xCurve, yCurve, colorCurve],
    queryFn: () => getCrossplot(wellId, xCurve, yCurve, colorCurve),
    enabled: Boolean(xCurve && yCurve),
  });

  function applyPreset(preset: Preset) {
    setXCurve(preset.x);
    setYCurve(preset.y);
    setColorCurve(preset.color);
    setLogX(Boolean(preset.logX));
    setLogY(Boolean(preset.logY));
    setReverseY(Boolean(preset.reverseY));
    setActivePreset(preset.label);
  }

  const { trace, layout } = useMemo(() => {
    if (!data) return { trace: null, layout: null };

    const isCategoricalColor = colorCurve === "ZONES";
    const points = data.points;

    let marker: Partial<Data>["marker"];
    if (isCategoricalColor) {
      marker = {
        color: points.map((p) => (typeof p.color === "number" ? zoneColors[p.color] ?? colors.inkFaint : colors.inkFaint)),
        size: 5,
        line: { width: 0 },
      };
    } else if (colorCurve) {
      marker = {
        color: points.map((p) => (typeof p.color === "number" ? p.color : null)),
        colorscale: "Viridis",
        showscale: true,
        colorbar: { title: colorCurve, titlefont: { size: 10 }, tickfont: { size: 9 } },
        size: 5,
        line: { width: 0 },
      };
    } else {
      marker = { color: colors.accent, size: 5, line: { width: 0 } };
    }

    const trace: Data = {
      type: "scattergl",
      mode: "markers",
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      marker,
      text: points.map((p) => `Depth: ${p.depth.toFixed(1)} m`),
      hovertemplate: `${xCurve}: %{x}<br>${yCurve}: %{y}<br>%{text}<extra></extra>`,
    };

    const layout: Partial<Layout> = {
      paper_bgcolor: colors.surface,
      plot_bgcolor: colors.surface,
      font: { color: colors.ink, family: "Inter, system-ui, sans-serif" },
      margin: { t: 20, r: 20, b: 50, l: 60 },
      xaxis: {
        title: xCurve,
        type: logX ? "log" : "linear",
        gridcolor: colors.gridLine,
        linecolor: colors.borderStrong,
        zerolinecolor: colors.border,
        tickfont: { color: colors.inkMuted },
      },
      yaxis: {
        title: yCurve,
        type: logY ? "log" : "linear",
        autorange: reverseY ? "reversed" : true,
        gridcolor: colors.gridLine,
        linecolor: colors.borderStrong,
        zerolinecolor: colors.border,
        tickfont: { color: colors.inkMuted },
      },
      showlegend: false,
    };

    return { trace, layout };
  }, [data, colorCurve, xCurve, yCurve, logX, logY, reverseY]);

  return (
    <div className="bg-surface border border-border rounded-lg p-4 shadow-sm space-y-4">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => applyPreset(p)}
            className={`text-xs font-medium px-3 py-1.5 rounded-full border transition-colors ${
              activePreset === p.label
                ? "bg-accent text-white border-accent"
                : "bg-surface text-ink-muted border-border-strong hover:bg-surface-sunken"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Select label="X axis" value={xCurve} onChange={(v) => { setXCurve(v); setActivePreset(null); }} options={curveNames} />
        <Select label="Y axis" value={yCurve} onChange={(v) => { setYCurve(v); setActivePreset(null); }} options={curveNames} />
        <Select
          label="Color by"
          value={colorCurve ?? ""}
          onChange={(v) => { setColorCurve(v || null); setActivePreset(null); }}
          options={["", ...curveNames]}
        />
        <Toggle label="Log X" checked={logX} onChange={setLogX} />
        <Toggle label="Log Y" checked={logY} onChange={setLogY} />
      </div>

      {activePreset && (
        <p className="text-xs text-ink-faint -mt-2">
          {PRESETS.find((p) => p.label === activePreset)?.description}
        </p>
      )}

      {isLoading && <div className="h-[420px] rounded-md bg-surface-sunken animate-pulse" />}
      {isError && (
        <div className="text-sm text-danger">Failed to load crossplot: {(error as Error).message}</div>
      )}

      {trace && layout && (
        <Plot
          data={[trace]}
          layout={layout}
          style={{ width: "100%", height: "420px" }}
          config={{ displaylogo: false, responsive: true }}
        />
      )}

      {colorCurve === "ZONES" && (
        <div className="flex gap-4 text-xs text-ink-muted">
          {Object.entries(zoneLabels).map(([code, label]) => (
            <span key={code} className="flex items-center gap-1.5">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: zoneColors[Number(code)] }}
              />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="text-xs text-ink-muted flex flex-col gap-1">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border-strong bg-surface px-2 py-1.5 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/40"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o || "(none)"}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="text-xs text-ink-muted flex flex-col gap-1 justify-end pb-1.5">
      <span className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="rounded border-border-strong text-accent focus:ring-accent/40"
        />
        {label}
      </span>
    </label>
  );
}
