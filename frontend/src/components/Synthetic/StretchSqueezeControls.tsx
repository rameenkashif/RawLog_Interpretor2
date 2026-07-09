import { useEffect, useState } from "react";
import { deleteSyntheticTiePoints, saveSyntheticTiePoints } from "@/api/client";
import type { TiePointModel, WaveletMethod } from "@/api/types";

/**
 * Manual stretch/squeeze controls: since no real checkshot exists to
 * calibrate the sonic-integration time-depth curve, a user can nudge it by
 * picking MD -> time-shift control points. Persisted per well via PUT/DELETE
 * /api/synthetic/{well_id}/tie so adjustments survive across sessions
 * instead of being recomputed from scratch (see synthetic_tie_repository.py).
 *
 * A numeric control-point table rather than true drag-on-chart interaction
 * -- functionally equivalent (add/edit/remove MD+shift pairs, apply,
 * persist) without the custom drag-handling a chart-based picker would need.
 */
export default function StretchSqueezeControls({
  wellId,
  appliedPoints,
  waveletMethod,
  waveletFreqHz,
  onApplied,
}: {
  wellId: string;
  appliedPoints: TiePointModel[];
  waveletMethod: WaveletMethod;
  waveletFreqHz: number;
  onApplied: () => void;
}) {
  const [points, setPoints] = useState<TiePointModel[]>(appliedPoints);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPoints(appliedPoints);
  }, [wellId, appliedPoints]);

  function updatePoint(index: number, field: keyof TiePointModel, value: number) {
    setPoints((prev) => prev.map((p, i) => (i === index ? { ...p, [field]: value } : p)));
  }

  function addPoint() {
    setPoints((prev) => [...prev, { md_m: 0, time_shift_ms: 0 }]);
  }

  function removePoint(index: number) {
    setPoints((prev) => prev.filter((_, i) => i !== index));
  }

  async function applyAndSave() {
    setSaving(true);
    setError(null);
    try {
      await saveSyntheticTiePoints(wellId, {
        points,
        wavelet_method: waveletMethod,
        wavelet_freq_hz: waveletFreqHz,
      });
      onApplied();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save tie points");
    } finally {
      setSaving(false);
    }
  }

  async function clearAll() {
    setSaving(true);
    setError(null);
    try {
      await deleteSyntheticTiePoints(wellId);
      setPoints([]);
      onApplied();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear tie points");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="bg-surface border border-border rounded-xl p-4 shadow-card space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-ink">Manual Stretch / Squeeze</h4>
        <span className="text-xs text-ink-faint">
          Control points nudge the sonic-integration time-depth curve (MD → time shift)
        </span>
      </div>

      <div className="space-y-2">
        {points.length === 0 && (
          <p className="text-xs text-ink-faint">No manual tie points -- using the sonic-integration curve as-is.</p>
        )}
        {points.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <label className="text-xs text-ink-muted flex items-center gap-1">
              MD (m)
              <input
                type="number"
                value={p.md_m}
                onChange={(e) => updatePoint(i, "md_m", Number(e.target.value))}
                className="w-28 text-xs border border-border-strong rounded-lg px-2 py-1"
              />
            </label>
            <label className="text-xs text-ink-muted flex items-center gap-1">
              Time shift (ms)
              <input
                type="number"
                value={p.time_shift_ms}
                onChange={(e) => updatePoint(i, "time_shift_ms", Number(e.target.value))}
                className="w-24 text-xs border border-border-strong rounded-lg px-2 py-1"
              />
            </label>
            <button
              onClick={() => removePoint(i)}
              className="text-xs font-semibold text-danger hover:underline"
            >
              Remove
            </button>
          </div>
        ))}
      </div>

      {error && <p className="text-xs text-danger">{error}</p>}

      <div className="flex gap-2">
        <button
          onClick={addPoint}
          className="text-xs font-semibold px-3 py-1.5 rounded-full border border-border-strong text-ink-muted hover:border-accent hover:text-accent transition-colors"
        >
          + Add control point
        </button>
        <button
          onClick={applyAndSave}
          disabled={saving}
          className="text-xs font-semibold px-3.5 py-1.5 rounded-full bg-brand-gradient text-white shadow-card disabled:opacity-50"
        >
          {saving ? "Applying…" : "Apply & Save"}
        </button>
        <button
          onClick={clearAll}
          disabled={saving}
          className="text-xs font-semibold px-3.5 py-1.5 rounded-full border border-border-strong text-ink-muted hover:border-danger hover:text-danger transition-colors disabled:opacity-50"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
