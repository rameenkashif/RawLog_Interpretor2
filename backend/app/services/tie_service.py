"""
services/tie_service.py
------------------------
Orchestrates the well-to-seismic tie: pulls well curves + seismic trace data,
calls well_seismic_tie.py, assembles the API response.

Tie algorithm: each well's own DPTM curve (vendor-precomputed when the LAS
carries one, else petrophysics.compute_dptm's sonic-integration fallback --
see that module) is trusted directly as the time axis, and the tie search
jointly sweeps Ricker wavelet frequency, polarity, and bulk time shift across
the ENTIRE seismic window (well_seismic_tie.search_best_tie_full_window)
rather than a fixed wavelet frequency with a narrow position-only search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from app.well_seismic_tie import (
    BOUNDARY_PINNED_FRACTION,
    TieError,
    find_nearest_trace_index,
    reflectivity_from_time_axis,
    search_best_tie_full_window,
)
from app.models.schemas import (
    SurveyFootprintPoint,
    WellSeismicTieBatchResponse,
    WellSeismicTieResponse,
    WellSeismicTieRow,
)

from app.services import well_service
from app.services import seismic_service
from app.services.seismic_service import SeismicDatasetNotFoundError

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tie_config.yaml"

# How many trace-coordinate points to send back for the map's background
# survey footprint -- a real dataset can have tens of thousands of traces,
# far more than a browser needs to render a footprint scatter.
MAX_FOOTPRINT_POINTS = 1500


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _extract_well_curves(well_id: str) -> dict[str, np.ndarray]:
    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]  # list of {curve_name: value} dicts, one per depth sample

    def _extract(curve_name: str) -> np.ndarray:
        arr = np.array(
            [row.get(curve_name) if row.get(curve_name) is not None else np.nan for row in rows],
            dtype=float,
        )
        arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
        return arr

    return {name: _extract(name) for name in ("DEPT", "DT", "RHOB", "DPTM")}


def _resolve_trace(
    well_id: str,
    well_x: float | None,
    well_y: float | None,
    dataset_id: str,
    traces: np.ndarray,
    trace_x: np.ndarray,
    trace_y: np.ndarray,
    trace_inline: np.ndarray,
    trace_crossline: np.ndarray,
    config: dict,
) -> tuple[int, float | None, str, str | None]:
    """Prefer a real spatial nearest-trace match when both the well (LAS
    header, see las_loader.py) and the seismic dataset (trace headers, see
    segy_loader.py) carry surface coordinates. Falls back to a manually
    configured trace index from tie_config.yaml when coordinates aren't
    available on one side or the other -- this keeps older wells/datasets
    without coordinate headers working, just without a spatial guarantee.

    Returns (trace_idx, distance_m, tie_method, geometry_warning).
    """
    has_well_coords = well_x is not None and well_y is not None
    has_trace_coords = trace_x.size > 0 and np.isfinite(trace_x).any() and np.isfinite(trace_y).any()

    if has_well_coords and has_trace_coords:
        trace_idx, distance_m = find_nearest_trace_index(
            well_x,
            well_y,
            trace_x,
            trace_y,
            max_radius_m=config.get("max_tie_search_radius_m"),
        )
        return trace_idx, distance_m, "nearest_trace", None

    override = config["well_coordinate_overrides"].get(well_id)
    if not override or "trace_index" not in override:
        raise TieError(
            f"No coordinates available for a spatial tie (well {well_id}: "
            f"{'has' if has_well_coords else 'missing'} LAS coordinates, "
            f"dataset {dataset_id}: {'has' if has_trace_coords else 'missing'} "
            "trace coordinates), and no trace_index configured in "
            "tie_config.yaml as a fallback -- add one before requesting a tie."
        )

    trace_idx = int(override["trace_index"])
    if trace_idx >= traces.shape[0]:
        raise TieError(
            f"trace_index {trace_idx} is out of range for dataset {dataset_id} "
            f"({traces.shape[0]} traces available)."
        )
    geometry_warning = (
        "Using a manually configured trace index (tie_config.yaml) -- "
        f"well {well_id} {'has' if has_well_coords else 'has no'} coordinates in "
        f"its LAS header, and the seismic dataset {'has' if has_trace_coords else 'has no'} "
        "stored trace coordinates. This is not a spatial nearest-trace match."
    )
    return trace_idx, None, "manual_override", geometry_warning


def _trace_inline_crossline(trace_inline: np.ndarray, trace_crossline: np.ndarray, trace_idx: int) -> tuple[int | None, int | None]:
    inline = float(trace_inline[trace_idx]) if trace_inline.size > trace_idx else float("nan")
    crossline = float(trace_crossline[trace_idx]) if trace_crossline.size > trace_idx else float("nan")
    return (
        int(inline) if np.isfinite(inline) else None,
        int(crossline) if np.isfinite(crossline) else None,
    )


def _compute_tie(well_id: str, dataset_id: str, freq_hz: float | None = None) -> dict:
    """Shared by get_well_seismic_tie and render_tie_section_image.

    freq_hz: when given, the frequency search is pinned to this single
    candidate (polarity and bulk shift are still optimized around it) --
    a manual override of the normal auto-optimized frequency sweep, for
    a reviewer who wants to see what a specific wavelet frequency's tie
    looks like rather than trusting the automatic best-correlation pick.
    """
    config = _load_config()
    max_shift_ms = float(config.get("tie_search_max_shift_ms", 100.0))

    well_summary = well_service.get_well_summary(well_id)
    curves = _extract_well_curves(well_id)

    # get_seismic_dataset() raises SeismicDatasetNotFoundError itself if the
    # dataset_id doesn't exist -- let it propagate, the router already
    # catches this exception type.
    metadata, traces, twt_axis_ms, trace_x, trace_y, trace_inline, trace_crossline, _attributes_df = (
        seismic_service.get_seismic_dataset(dataset_id)
    )
    seismic_dt_ms = metadata.sample_interval_ms

    trace_idx, distance_m, tie_method, geometry_warning = _resolve_trace(
        well_id,
        well_summary.well_x,
        well_summary.well_y,
        dataset_id,
        traces,
        trace_x,
        trace_y,
        trace_inline,
        trace_crossline,
        config,
    )
    inline, crossline = _trace_inline_crossline(trace_inline, trace_crossline, trace_idx)
    real_trace = traces[trace_idx].astype(float)

    t_rc, rc = reflectivity_from_time_axis(curves["DPTM"], curves["DT"], curves["RHOB"], seismic_dt_ms)
    search_kwargs = {"max_shift_ms": max_shift_ms}
    if freq_hz is not None:
        search_kwargs["candidate_freqs_hz"] = (float(freq_hz),)
    tie = search_best_tie_full_window(t_rc, rc, twt_axis_ms, seismic_dt_ms, real_trace, **search_kwargs)
    boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - BOUNDARY_PINNED_FRACTION) * max_shift_ms

    return dict(
        tie=tie,
        trace_idx=trace_idx,
        inline=inline,
        crossline=crossline,
        distance_m=distance_m,
        tie_method=tie_method,
        geometry_warning=geometry_warning,
        boundary_pinned=boundary_pinned,
        max_shift_ms=max_shift_ms,
        traces=traces,
        twt_axis_ms=twt_axis_ms,
        trace_inline=trace_inline,
        trace_crossline=trace_crossline,
        seismic_dt_ms=seismic_dt_ms,
    )


def get_well_seismic_tie(well_id: str, dataset_id: str, freq_hz: float | None = None) -> WellSeismicTieResponse:
    r = _compute_tie(well_id, dataset_id, freq_hz)
    tie = r["tie"]
    return WellSeismicTieResponse(
        well_id=well_id,
        dataset_id=dataset_id,
        trace_index=r["trace_idx"],
        distance_m=r["distance_m"],
        tie_method=r["tie_method"],
        inline=r["inline"],
        crossline=r["crossline"],
        best_freq_hz=tie.best_freq_hz,
        polarity=tie.polarity,
        bulk_shift_ms=tie.bulk_shift_ms,
        correlation=tie.correlation,
        max_shift_ms=r["max_shift_ms"],
        boundary_pinned=r["boundary_pinned"],
        n_used=tie.n_used,
        time_ms=tie.time_ms.tolist(),
        synthetic_amplitude=tie.synthetic_amplitude.tolist(),
        seismic_amplitude=tie.seismic_amplitude.tolist(),
        reflectivity=tie.reflectivity.tolist(),
        geometry_warning=r["geometry_warning"],
    )


def render_tie_section_image(well_id: str, dataset_id: str, freq_hz: float | None = None) -> bytes:
    """PNG: the Ricker wavelet used for this tie (left) + the inline
    section through the well's own tied trace, with the tie's synthetic
    trace overlaid as a filled wiggle at the well's crossline position
    (right) -- lets a reviewer see how the single-trace synthetic-vs-real
    match (the interactive overlay chart above) sits inside the
    surrounding section, not just at that one trace in isolation.
    freq_hz: same manual-override meaning as get_well_seismic_tie's."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.well_seismic_tie import ricker_wavelet

    r = _compute_tie(well_id, dataset_id, freq_hz)
    tie = r["tie"]
    inline, crossline = r["inline"], r["crossline"]
    if inline is None or crossline is None:
        raise TieError(
            f"Well '{well_id}''s tied trace has no inline/crossline header info -- cannot render a section."
        )

    traces = r["traces"]
    twt_axis_ms = r["twt_axis_ms"]
    trace_inline_f = np.asarray(r["trace_inline"], dtype=float)
    trace_crossline_f = np.asarray(r["trace_crossline"], dtype=float)
    seismic_dt_ms = r["seismic_dt_ms"]

    finite = np.isfinite(trace_inline_f) & np.isfinite(trace_crossline_f)
    mask = finite & (np.round(trace_inline_f) == inline)
    idxs = np.where(mask)[0]
    if len(idxs) < 2:
        raise TieError(f"Only {len(idxs)} trace(s) share inline {inline} -- not enough to render a section.")
    order = np.argsort(trace_crossline_f[idxs])
    idxs = idxs[order]
    xl_full = trace_crossline_f[idxs]

    # Zoom to a small window of traces straddling the well's own trace
    # rather than the whole inline -- a multi-kilometre inline squeezes the
    # handful of traces that actually matter for this tie into an
    # unreadable sliver. Kept tight (a handful of traces either side) so
    # the well's own synthetic wiggle dominates the panel rather than
    # disappearing into a wide section.
    HALF_WINDOW_TRACES = 6
    center_pos = int(np.argmin(np.abs(xl_full - crossline)))
    lo = max(0, center_pos - HALF_WINDOW_TRACES)
    hi = min(len(idxs), center_pos + HALF_WINDOW_TRACES + 1)
    window_idxs = idxs[lo:hi]
    xl_axis = trace_crossline_f[window_idxs]
    amplitude_full = traces[window_idxs].T.astype(float)  # (n_samples, n_traces_in_window)

    # Zoom vertically to the well's own tied interval too (plus a little
    # padding), not the seismic volume's entire recorded time range --
    # a synthetic covering ~100ms of log otherwise gets lost inside a
    # multi-second recording window.
    pad_ms = max(float(tie.time_ms.max() - tie.time_ms.min()) * 0.4, 20.0)
    y_lo = float(tie.time_ms.min()) - pad_ms
    y_hi = float(tie.time_ms.max()) + pad_ms
    time_mask = (twt_axis_ms >= y_lo) & (twt_axis_ms <= y_hi)
    if time_mask.sum() < 2:
        time_mask = np.ones_like(twt_axis_ms, dtype=bool)
        y_lo, y_hi = float(twt_axis_ms.min()), float(twt_axis_ms.max())
    twt_axis_ms = twt_axis_ms[time_mask]
    amplitude = amplitude_full[time_mask, :]
    max_abs = float(np.abs(amplitude).max()) or 1e-6

    _, wavelet = ricker_wavelet(tie.best_freq_hz, seismic_dt_ms / 1000.0)
    wav_t_ms = (np.arange(len(wavelet)) - len(wavelet) // 2) * seismic_dt_ms

    fig, (ax_wav, ax_sec) = plt.subplots(1, 2, figsize=(11, 6), dpi=150, gridspec_kw={"width_ratios": [1, 4]})

    ax_wav.plot(wavelet, wav_t_ms, color="0.2", linewidth=1)
    ax_wav.fill_betweenx(wav_t_ms, 0, wavelet, where=(wavelet >= 0), color="#DC2626", alpha=0.6)
    ax_wav.fill_betweenx(wav_t_ms, 0, wavelet, where=(wavelet < 0), color="#2563EB", alpha=0.6)
    ax_wav.invert_yaxis()
    ax_wav.axvline(0, color="0.6", linewidth=0.6)
    ax_wav.set_ylabel("Time (ms)")
    ax_wav.set_xlabel("Amplitude")
    ax_wav.set_title(f"Ricker {tie.best_freq_hz:.0f}Hz")

    mesh = ax_sec.pcolormesh(
        xl_axis, twt_axis_ms, amplitude, cmap="seismic", vmin=-max_abs, vmax=max_abs, shading="auto"
    )
    ax_sec.set_ylim(y_hi, y_lo)  # bottom=later time, top=earlier -- axis reads top-down like the wavelet panel
    ax_sec.set_xlabel("Crossline")
    ax_sec.set_title(f"Inline {inline} -- synthetic vs. seismic at {well_id}")
    fig.colorbar(mesh, ax=ax_sec, label="Amplitude", pad=0.01)

    # Synthetic overlay drawn the same way as the wavelet panel -- a
    # red/blue filled wiggle straddling the well's trace position, not a
    # solid silhouette -- so it reads as a seismic trace, not a blob.
    syn = np.asarray(tie.synthetic_amplitude, dtype=float)
    syn_norm = syn / (np.abs(syn).max() or 1e-6)
    trace_spacing = float(np.median(np.abs(np.diff(xl_axis)))) if len(xl_axis) > 1 else 1.0
    deflection = trace_spacing * 3.0
    xs = crossline + syn_norm * deflection
    ax_sec.fill_betweenx(
        tie.time_ms, crossline, xs, where=(syn_norm >= 0), color="#DC2626", alpha=0.8, linewidth=0, interpolate=True
    )
    ax_sec.fill_betweenx(
        tie.time_ms, crossline, xs, where=(syn_norm < 0), color="#2563EB", alpha=0.8, linewidth=0, interpolate=True
    )
    ax_sec.plot(xs, tie.time_ms, color="black", linewidth=0.7)
    ax_sec.axvline(crossline, color="0.3", linestyle=":", linewidth=0.7)
    ax_sec.text(crossline, float(twt_axis_ms.min()), f" {well_id}", fontsize=8, va="bottom")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def get_all_well_ties(dataset_id: str) -> WellSeismicTieBatchResponse:
    """Batch tie: run get_well_seismic_tie's same algorithm for every well in
    the repository against one seismic dataset, for a results table + map
    (mirrors the notebook's per-well loop + results DataFrame). Wells that
    fail (missing curves, no coordinates, etc.) get a row with `error` set
    rather than being silently dropped -- so a partial batch is still
    visible as partial, not indistinguishable from "only these wells exist".
    """
    config = _load_config()
    max_shift_ms = float(config.get("tie_search_max_shift_ms", 100.0))

    metadata, traces, twt_axis_ms, trace_x, trace_y, trace_inline, trace_crossline, _attributes_df = (
        seismic_service.get_seismic_dataset(dataset_id)
    )
    seismic_dt_ms = metadata.sample_interval_ms

    rows: list[WellSeismicTieRow] = []
    warnings: list[str] = []

    for well_summary in well_service.list_well_summaries():
        well_id = well_summary.well_id
        try:
            curves = _extract_well_curves(well_id)
            trace_idx, distance_m, tie_method, geometry_warning = _resolve_trace(
                well_id,
                well_summary.well_x,
                well_summary.well_y,
                dataset_id,
                traces,
                trace_x,
                trace_y,
                trace_inline,
                trace_crossline,
                config,
            )
            if geometry_warning:
                warnings.append(f"{well_id}: {geometry_warning}")
            inline, crossline = _trace_inline_crossline(trace_inline, trace_crossline, trace_idx)
            real_trace = traces[trace_idx].astype(float)
            t_rc, rc = reflectivity_from_time_axis(
                curves["DPTM"], curves["DT"], curves["RHOB"], seismic_dt_ms
            )
            tie = search_best_tie_full_window(
                t_rc, rc, twt_axis_ms, seismic_dt_ms, real_trace, max_shift_ms=max_shift_ms
            )
            boundary_pinned = abs(tie.bulk_shift_ms) >= (1.0 - BOUNDARY_PINNED_FRACTION) * max_shift_ms

            rows.append(
                WellSeismicTieRow(
                    well_id=well_id,
                    well_x=well_summary.well_x,
                    well_y=well_summary.well_y,
                    trace_index=trace_idx,
                    trace_x=float(trace_x[trace_idx]) if trace_x.size > trace_idx and np.isfinite(trace_x[trace_idx]) else None,
                    trace_y=float(trace_y[trace_idx]) if trace_y.size > trace_idx and np.isfinite(trace_y[trace_idx]) else None,
                    inline=inline,
                    crossline=crossline,
                    distance_m=distance_m,
                    tie_method=tie_method,
                    best_freq_hz=tie.best_freq_hz,
                    polarity=tie.polarity,
                    bulk_shift_ms=tie.bulk_shift_ms,
                    correlation=tie.correlation,
                    boundary_pinned=boundary_pinned,
                )
            )
        except (TieError, well_service.WellNotFoundError) as exc:
            rows.append(
                WellSeismicTieRow(
                    well_id=well_id,
                    well_x=well_summary.well_x,
                    well_y=well_summary.well_y,
                    error=str(exc),
                )
            )
            warnings.append(f"{well_id}: {exc}")

    survey_footprint: list[SurveyFootprintPoint] = []
    finite = np.isfinite(trace_x) & np.isfinite(trace_y)
    if finite.any():
        xs, ys = trace_x[finite], trace_y[finite]
        step = max(1, len(xs) // MAX_FOOTPRINT_POINTS)
        survey_footprint = [
            SurveyFootprintPoint(x=float(x), y=float(y)) for x, y in zip(xs[::step], ys[::step])
        ]

    return WellSeismicTieBatchResponse(
        dataset_id=dataset_id, rows=rows, survey_footprint=survey_footprint, warnings=warnings
    )
