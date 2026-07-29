"""
services/section_image_service.py
------------------------------------
Static (Matplotlib) rendering of the Inline/Crossline Section view -- an
ADDITIONAL, clean-looking alternative to the interactive Plotly rendering
already served as JSON by GET /api/seismic/inline|crossline
(SeismicSectionView.tsx). Reuses section_well_log_service.py for the
well-log overlay, so both renderings place the same wells/curves at the
same positions; only the rendering technology and one cosmetic choice
differ (see render_section_image's docstring on per-curve normalization).

Purely additive: the existing JSON section endpoints and their Plotly
frontend are untouched.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless backend -- this process has no display server
import matplotlib.pyplot as plt
import numpy as np

from app.services import section_well_log_service as swl

LANE_KEYS = ("vsh", "phie", "swe")
# Darker variants of the interactive view's colors -- readable against a
# white Matplotlib background instead of the app's dark surface.
LANE_COLORS = {"vsh": "#B45309", "phie": "#047857", "swe": "#BE185D"}
LANE_LABELS = {"vsh": "VSH", "phie": "PHIE", "swe": "SWE"}


def render_section_image(orientation: str, line_number: int) -> bytes:
    """PNG bytes for one inline/crossline amplitude section plus every
    well's VSH/PHIE/SWE, each curve independently normalized to ITS OWN
    min/max here -- unlike the interactive Plotly overlay, which shares
    one 0-1 deflection scale across all three curves, making PHIE (whose
    real range is much narrower than VSH/SWE's 0-1) look nearly flat.
    Per-curve normalization trades away direct cross-curve magnitude
    comparison for every lane actually being readable, which is the more
    useful default for a "does this curve have real structure" read.
    """
    from app.services import seismic_processor as sp

    if orientation not in ("inline", "crossline"):
        raise ValueError(f"orientation must be 'inline' or 'crossline', got {orientation!r}.")

    volume = sp.get_segy_volume()
    if orientation == "inline":
        section = volume.get_inline_section(line_number)
        position_axis = np.array(section["crossline_axis"], dtype=float)
        position_label = "Crossline"
    else:
        section = volume.get_crossline_section(line_number)
        position_axis = np.array(section["inline_axis"], dtype=float)
        position_label = "Inline"

    twt_axis_ms = np.array(section["twt_axis_ms"], dtype=float)
    amplitude = np.array(section["amplitude"], dtype=float)
    max_abs = float(np.abs(amplitude).max()) or 1e-6

    logs = swl.get_section_well_logs(orientation, line_number)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    mesh = ax.pcolormesh(
        position_axis, twt_axis_ms, amplitude, cmap="seismic", vmin=-max_abs, vmax=max_abs, shading="auto"
    )
    ax.invert_yaxis()
    ax.set_xlabel(position_label)
    ax.set_ylabel("Two-Way Time (ms)")
    ax.set_title(f"{orientation.capitalize()} {line_number}")
    fig.colorbar(mesh, ax=ax, label="Amplitude", pad=0.01)

    axis_span = max(float(position_axis.max() - position_axis.min()), 1.0)
    deflection = axis_span * 0.02
    gap = axis_span * 0.028
    lane_offsets = {"vsh": -gap, "phie": 0.0, "swe": gap}
    twt_span = float(twt_axis_ms.max() - twt_axis_ms.min()) or 1.0

    for i, well in enumerate(logs["wells"]):
        x0 = well["position_on_axis"]
        twt = np.array(well["twt_ms"], dtype=float)

        # Vertical well-track marker, labeled -- staggered slightly per
        # well index so labels for closely-spaced wells don't collide.
        ax.axvline(x0, color="0.55", linestyle=":", linewidth=0.8)
        label_y = float(twt.min()) - (i % 3) * 0.035 * twt_span
        ax.text(
            x0, label_y, f"{well['well_id']}\nr={well['correlation']:.2f}",
            fontsize=7, ha="center", va="bottom", color="0.25",
        )

        for key in LANE_KEYS:
            values = np.array([v if v is not None else np.nan for v in well[key]], dtype=float)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            vrange = (vmax - vmin) or 1.0
            normalized = np.where(finite, (values - vmin) / vrange, 0.0)
            baseline = x0 + lane_offsets[key]
            xs = baseline + normalized * deflection
            ax.plot(xs, twt, color=LANE_COLORS[key], linewidth=0.9)
            ax.fill_betweenx(twt, baseline, xs, color=LANE_COLORS[key], alpha=0.35, linewidth=0)

    legend_handles = [
        plt.Line2D([0], [0], color=LANE_COLORS[k], lw=2, label=LANE_LABELS[k]) for k in LANE_KEYS
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
