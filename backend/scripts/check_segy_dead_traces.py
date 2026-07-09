"""
scripts/check_segy_dead_traces.py
-------------------------------------
Follow-up to check_segy_grid_coverage.py: that script confirmed the
(inline, crossline) grid has no duplicate/missing header pairs (every
cell in the 245x252 rectangle has *a* trace). This script checks whether
those traces actually contain real amplitude data, or are zero-filled
"pad" traces -- extremely common in real 3D seismic exports, where the
true acquired survey footprint is an irregular polygon but the SEG-Y
volume is regularized to a rectangular inline x crossline grid, with
every cell outside the real polygon filled by an all-zero dummy trace.

Reports:
  - how many traces are entirely (or near-entirely) zero
  - the inline/crossline bounding box of "live" (non-zero) traces vs.
    the full grid bounding box
  - a coarse ASCII map of live vs. dead cells, so you can see the shape
    of the real survey footprint at a glance

Usage:
    cd backend
    python scripts/check_segy_dead_traces.py path/to/file.sgy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import segyio
import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_segy_dead_traces.py path/to/file.sgy")
        sys.exit(1)

    path = sys.argv[1]
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        inline = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=int)
        crossline = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:], dtype=int)
        traces = f.trace.raw[:]  # (n_traces, n_samples)

    inlines_sorted = np.unique(inline)
    crosslines_sorted = np.unique(crossline)
    n_inlines = len(inlines_sorted)
    n_crosslines = len(crosslines_sorted)

    max_abs_per_trace = np.max(np.abs(traces), axis=1)
    is_dead = max_abs_per_trace == 0.0
    n_dead = int(is_dead.sum())

    print(f"File: {path}")
    print(f"Traces: {n_traces}")
    print(f"Entirely-zero (dead/pad) traces: {n_dead} ({100 * n_dead / n_traces:.1f}%)")
    print(f"Live traces: {n_traces - n_dead} ({100 * (n_traces - n_dead) / n_traces:.1f}%)")
    print()

    if n_dead == 0:
        print(
            "=> No dead traces at all. Every trace has some nonzero amplitude somewhere in "
            "the time axis. The gray region in the Time Slice must be zero (or near-zero) "
            "specifically AT THAT TIME SAMPLE, not a dead trace overall -- check whether the "
            "gray area follows a mute/taper zone (e.g. above first arrival) that moves as you "
            "scrub the time slider, which would point to a top-mute in processing rather than "
            "missing acquisition."
        )
        return

    live_inline = inline[~is_dead]
    live_crossline = crossline[~is_dead]
    print(f"Full grid inline range:     {inline.min()}-{inline.max()}")
    print(f"Live-trace inline range:    {live_inline.min()}-{live_inline.max()}")
    print(f"Full grid crossline range:  {crossline.min()}-{crossline.max()}")
    print(f"Live-trace crossline range: {live_crossline.min()}-{live_crossline.max()}")
    print()

    # Coarse ASCII map (downsample to ~80 columns x ~40 rows) of live vs dead
    # cells, oriented so it roughly matches the Time Slice plot's axes
    # (inline on Y, crossline on X).
    max_cols, max_rows = 80, 40
    col_step = max(1, n_crosslines // max_cols)
    row_step = max(1, n_inlines // max_rows)

    il_pos = np.searchsorted(inlines_sorted, inline)
    xl_pos = np.searchsorted(crosslines_sorted, crossline)
    dead_grid = np.zeros((n_inlines, n_crosslines), dtype=bool)
    dead_grid[il_pos, xl_pos] = is_dead

    print("ASCII map (# = dead/pad trace, . = live trace; inline increases upward):")
    for r in range(n_inlines - 1, -1, -row_step):
        row_slice = dead_grid[r]
        cells = row_slice[::col_step]
        print("".join("#" if c else "." for c in cells))

    print(
        "\n=> If the dead cells form a coherent shape (a corner, a diagonal edge, an L-shape) "
        "rather than random scatter, this confirms the real acquired survey footprint is an "
        "irregular polygon inside the rectangular inline/crossline grid, and the dummy/pad "
        "traces outside it are correctly zero -- not a bug. The fix at that point is a "
        "frontend one: mask exact-zero cells as NaN (transparent/no-data) in the Time Slice "
        "heatmap instead of plotting them as a real amplitude value, so the empty area reads "
        "as 'no data' instead of a flat mid-colorscale gray/white blob."
    )


if __name__ == "__main__":
    main()
