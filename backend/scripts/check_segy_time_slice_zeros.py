"""
scripts/check_segy_time_slice_zeros.py
-------------------------------------------
Follow-up to check_segy_dead_traces.py: that script confirmed no trace is
ENTIRELY zero (100% live traces), which rules out whole-trace pad/dummy
traces. But the Time Slice gray region is at a SINGLE time sample -- a
trace can be zero at one sample (e.g. inside a top-mute/inner-trace-mute
zone applied in processing) while having real amplitude everywhere else.
This script reproduces SegyVolume.get_time_slice()'s exact nearest-sample
lookup for a given time and reports how many traces are zero AT THAT
SAMPLE specifically, plus a spatial map of which cells are zero.

Usage:
    cd backend
    python scripts/check_segy_time_slice_zeros.py path/to/file.sgy <time_ms>

Example (use whatever time value produced the gray screenshot):
    python scripts/check_segy_time_slice_zeros.py data/seismic_raw/origional.segy 800
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import segyio
import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/check_segy_time_slice_zeros.py path/to/file.sgy <time_ms>")
        sys.exit(1)

    path = sys.argv[1]
    time_ms = float(sys.argv[2])

    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        twt_axis_ms = np.array(f.samples, dtype=float)
        inline = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=int)
        crossline = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:], dtype=int)
        traces = f.trace.raw[:]  # (n_traces, n_samples)

    sample_idx = int(np.argmin(np.abs(twt_axis_ms - time_ms)))
    actual_time_ms = float(twt_axis_ms[sample_idx])
    print(f"File: {path}")
    print(f"Requested time: {time_ms} ms -> nearest sample: index {sample_idx}, actual {actual_time_ms} ms")
    print(f"Full time axis: {twt_axis_ms[0]}-{twt_axis_ms[-1]} ms, {len(twt_axis_ms)} samples")
    print()

    slice_values = traces[:, sample_idx]
    is_zero = slice_values == 0.0
    n_zero = int(is_zero.sum())
    print(f"Traces with exactly zero amplitude AT THIS SAMPLE: {n_zero}/{n_traces} ({100 * n_zero / n_traces:.1f}%)")

    if n_zero == 0:
        print("\n=> No traces are zero at this exact time sample. The gray region isn't literal "
              "zero-valued data at this time -- double check the time value used here matches "
              "what the frontend actually requested (open browser devtools Network tab and look "
              "at the /api/seismic/.../time-slice request's time_ms query param).")
        return

    inlines_sorted = np.unique(inline)
    crosslines_sorted = np.unique(crossline)
    n_inlines, n_crosslines = len(inlines_sorted), len(crosslines_sorted)
    il_pos = np.searchsorted(inlines_sorted, inline)
    xl_pos = np.searchsorted(crosslines_sorted, crossline)

    zero_grid = np.zeros((n_inlines, n_crosslines), dtype=bool)
    zero_grid[il_pos, xl_pos] = is_zero

    max_cols, max_rows = 80, 40
    col_step = max(1, n_crosslines // max_cols)
    row_step = max(1, n_inlines // max_rows)

    print("\nASCII map (# = zero at this time sample, . = nonzero; inline increases upward):")
    for r in range(n_inlines - 1, -1, -row_step):
        cells = zero_grid[r][::col_step]
        print("".join("#" if c else "." for c in cells))

    # Also check whether the same traces are zero across a broader window
    # around this sample, to distinguish "mute front passing through here"
    # from "isolated single-sample zero".
    lo = max(0, sample_idx - 5)
    hi = min(traces.shape[1], sample_idx + 6)
    window_all_zero = np.all(traces[:, lo:hi] == 0.0, axis=1)
    print(
        f"\nOf those {n_zero} zero traces, {int((window_all_zero & is_zero).sum())} are ALSO zero "
        f"for the whole +/-5 sample window around this time -- consistent with a mute zone "
        f"(a contiguous zeroed span) rather than a single stray zero sample."
    )

    print(
        "\n=> If the ASCII map shows a coherent front/boundary (not random speckle), this is a "
        "mute zone (top mute / inner trace mute / survey-edge taper applied in processing) that "
        "zeros the shallow part of some traces while leaving deeper samples real. This is normal "
        "seismic processing, not a bug in this app. Try a deeper time_ms value (larger number = "
        "later arrival, likely below the mute front) and re-run this script -- if the zero "
        "fraction drops sharply at greater depth/time, that confirms it. The application-level "
        "fix is still the same: mask exact zero as NaN in the Time Slice heatmap so mute zones "
        "render as transparent 'no data' instead of a flat colored blob."
    )


if __name__ == "__main__":
    main()
