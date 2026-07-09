"""
scripts/check_segy_mute_profile.py
-------------------------------------
Follow-up to check_segy_time_slice_zeros.py: at time_ms=2196 that script
found 70% of traces zero at that one sample, in a coherent boundary shape
(not noise) -- consistent with a mute/taper zone. This script scans the
ENTIRE time axis and reports the zero-trace percentage at every sample,
so you can see in one shot whether that percentage shrinks as time
increases (a mute front that clears deeper in the volume) or stays flat
across the whole recorded window (a fixed survey-footprint boundary,
constant at every time slice).

Usage:
    cd backend
    python scripts/check_segy_mute_profile.py path/to/file.sgy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import segyio
import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_segy_mute_profile.py path/to/file.sgy")
        sys.exit(1)

    path = sys.argv[1]
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        twt_axis_ms = np.array(f.samples, dtype=float)
        traces = f.trace.raw[:]  # (n_traces, n_samples)

    is_zero = traces == 0.0
    zero_pct_per_sample = 100.0 * is_zero.sum(axis=0) / n_traces

    print(f"File: {path}")
    print(f"Time axis: {twt_axis_ms[0]}-{twt_axis_ms[-1]} ms, {len(twt_axis_ms)} samples\n")
    print(f"{'time_ms':>10} {'% zero traces':>15}")
    # Print every Nth sample so the table stays readable, plus first/last.
    n = len(twt_axis_ms)
    step = max(1, n // 40)
    for i in range(0, n, step):
        print(f"{twt_axis_ms[i]:>10.1f} {zero_pct_per_sample[i]:>14.1f}%")
    if (n - 1) % step != 0:
        print(f"{twt_axis_ms[-1]:>10.1f} {zero_pct_per_sample[-1]:>14.1f}%")

    print()
    first, last = zero_pct_per_sample[0], zero_pct_per_sample[-1]
    min_pct, max_pct = zero_pct_per_sample.min(), zero_pct_per_sample.max()
    print(f"Min zero%: {min_pct:.1f}  Max zero%: {max_pct:.1f}  (first sample: {first:.1f}%, last sample: {last:.1f}%)")

    if max_pct - min_pct < 5.0:
        print(
            "\n=> The zero-trace percentage is roughly CONSTANT across the entire recorded time "
            "window. This is NOT a mute front that clears with depth -- it's a fixed boundary, "
            "meaning the real acquired survey footprint is genuinely an irregular polygon "
            "(e.g. narrower at the edges of the inline/crossline range) padded into your "
            "245x252 rectangle. This is real, correct data: your survey just isn't a full "
            "rectangle. Fix is on the frontend: mask exact-zero cells as NaN in the Time Slice "
            "heatmap so the padding renders transparent instead of a flat colored blob "
            "competing with the real structure."
        )
    elif last < first - 20:
        print(
            "\n=> The zero-trace percentage drops substantially from shallow to deep time. This "
            "is consistent with a mute zone that clears with depth (e.g. a top mute / "
            "inner-trace mute applied in processing) -- try viewing Time Slice at a later "
            "(larger) time_ms value, where the zero fraction is lowest, to see a fuller "
            "picture. The frontend fix is the same either way: mask exact-zero as NaN."
        )
    else:
        print(
            "\n=> Mixed pattern -- check the printed table above for where zero% is lowest and "
            "view the Time Slice near that time_ms."
        )


if __name__ == "__main__":
    main()
