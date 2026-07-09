"""
scripts/check_segy_grid_coverage.py
-------------------------------------
Diagnostic for the Time Slice "gray gaps" symptom: even when a survey's
total trace count exactly equals n_inlines x n_crosslines (a "full
rectangle" by the numbers), the (inline, crossline) grid built in
SegyVolume.__init__ can still have unpopulated cells if some (inline,
crossline) pairs are duplicated across traces -- numpy fancy-index
assignment (`grid[il_pos, xl_pos] = trace_idx`) silently keeps only the
last trace written for each duplicate pair, while other combinations
inside the rectangle never get written at all (stay -1 -> render as gray
NaN cells in the Time Slice heatmap).

This reads the SAME non-standard header bytes seismic_processor.py uses
(inline = FieldRecord, bytes 9-12; crossline = TraceNumber, bytes 13-16)
and reports:
  - total traces vs n_inlines * n_crosslines
  - number of duplicate (inline, crossline) pairs
  - number of missing (inline, crossline) pairs (cells that would be -1)
  - a handful of example duplicates/missing pairs

Usage:
    cd backend
    python scripts/check_segy_grid_coverage.py path/to/file.sgy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import segyio
import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_segy_grid_coverage.py path/to/file.sgy")
        sys.exit(1)

    path = sys.argv[1]
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        inline = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=int)
        crossline = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:], dtype=int)

    inlines_sorted = np.unique(inline)
    crosslines_sorted = np.unique(crossline)
    n_inlines = len(inlines_sorted)
    n_crosslines = len(crosslines_sorted)
    rectangle_size = n_inlines * n_crosslines

    print(f"File: {path}")
    print(f"Traces: {n_traces}")
    print(f"Inline range: {inline.min()}-{inline.max()} ({n_inlines} unique values)")
    print(f"Crossline range: {crossline.min()}-{crossline.max()} ({n_crosslines} unique values)")
    print(f"n_inlines * n_crosslines = {rectangle_size}")
    print()

    pairs = np.stack([inline, crossline], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    n_unique_pairs = len(unique_pairs)
    n_duplicated_traces = n_traces - n_unique_pairs

    print(f"Unique (inline, crossline) pairs: {n_unique_pairs}")
    print(f"Traces sharing a pair with another trace (excess writes): {n_duplicated_traces}")

    missing = rectangle_size - n_unique_pairs
    print(f"Grid cells that would be unpopulated (-1, renders gray): {missing}")
    print()

    if n_duplicated_traces == 0 and missing == 0:
        print(
            "=> No duplicate or missing (inline, crossline) pairs. The grid is genuinely "
            "fully populated -- the gray region is not explained by this hypothesis. Look "
            "elsewhere (e.g. actual NaN/dead traces in the data itself, or a frontend "
            "z-range/colorscale issue)."
        )
        return

    dup_mask = counts > 1
    print(f"=> Found {dup_mask.sum()} distinct (inline, crossline) pairs with duplicate traces.")
    if dup_mask.any():
        print("   Example duplicated pairs (inline, crossline, count):")
        for (il, xl), c in list(zip(unique_pairs[dup_mask], counts[dup_mask]))[:10]:
            print(f"     ({il}, {xl}) x{c}")

    if missing > 0:
        all_pairs_set = {(int(il), int(xl)) for il, xl in unique_pairs}
        example_missing = []
        for il in inlines_sorted:
            for xl in crosslines_sorted:
                if (int(il), int(xl)) not in all_pairs_set:
                    example_missing.append((int(il), int(xl)))
                    if len(example_missing) >= 10:
                        break
            if len(example_missing) >= 10:
                break
        print(f"\n   {missing} cells inside the {n_inlines}x{n_crosslines} rectangle have NO trace at all.")
        print("   Example missing (inline, crossline) pairs:")
        for il, xl in example_missing:
            print(f"     ({il}, {xl})")

    print(
        "\n=> This confirms the duplicate/missing-pair hypothesis: total trace count can "
        "equal n_inlines * n_crosslines while the grid still has real gaps, because some "
        "pairs are written more than once (overwriting each other) and others are never "
        "written. A fix would need to either (a) resolve duplicates deterministically "
        "(e.g. keep first, or average) instead of silently keeping 'last wins', and/or "
        "(b) surface missing cells clearly as 'no data' rather than looking like a bug."
    )


if __name__ == "__main__":
    main()
