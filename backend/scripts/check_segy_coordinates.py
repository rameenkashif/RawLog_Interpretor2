"""
scripts/check_segy_coordinates.py
-----------------------------------
Diagnostic: prints what segy_loader.py's coordinate extraction actually
finds in a SEG-Y file's trace headers (CDP_X/CDP_Y, SourceX/SourceY,
SourceGroupScalar), so you can tell whether a tie is falling back to
tie_config.yaml's manual trace_index because the file genuinely has no
navigation in its trace headers (common for raw vendor exports before a
"load geometry" step), vs. some other issue.

Usage:
    cd backend
    python scripts/check_segy_coordinates.py path/to/file.sgy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import segyio
import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_segy_coordinates.py path/to/file.sgy")
        sys.exit(1)

    path = sys.argv[1]
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_n = min(n_traces, 10)

        print(f"File: {path}")
        print(f"Traces: {n_traces}\n")

        print(f"{'trace':>6} {'CDP_X':>12} {'CDP_Y':>12} {'SourceX':>12} {'SourceY':>12} {'scalar':>8}")
        for i in range(sample_n):
            h = f.header[i]
            print(
                f"{i:>6} "
                f"{h[segyio.TraceField.CDP_X]:>12} "
                f"{h[segyio.TraceField.CDP_Y]:>12} "
                f"{h[segyio.TraceField.SourceX]:>12} "
                f"{h[segyio.TraceField.SourceY]:>12} "
                f"{h[segyio.TraceField.SourceGroupScalar]:>8}"
            )

        cdp_x_all = np.array(f.attributes(segyio.TraceField.CDP_X)[:])
        cdp_y_all = np.array(f.attributes(segyio.TraceField.CDP_Y)[:])
        src_x_all = np.array(f.attributes(segyio.TraceField.SourceX)[:])
        src_y_all = np.array(f.attributes(segyio.TraceField.SourceY)[:])

        print()
        print(f"CDP_X: {'all zero' if not np.any(cdp_x_all) else f'{np.count_nonzero(cdp_x_all)}/{n_traces} nonzero'}")
        print(f"CDP_Y: {'all zero' if not np.any(cdp_y_all) else f'{np.count_nonzero(cdp_y_all)}/{n_traces} nonzero'}")
        print(f"SourceX: {'all zero' if not np.any(src_x_all) else f'{np.count_nonzero(src_x_all)}/{n_traces} nonzero'}")
        print(f"SourceY: {'all zero' if not np.any(src_y_all) else f'{np.count_nonzero(src_y_all)}/{n_traces} nonzero'}")

        if not np.any(cdp_x_all) and not np.any(cdp_y_all) and not np.any(src_x_all) and not np.any(src_y_all):
            print(
                "\n=> No coordinates found anywhere in the trace headers. This file has no "
                "navigation baked in -- the tie will fall back to tie_config.yaml's manual "
                "trace_index until real coordinates are written into the headers (e.g. by "
                "merging a .p190/.sps/UKOOA navigation file, or reprocessing/re-exporting "
                "with geometry loaded)."
            )
        else:
            print("\n=> Coordinates are present. If the tie is still falling back, check that "
                  "the well's LAS header (XWELL/YWELL) also has coordinates, and that this "
                  "dataset was re-uploaded/reprocessed *after* the coordinate-extraction code "
                  "was added (old cached datasets in backend/data/seismic_processed/ won't "
                  "have trace_x/trace_y until re-loaded).")


if __name__ == "__main__":
    main()
