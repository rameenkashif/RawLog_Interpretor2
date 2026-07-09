"""
scripts/bulk_load_seismic.py
------------------------------
Convenience CLI to process every raw SEG-Y file in backend/data/seismic_raw/
through the seismic attribute pipeline and persist the results, without
going through the HTTP upload endpoint. Mirrors bulk_load_wells.py.

Usage:
    cd backend
    python scripts/bulk_load_seismic.py
    python scripts/bulk_load_seismic.py --folder path/to/other/segy/files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import seismic_attributes as sa
from app.config_loader import get_seismic_config
from app.segy_loader import SegyValidationError, load_segy_file
from app.seismic_repository import get_seismic_repository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-load raw SEG-Y files into the repository."
    )
    parser.add_argument(
        "--folder",
        default=str(Path(__file__).parent.parent / "data" / "seismic_raw"),
        help="Folder containing .sgy/.segy files (default: backend/data/seismic_raw)",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    segy_paths = sorted(list(folder.glob("*.sgy")) + list(folder.glob("*.segy")))
    if not segy_paths:
        print(f"No .sgy/.segy files found in {folder}")
        sys.exit(1)

    repo = get_seismic_repository()
    n_loaded = 0
    for path in segy_paths:
        try:
            loaded = load_segy_file(path)
        except SegyValidationError as exc:
            print(f"  SKIPPED {path.name}: {exc}")
            continue

        config = get_seismic_config(loaded.metadata.dataset_id)
        attributes = sa.run_seismic_interpretation(
            loaded.traces, loaded.metadata.sample_interval_ms, config
        )
        repo.save_dataset(
            loaded.metadata,
            loaded.traces,
            loaded.twt_axis_ms,
            loaded.trace_x,
            loaded.trace_y,
            attributes,
        )
        n_loaded += 1
        print(
            f"  Loaded {loaded.metadata.dataset_id} ({loaded.metadata.source_filename}): "
            f"{loaded.metadata.n_traces} traces, {loaded.metadata.n_samples} samples/trace, "
            f"{loaded.metadata.sample_interval_ms:.2f} ms sample interval"
        )

    print(
        f"\nDone. {n_loaded} seismic dataset(s) processed and saved to backend/data/seismic_processed/."
    )


if __name__ == "__main__":
    main()
