"""
scripts/bulk_load_wells.py
---------------------------
Convenience CLI to process every raw LAS file in backend/data/raw/
(Z-02.las ... Z-08.las) through the full interpretation pipeline and
persist the results, without going through the HTTP upload endpoint.

Usage:
    cd backend
    python scripts/bulk_load_wells.py
    python scripts/bulk_load_wells.py --folder path/to/other/las/files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import petrophysics as pp
from app.config_loader import get_well_config
from app.las_loader import LasValidationError, load_las_folder
from app.repository import get_repository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-load raw LAS files into the repository."
    )
    parser.add_argument(
        "--folder",
        default=str(Path(__file__).parent.parent / "data" / "raw"),
        help="Folder containing .las files (default: backend/data/raw)",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    loaded_wells = load_las_folder(folder)
    if not loaded_wells:
        print(f"No valid .las files found in {folder}")
        sys.exit(1)

    repo = get_repository()
    for well in loaded_wells:
        config = get_well_config(well.metadata.well_id)
        try:
            interpreted = pp.run_full_interpretation(
                well.df, config, step_depth=well.metadata.step
            )
        except LasValidationError as exc:
            print(f"  SKIPPED {well.metadata.well_id}: {exc}")
            continue

        repo.save_well(well.metadata, interpreted)
        print(
            f"  Loaded {well.metadata.well_id} ({well.metadata.well_name}): "
            f"{well.metadata.n_samples} samples, "
            f"{well.metadata.start_depth:.1f}-{well.metadata.stop_depth:.1f} m"
        )

    print(
        f"\nDone. {len(loaded_wells)} well(s) processed and saved to backend/data/processed/."
    )


if __name__ == "__main__":
    main()
