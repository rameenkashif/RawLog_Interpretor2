"""
scripts/train_core_perm_model.py
----------------------------------
Trains the CORE_PERM_PRED proxy regression model (section 3.8) using every
well currently in the repository, and saves it to
backend/data/models/core_perm_model.joblib.

*** PROXY MODEL ***
No real core plug measurements exist for this field yet, so the model is
trained to predict PERM_TIXIER (i.e. it currently just learns a smoothed/
generalized version of the Tixier estimate from PHIE, VSH, and
PERM_TIXIER itself). As soon as real core permeability measurements are
available, update this script to load them and pass
`target_col="CORE_PERM_MEASURED"` to `train_core_perm_model()` instead.

Run this AFTER loading wells (bulk_load_wells.py or via the /wells/upload
endpoint), then re-run bulk_load_wells.py (or re-upload) so CORE_PERM_PRED
gets included in each well's processed curves.

Usage:
    cd backend
    python scripts/bulk_load_wells.py          # 1. load raw wells
    python scripts/train_core_perm_model.py    # 2. train the proxy model
    python scripts/bulk_load_wells.py          # 3. reprocess wells so CORE_PERM_PRED is included
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app import petrophysics as pp
from app.config_loader import get_well_config
from app.repository import MODELS_DIR, get_repository


def main() -> None:
    repo = get_repository()
    metadatas = repo.list_wells()
    if not metadatas:
        print(
            "No processed wells found. Run bulk_load_wells.py or upload wells via the API first."
        )
        sys.exit(1)

    frames = []
    for metadata in metadatas:
        result = repo.get_well(metadata.well_id)
        if result is None:
            continue
        _, df = result
        if {"PHIE", "VSH", "PERM_TIXIER"}.issubset(df.columns):
            frames.append(df[["PHIE", "VSH", "PERM_TIXIER"]])

    if not frames:
        print(
            "No wells with PHIE/VSH/PERM_TIXIER found -- did the interpretation pipeline run?"
        )
        sys.exit(1)

    training_df = pd.concat(frames, ignore_index=True)
    config = get_well_config(None)  # field-wide defaults for model hyperparameters

    print(f"Training on {len(training_df)} samples from {len(frames)} well(s)...")
    model = pp.train_core_perm_model(training_df, config)

    model_path = MODELS_DIR / "core_perm_model.joblib"
    pp.save_model(model, model_path)
    print(f"Saved model to {model_path}")
    print(
        "\nNOTE: CORE_PERM_PRED is a proxy target trained on PERM_TIXIER, not real core plugs. "
        "Re-run bulk_load_wells.py (or re-upload wells) to include CORE_PERM_PRED in the "
        "processed curves now that a model exists."
    )


if __name__ == "__main__":
    main()
