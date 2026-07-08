"""
well_service.py
----------------
Application service layer that ties together las_loader, petrophysics,
config_loader, and the repository. Routers should call into this module
rather than talking to those lower-level modules directly, so the
interpretation pipeline logic lives in exactly one place (also reused by
the Anthropic agent's tool functions).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from app import petrophysics as pp
from app.config_loader import get_well_config
from app.las_loader import LoadedWell, WellMetadata, load_las_file
from app.models.schemas import (
    CrossplotPoint,
    CrossplotResponse,
    DashboardSummary,
    WellSummary,
    WellZonesResponse,
    ZoneSummaryRow,
)
from app.repository import MODELS_DIR, WellRepository, get_repository

COMPUTED_CURVES = [
    "MD",
    "TVD",
    "VSH",
    "PHIT",
    "PHIE",
    "PHIE_DN",
    "SWE",
    "DPTM",
    "PERM_TIXIER",
    "VVOLC",
    "ZONES",
    "ZONES_LABEL",
]


class WellNotFoundError(Exception):
    def __init__(self, well_id: str):
        self.well_id = well_id
        super().__init__(f"Well '{well_id}' not found")


_core_perm_model_cache: Any = None
_core_perm_model_loaded = False


def get_core_perm_model() -> Any | None:
    """Lazily load the trained CORE_PERM_PRED regression model (see
    scripts/train_core_perm_model.py) from backend/data/models/, if it
    exists. Returns None if no model has been trained yet -- in that case
    CORE_PERM_PRED is simply omitted from the interpretation output.
    """
    global _core_perm_model_cache, _core_perm_model_loaded
    if not _core_perm_model_loaded:
        model_path = MODELS_DIR / "core_perm_model.joblib"
        if model_path.exists():
            _core_perm_model_cache = pp.load_model(model_path)
        _core_perm_model_loaded = True
    return _core_perm_model_cache


def process_and_store_las_bytes(
    file_bytes: bytes, filename: str, repo: WellRepository | None = None
) -> WellSummary:
    """Full pipeline: raw LAS bytes -> validated DataFrame -> petrophysical
    interpretation -> persisted to the repository -> summary returned.
    """
    repo = repo or get_repository()

    loaded: LoadedWell = load_las_file(io.BytesIO(file_bytes), filename=filename)
    config = get_well_config(loaded.metadata.well_id)

    interpreted_df = pp.run_full_interpretation(
        loaded.df,
        config,
        step_depth=loaded.metadata.step,
        core_perm_model=get_core_perm_model(),
    )

    repo.save_well(loaded.metadata, interpreted_df)
    return _build_well_summary(loaded.metadata, interpreted_df)


def _build_well_summary(metadata: WellMetadata, df: pd.DataFrame) -> WellSummary:
    footage = metadata.stop_depth - metadata.start_depth

    net_pay_thickness = None
    if "ZONES" in df.columns:
        n_pay_samples = int((df["ZONES"] == pp.ZONE_PAY).sum())
        net_pay_thickness = n_pay_samples * metadata.step

    def safe_mean(col: str) -> float | None:
        if col not in df.columns or df[col].dropna().empty:
            return None
        return float(df[col].mean())

    return WellSummary(
        well_id=metadata.well_id,
        well_name=metadata.well_name,
        start_depth=metadata.start_depth,
        stop_depth=metadata.stop_depth,
        step=metadata.step,
        n_samples=metadata.n_samples,
        footage_logged=footage,
        avg_vsh=safe_mean("VSH"),
        avg_phie=safe_mean("PHIE"),
        avg_swe=safe_mean("SWE"),
        net_pay_thickness=net_pay_thickness,
        null_counts=metadata.null_counts,
    )


def list_well_summaries(repo: WellRepository | None = None) -> list[WellSummary]:
    repo = repo or get_repository()
    summaries = []
    for metadata in repo.list_wells():
        loaded = repo.get_well(metadata.well_id)
        if loaded is None:
            continue
        _, df = loaded
        summaries.append(_build_well_summary(metadata, df))
    return summaries


def get_well_df(
    well_id: str, repo: WellRepository | None = None
) -> tuple[WellMetadata, pd.DataFrame]:
    repo = repo or get_repository()
    result = repo.get_well(well_id)
    if result is None:
        raise WellNotFoundError(well_id)
    return result


def get_well_summary(well_id: str, repo: WellRepository | None = None) -> WellSummary:
    metadata, df = get_well_df(well_id, repo)
    return _build_well_summary(metadata, df)


def get_well_curves(well_id: str, repo: WellRepository | None = None) -> dict[str, Any]:
    metadata, df = get_well_df(well_id, repo)
    clean_df = df.replace({np.nan: None})
    return {
        "well_id": well_id,
        "curve_names": list(df.columns),
        "depth_step": metadata.step,
        "n_samples": len(df),
        "data": clean_df.to_dict(orient="records"),
    }


def get_well_zones(
    well_id: str, repo: WellRepository | None = None
) -> WellZonesResponse:
    metadata, df = get_well_df(well_id, repo)

    if "ZONES" not in df.columns:
        return WellZonesResponse(well_id=well_id, zones=[])

    rows: list[ZoneSummaryRow] = []
    for code, label in pp.ZONE_LABELS.items():
        subset = df[df["ZONES"] == code]
        if subset.empty:
            rows.append(
                ZoneSummaryRow(
                    zone_code=code,
                    zone_label=label,
                    thickness=0.0,
                    n_samples=0,
                    avg_phie=None,
                    avg_swe=None,
                    avg_vsh=None,
                )
            )
            continue
        rows.append(
            ZoneSummaryRow(
                zone_code=code,
                zone_label=label,
                thickness=len(subset) * metadata.step,
                n_samples=len(subset),
                avg_phie=float(subset["PHIE"].mean()) if "PHIE" in subset else None,
                avg_swe=float(subset["SWE"].mean()) if "SWE" in subset else None,
                avg_vsh=float(subset["VSH"].mean()) if "VSH" in subset else None,
            )
        )

    return WellZonesResponse(well_id=well_id, zones=rows)


def get_crossplot(
    well_id: str,
    x_curve: str,
    y_curve: str,
    color_curve: str | None = None,
    repo: WellRepository | None = None,
) -> CrossplotResponse:
    metadata, df = get_well_df(well_id, repo)

    for curve in [x_curve, y_curve] + ([color_curve] if color_curve else []):
        if curve not in df.columns:
            raise ValueError(
                f"Curve '{curve}' not found in well '{well_id}'. Available: {list(df.columns)}"
            )

    points: list[CrossplotPoint] = []
    for _, row in df.iterrows():
        x_val = row[x_curve]
        y_val = row[y_curve]
        color_val = row[color_curve] if color_curve else None
        points.append(
            CrossplotPoint(
                x=None if pd.isna(x_val) else float(x_val),
                y=None if pd.isna(y_val) else float(y_val),
                color=(None if pd.isna(color_val) else color_val)
                if color_curve
                else None,
                depth=float(row["DEPT"]),
            )
        )

    return CrossplotResponse(
        well_id=well_id,
        x_curve=x_curve,
        y_curve=y_curve,
        color_curve=color_curve,
        points=points,
    )


def get_dashboard_summary(repo: WellRepository | None = None) -> DashboardSummary:
    summaries = list_well_summaries(repo)

    def safe_avg(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return float(np.mean(clean)) if clean else None

    # Seismic datasets are optional -- imported lazily here (rather than at
    # module load time) to keep well_service usable even if the seismic
    # module/dependencies (segyio, scipy) aren't installed in a given
    # deployment.
    try:
        from app.services import seismic_service

        seismic_summaries = seismic_service.list_seismic_summaries()
    except Exception:
        seismic_summaries = []

    return DashboardSummary(
        n_wells=len(summaries),
        total_footage=sum(s.footage_logged for s in summaries),
        avg_vsh=safe_avg([s.avg_vsh for s in summaries]),
        avg_phie=safe_avg([s.avg_phie for s in summaries]),
        avg_swe=safe_avg([s.avg_swe for s in summaries]),
        wells=summaries,
        n_seismic_datasets=len(seismic_summaries),
        seismic_datasets=seismic_summaries,
    )


def export_well_csv(well_id: str, repo: WellRepository | None = None) -> str:
    """Export the interpreted well as CSV text (raw + computed curves)."""
    _, df = get_well_df(well_id, repo)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_well_las(well_id: str, repo: WellRepository | None = None) -> str:
    """Export the interpreted well as a LAS 2.0 text file (raw + computed curves)."""
    import lasio

    metadata, df = get_well_df(well_id, repo)

    las = lasio.LASFile()
    las.well["WELL"] = metadata.well_name
    las.well["STRT"] = metadata.start_depth
    las.well["STOP"] = metadata.stop_depth
    las.well["STEP"] = metadata.step
    las.well["NULL"] = -9999.25

    for col in df.columns:
        if col == "ZONES_LABEL":
            continue  # LAS curves must be numeric; label is only in CSV/JSON export
        values = df[col].to_numpy(dtype=float)
        las.append_curve(col, np.nan_to_num(values, nan=-9999.25), unit="", descr="")

    buf = io.StringIO()
    las.write(buf, version=2.0)
    return buf.getvalue()
