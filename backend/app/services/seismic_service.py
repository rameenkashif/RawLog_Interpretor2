"""
seismic_service.py
--------------------
Application service layer tying together segy_loader, seismic_attributes,
config_loader, and the seismic repository. Routers should call into this
module rather than talking to those lower-level modules directly -- same
pattern as well_service.py.
"""

from __future__ import annotations

import io

import numpy as np

from app import seismic_attributes as sa
from app.config_loader import get_seismic_config
from app.models.schemas import (
    SeismicAttributesResponse,
    SeismicSectionResponse,
    SeismicSummary,
)
from app.segy_loader import LoadedSegy, SegyMetadata, load_segy_file
from app.seismic_repository import SeismicRepository, get_seismic_repository

# Caps on how much raw amplitude data is ever sent to the frontend in one
# response, regardless of the underlying dataset size -- a real SEG-Y line
# can have tens of thousands of traces and thousands of samples per trace,
# which would be far too large (and far too detailed to usefully render)
# in a single browser payload.
MAX_SECTION_TRACES = 400
MAX_SECTION_SAMPLES = 800


class SeismicDatasetNotFoundError(Exception):
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Seismic dataset '{dataset_id}' not found")


def process_and_store_segy_bytes(
    file_bytes: bytes, filename: str, repo: SeismicRepository | None = None
) -> SeismicSummary:
    """Full pipeline: raw SEG-Y bytes -> validated trace matrix -> seismic
    attribute computation -> persisted to the repository -> summary returned.
    """
    repo = repo or get_seismic_repository()

    loaded: LoadedSegy = load_segy_file(io.BytesIO(file_bytes), filename=filename)
    config = get_seismic_config(loaded.metadata.dataset_id)

    attributes = sa.run_seismic_interpretation(
        loaded.traces, loaded.metadata.sample_interval_ms, config
    )

    repo.save_dataset(
        loaded.metadata, loaded.traces, loaded.twt_axis_ms, loaded.trace_x, loaded.trace_y, attributes
    )
    return _build_seismic_summary(loaded.metadata, attributes)


def _build_seismic_summary(metadata: SegyMetadata, attributes) -> SeismicSummary:
    def safe_mean(col: str) -> float | None:
        if col not in attributes.columns or attributes[col].dropna().empty:
            return None
        return float(attributes[col].mean())

    return SeismicSummary(
        dataset_id=metadata.dataset_id,
        source_filename=metadata.source_filename,
        n_traces=metadata.n_traces,
        n_samples=metadata.n_samples,
        sample_interval_ms=metadata.sample_interval_ms,
        duration_ms=metadata.duration_ms,
        avg_rms_amplitude=safe_mean("RMS_AMPLITUDE"),
        avg_vsh_proxy=safe_mean("VSH_SEISMIC_PROXY"),
        avg_phie_proxy=safe_mean("PHIE_SEISMIC_PROXY"),
        avg_swe_proxy=safe_mean("SWE_SEISMIC_PROXY"),
    )


def list_seismic_summaries(
    repo: SeismicRepository | None = None,
) -> list[SeismicSummary]:
    repo = repo or get_seismic_repository()
    summaries = []
    for metadata in repo.list_datasets():
        result = repo.get_dataset(metadata.dataset_id)
        if result is None:
            continue
        _, _, _, _, _, attributes = result
        summaries.append(_build_seismic_summary(metadata, attributes))
    return summaries


def get_seismic_dataset(dataset_id: str, repo: SeismicRepository | None = None):
    repo = repo or get_seismic_repository()
    result = repo.get_dataset(dataset_id)
    if result is None:
        raise SeismicDatasetNotFoundError(dataset_id)
    return result


def get_seismic_summary(
    dataset_id: str, repo: SeismicRepository | None = None
) -> SeismicSummary:
    metadata, _, _, _, _, attributes = get_seismic_dataset(dataset_id, repo)
    return _build_seismic_summary(metadata, attributes)


def get_seismic_section(
    dataset_id: str, repo: SeismicRepository | None = None
) -> SeismicSectionResponse:
    """Return a display-ready, subsampled amplitude section (trace x time).

    Subsamples both axes down to MAX_SECTION_TRACES / MAX_SECTION_SAMPLES so
    the response payload stays a reasonable size regardless of how large the
    underlying SEG-Y file is.
    """
    metadata, traces, twt_axis_ms, _, _, _ = get_seismic_dataset(dataset_id, repo)

    n_traces, n_samples = traces.shape
    trace_step = max(1, n_traces // MAX_SECTION_TRACES)
    sample_step = max(1, n_samples // MAX_SECTION_SAMPLES)

    trace_indices = list(range(0, n_traces, trace_step))
    sample_indices = np.arange(0, n_samples, sample_step)

    subsampled = traces[np.ix_(trace_indices, sample_indices)]
    twt_subsampled = twt_axis_ms[sample_indices]

    return SeismicSectionResponse(
        dataset_id=dataset_id,
        trace_indices=trace_indices,
        twt_axis_ms=twt_subsampled.tolist(),
        amplitude=subsampled.tolist(),
    )


def get_seismic_attribute_series(
    dataset_id: str, repo: SeismicRepository | None = None
) -> SeismicAttributesResponse:
    _, _, _, _, _, attributes = get_seismic_dataset(dataset_id, repo)

    return SeismicAttributesResponse(
        dataset_id=dataset_id,
        trace_index=attributes["TRACE_INDEX"].astype(int).tolist(),
        rms_amplitude=attributes["RMS_AMPLITUDE"].tolist(),
        avg_envelope=attributes["AVG_ENVELOPE"].tolist(),
        dominant_freq_hz=attributes["DOMINANT_FREQ_HZ"].tolist(),
        vsh_seismic_proxy=attributes["VSH_SEISMIC_PROXY"].tolist(),
        phie_seismic_proxy=attributes["PHIE_SEISMIC_PROXY"].tolist(),
        swe_seismic_proxy=attributes["SWE_SEISMIC_PROXY"].tolist(),
    )


def export_seismic_attributes_csv(
    dataset_id: str, repo: SeismicRepository | None = None
) -> str:
    """Export the per-trace computed seismic attributes as CSV text."""
    _, _, _, _, _, attributes = get_seismic_dataset(dataset_id, repo)
    buf = io.StringIO()
    attributes.to_csv(buf, index=False)
    return buf.getvalue()
