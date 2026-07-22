"""
services/dashboard_upload_service.py
---------------------------------------
Orchestrates the dashboard-level combined well+seismic upload: schedules
the slow SEG-Y/tie/synthetic/spectral work as a FastAPI BackgroundTasks
job, writes results into well_processing_cache_repository.py, and exposes
cache-first/live-fallback summary readers used by both the upload-status
endpoint (routers/dashboard.py) and the new agent tools
(services/anthropic_agent.py).

Deliberately orchestration only -- it does not implement or alter any tie
math. It calls tie_service.get_well_seismic_tie and
synthetic_seismogram_service.generate exactly as their existing routers
do, and seismic_processor.get_segy_volume()'s existing scalar-summary
methods (survey_info, get_amplitude_spectrum).

All seismic-dependent modules (seismic_service, tie_service,
synthetic_seismogram_service, seismic_processor) are imported lazily
inside function bodies, not at module top level -- routers/dashboard.py
is always registered (see main.py), so this module must not gain a hard
segyio/scipy dependency at import time; a missing dependency should only
break the seismic-specific parts of the dashboard, not /dashboard/summary.
"""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone

from app.well_processing_cache_repository import (
    WellProcessingCacheRecord,
    get_well_processing_cache_repository,
)

# A tie/synthetic result with correlation below this, or with its shift
# search pinned to the search boundary, is flagged low_confidence -- a new
# threshold for this feature only. Does NOT touch
# well_seismic_tie.BOUNDARY_PINNED_FRACTION, which governs the existing
# boundary_pinned flag itself.
TIE_LOW_CONFIDENCE_THRESHOLD = 0.3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seismic_deps_available() -> bool:
    try:
        import segyio  # noqa: F401
        import scipy  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def start_upload(well_id: str, segy_filename: str) -> str:
    """Synchronous: writes an initial status='processing' record and
    returns a run_token for the caller to pass into the background task.
    Called by the router right after LAS processing succeeds, before
    background_tasks.add_task schedules run_upload_pipeline.
    """
    run_token = uuid.uuid4().hex
    repo = get_well_processing_cache_repository()
    now = _now()
    repo.save(
        WellProcessingCacheRecord(
            well_id=well_id,
            status="processing",
            run_token=run_token,
            created_at=now,
            updated_at=now,
            segy_filename=segy_filename,
        )
    )
    return run_token


def _new_record(well_id: str) -> WellProcessingCacheRecord:
    """A minimal record for a well that predates this feature or was
    uploaded via the standalone UploadWells/SeismicUpload widgets -- so a
    live-fallback result in get_*_summary() below has somewhere to be
    opportunistically written back to, and is cached for the next call."""
    now = _now()
    return WellProcessingCacheRecord(
        well_id=well_id, status="ready", run_token=uuid.uuid4().hex, created_at=now, updated_at=now
    )


def _is_current_run(repo, well_id: str, run_token: str) -> bool:
    """A newer same-well upload may have started (and already written its
    own 'processing' record with a different run_token) while this run was
    still working -- in that case this run's result is stale and must not
    clobber the newer one. Simple compare-and-swap, no queue needed."""
    current = repo.get(well_id)
    return current is not None and current.run_token == run_token


def run_upload_pipeline(well_id: str, run_token: str, segy_bytes: bytes, segy_filename: str) -> None:
    """BackgroundTasks target. Never raises -- any unhandled exception is
    caught and written to the cache as status='failed', since
    BackgroundTasks otherwise only logs and discards exceptions, leaving
    the record stuck at 'processing' forever."""
    repo = get_well_processing_cache_repository()

    try:
        _run_pipeline_steps(repo, well_id, run_token, segy_bytes, segy_filename)
    except Exception as exc:  # noqa: BLE001
        if _is_current_run(repo, well_id, run_token):
            record = repo.get(well_id) or WellProcessingCacheRecord(
                well_id=well_id, status="failed", run_token=run_token, created_at=_now(), updated_at=_now()
            )
            record.status = "failed"
            record.error = f"{exc}\n{traceback.format_exc(limit=3)}"
            record.updated_at = _now()
            repo.save(record)


def _run_pipeline_steps(repo, well_id: str, run_token: str, segy_bytes: bytes, segy_filename: str) -> None:
    from app.segy_loader import SegyValidationError
    from app.services import seismic_service
    from app.services import seismic_processor as sp
    from app.services import tie_service
    from app.services import synthetic_seismogram_service
    from app.services.well_service import WellNotFoundError
    from app.well_seismic_tie import TieError

    def _save(**updates) -> WellProcessingCacheRecord | None:
        if not _is_current_run(repo, well_id, run_token):
            return None
        record = repo.get(well_id)
        for key, value in updates.items():
            setattr(record, key, value)
        record.updated_at = _now()
        repo.save(record)
        return record

    # Step 1: upload-pipeline SEG-Y processing (validates + gets a
    # dataset_id for the attribute cards + tie_service). A validation
    # failure here aborts the whole pipeline -- must not mutate the active
    # volume (step 2) with an unvalidated file.
    try:
        seismic_summary = seismic_service.process_and_store_segy_bytes(segy_bytes, segy_filename)
    except SegyValidationError as exc:
        _save(status="failed", error=f"Invalid SEG-Y file: {exc}")
        return
    dataset_id = seismic_summary.dataset_id
    record = _save(status="processing", dataset_id=dataset_id, segy_filename=segy_filename)
    if record is None:
        return  # superseded by a newer upload for this well

    # Step 2: this upload becomes the single active volume for Seismic
    # Visualization / the Synthetic Seismogram page. Prune older raw files
    # first -- the single-active-volume model has no use for them, and
    # leaving them around grows disk usage unboundedly (get_segy_volume's
    # discovery just picks the newest by mtime, so old files are pure
    # clutter, never used again).
    sp.RAW_SEISMIC_DIR.mkdir(parents=True, exist_ok=True)
    for stale in list(sp.RAW_SEISMIC_DIR.glob("*.sgy")) + list(sp.RAW_SEISMIC_DIR.glob("*.segy")):
        stale.unlink(missing_ok=True)
    (sp.RAW_SEISMIC_DIR / segy_filename).write_bytes(segy_bytes)
    volume = sp.get_segy_volume(refresh=True)

    # Step 3: the validated, unmodified well-to-seismic tie search.
    tie_inline: int | None = None
    try:
        tie = tie_service.get_well_seismic_tie(well_id, dataset_id)
        low_confidence = tie.correlation < TIE_LOW_CONFIDENCE_THRESHOLD or tie.boundary_pinned
        tie_inline = tie.inline
        _save(
            tie_available=True,
            tie_error=None,
            tie_correlation=tie.correlation,
            tie_boundary_pinned=tie.boundary_pinned,
            tie_low_confidence=low_confidence,
            tie_best_freq_hz=tie.best_freq_hz,
            tie_polarity=tie.polarity,
            tie_bulk_shift_ms=tie.bulk_shift_ms,
            tie_distance_m=tie.distance_m,
            tie_trace_index=tie.trace_index,
            tie_inline=tie.inline,
            tie_crossline=tie.crossline,
        )
    except (TieError, WellNotFoundError) as exc:
        _save(tie_available=False, tie_error=str(exc))

    # Step 4: the validated, unmodified synthetic seismogram generation.
    # wavelet_method="ricker" + auto_optimize_tie=True: generate()'s
    # defaults (statistical wavelet, no frequency search) are a much
    # weaker search than tie_service.get_well_seismic_tie's joint
    # frequency/polarity/shift search, so the two tie surfaces can
    # disagree sharply on the SAME well for reasons that are just search
    # thoroughness, not a real geological difference. This opts into the
    # comparable (Ricker frequency grid + polarity + shift) search so
    # eligibility/confidence here reflects the well's actual tie quality,
    # not this call's default search's weaker starting point. Still a
    # different algorithm/trace resolution than tie_service (see this
    # module's docstring), so results won't be bit-for-bit identical --
    # just no longer needlessly pessimistic.
    try:
        synth = synthetic_seismogram_service.generate(well_id, wavelet_method="ricker", auto_optimize_tie=True)
        synth_low_confidence = (
            synth["correlation"] < TIE_LOW_CONFIDENCE_THRESHOLD
            or synth["boundary_pinned"]
            or not synth["datum_check"]["plausible"]
        )
        _save(
            synthetic_available=True,
            synthetic_error=None,
            synthetic_correlation=synth["correlation"],
            synthetic_boundary_pinned=synth["boundary_pinned"],
            synthetic_low_confidence=synth_low_confidence,
            synthetic_datum_check_plausible=synth["datum_check"]["plausible"],
            synthetic_washout_count=int(sum(1 for w in synth["washout_flag"] if w)),
            synthetic_polarity=synth["polarity"],
            synthetic_best_shift_ms=synth["best_shift_ms"],
        )
    except Exception as exc:  # noqa: BLE001 -- broad: several distinct domain error types can surface here
        _save(synthetic_available=False, synthetic_error=str(exc))

    # Step 5: spectral summary at the tied inline, if a tie succeeded.
    # get_amplitude_spectrum returns scalar dominant_freq/bandwidth/snr --
    # not get_spectral_decomposition_inline, which returns a full
    # time x freq x position volume meant for the frontend's slider UI.
    if tie_inline is not None:
        try:
            spectrum = volume.get_amplitude_spectrum(inline_number=tie_inline)
            _save(
                spectral_available=True,
                spectral_error=None,
                spectral_inline=tie_inline,
                spectral_dominant_freq_hz=spectrum["dominant_freq_hz"],
                spectral_bandwidth_hz=spectrum["bandwidth_hz"],
                spectral_snr_proxy=spectrum.get("snr_proxy"),
            )
        except Exception as exc:  # noqa: BLE001
            _save(spectral_available=False, spectral_error=str(exc))

    _save(status="ready", error=None)


# -----------------------------------------------------------------------------
# Cache-first, live-compute-fallback summary readers -- used by both
# GET /dashboard/upload/{well_id}/status and the new agent tools. A cache
# miss (a well never uploaded through the new combined flow -- e.g. the
# original Z-02..Z-08 wells, or one uploaded via the standalone
# UploadWells/SeismicUpload widgets) falls back to the same live calls the
# existing routers make, so these stay correct for every well, not just
# ones processed by run_upload_pipeline. A successful live fallback is
# opportunistically written back into the cache.
# -----------------------------------------------------------------------------
def get_tie_summary(well_id: str) -> dict:
    repo = get_well_processing_cache_repository()
    record = repo.get(well_id)
    if record is not None and record.tie_available:
        return {
            "well_id": well_id,
            "dataset_id": record.dataset_id,
            "correlation": record.tie_correlation,
            "boundary_pinned": record.tie_boundary_pinned,
            "low_confidence": record.tie_low_confidence,
            "best_freq_hz": record.tie_best_freq_hz,
            "polarity": record.tie_polarity,
            "bulk_shift_ms": record.tie_bulk_shift_ms,
            "distance_m": record.tie_distance_m,
            "trace_index": record.tie_trace_index,
            "inline": record.tie_inline,
            "crossline": record.tie_crossline,
        }
    if record is not None and record.tie_error:
        return {"error": record.tie_error}

    # Live fallback: same call GET /tie/{well_id} makes, for a well not
    # (yet) processed by the dashboard-upload pipeline.
    from app.services import seismic_service, tie_service

    datasets = seismic_service.list_seismic_summaries()
    if not datasets:
        return {"error": "No seismic datasets available to tie against."}
    dataset_id = datasets[0].dataset_id
    try:
        tie = tie_service.get_well_seismic_tie(well_id, dataset_id)
    except Exception as exc:  # noqa: BLE001 -- TieError, WellNotFoundError, etc.
        return {"error": str(exc)}

    low_confidence = tie.correlation < TIE_LOW_CONFIDENCE_THRESHOLD or tie.boundary_pinned
    result = {
        "well_id": well_id,
        "dataset_id": dataset_id,
        "correlation": tie.correlation,
        "boundary_pinned": tie.boundary_pinned,
        "low_confidence": low_confidence,
        "best_freq_hz": tie.best_freq_hz,
        "polarity": tie.polarity,
        "bulk_shift_ms": tie.bulk_shift_ms,
        "distance_m": tie.distance_m,
        "trace_index": tie.trace_index,
        "inline": tie.inline,
        "crossline": tie.crossline,
    }
    record = record or _new_record(well_id)
    record.tie_available = True
    record.tie_error = None
    record.dataset_id = dataset_id
    record.tie_correlation = tie.correlation
    record.tie_boundary_pinned = tie.boundary_pinned
    record.tie_low_confidence = low_confidence
    record.tie_best_freq_hz = tie.best_freq_hz
    record.tie_polarity = tie.polarity
    record.tie_bulk_shift_ms = tie.bulk_shift_ms
    record.tie_distance_m = tie.distance_m
    record.tie_trace_index = tie.trace_index
    record.tie_inline = tie.inline
    record.tie_crossline = tie.crossline
    record.updated_at = _now()
    repo.save(record)
    return result


def get_synthetic_summary(well_id: str) -> dict:
    repo = get_well_processing_cache_repository()
    record = repo.get(well_id)
    if record is not None and record.synthetic_available:
        return {
            "well_id": well_id,
            "correlation": record.synthetic_correlation,
            "boundary_pinned": record.synthetic_boundary_pinned,
            "low_confidence": record.synthetic_low_confidence,
            "datum_check_plausible": record.synthetic_datum_check_plausible,
            "washout_count": record.synthetic_washout_count,
            "polarity": record.synthetic_polarity,
            "best_shift_ms": record.synthetic_best_shift_ms,
        }
    if record is not None and record.synthetic_error:
        return {"error": record.synthetic_error}

    # Live fallback for a well not yet processed by the dashboard pipeline.
    # Same wavelet_method="ricker" + auto_optimize_tie=True override as
    # run_upload_pipeline above, for the same reason (generate()'s plain
    # defaults are a much weaker search than tie_service's) -- NOT the
    # same as GET /api/synthetic/{well_id}/generate's own defaults, which
    # stay user-controlled per-request on the Synthetic Seismogram page;
    # this is specifically the confidence signal this cache/tools rely on.
    from app.services import synthetic_seismogram_service

    try:
        synth = synthetic_seismogram_service.generate(well_id, wavelet_method="ricker", auto_optimize_tie=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    low_confidence = (
        synth["correlation"] < TIE_LOW_CONFIDENCE_THRESHOLD
        or synth["boundary_pinned"]
        or not synth["datum_check"]["plausible"]
    )
    result = {
        "well_id": well_id,
        "correlation": synth["correlation"],
        "boundary_pinned": synth["boundary_pinned"],
        "low_confidence": low_confidence,
        "datum_check_plausible": synth["datum_check"]["plausible"],
        "washout_count": int(sum(1 for w in synth["washout_flag"] if w)),
        "polarity": synth["polarity"],
        "best_shift_ms": synth["best_shift_ms"],
    }
    record = record or _new_record(well_id)
    record.synthetic_available = True
    record.synthetic_error = None
    record.synthetic_correlation = synth["correlation"]
    record.synthetic_boundary_pinned = synth["boundary_pinned"]
    record.synthetic_low_confidence = low_confidence
    record.synthetic_datum_check_plausible = synth["datum_check"]["plausible"]
    record.synthetic_washout_count = result["washout_count"]
    record.synthetic_polarity = synth["polarity"]
    record.synthetic_best_shift_ms = synth["best_shift_ms"]
    record.updated_at = _now()
    repo.save(record)
    return result


def get_spectral_summary(well_id: str) -> dict:
    repo = get_well_processing_cache_repository()
    record = repo.get(well_id)
    if record is not None and record.spectral_available:
        return {
            "well_id": well_id,
            "available": True,
            "inline": record.spectral_inline,
            "dominant_freq_hz": record.spectral_dominant_freq_hz,
            "bandwidth_hz": record.spectral_bandwidth_hz,
            "snr_proxy": record.spectral_snr_proxy,
        }
    if record is not None and record.spectral_error:
        return {"available": False, "error": record.spectral_error}

    # Live fallback: needs a tie to know which inline to anchor to.
    tie = get_tie_summary(well_id)
    if "error" in tie or tie.get("inline") is None:
        return {
            "available": False,
            "error": tie.get("error", f"No tie available for well '{well_id}' to anchor a spectral summary."),
        }

    from app.services import seismic_processor as sp

    try:
        volume = sp.get_segy_volume()
        spectrum = volume.get_amplitude_spectrum(inline_number=tie["inline"])
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}

    result = {
        "well_id": well_id,
        "available": True,
        "inline": tie["inline"],
        "dominant_freq_hz": spectrum["dominant_freq_hz"],
        "bandwidth_hz": spectrum["bandwidth_hz"],
        "snr_proxy": spectrum.get("snr_proxy"),
    }
    # Re-fetch rather than reuse the `record` read at the top of this
    # function -- get_tie_summary() above may have just created/updated
    # the cache record via its own live fallback, and writing back a stale
    # in-memory copy here would silently clobber those tie fields.
    record = repo.get(well_id) or _new_record(well_id)
    record.spectral_available = True
    record.spectral_error = None
    record.spectral_inline = tie["inline"]
    record.spectral_dominant_freq_hz = spectrum["dominant_freq_hz"]
    record.spectral_bandwidth_hz = spectrum["bandwidth_hz"]
    record.spectral_snr_proxy = spectrum.get("snr_proxy")
    record.updated_at = _now()
    repo.save(record)
    return result


def is_active_volume_stale(record: WellProcessingCacheRecord) -> bool:
    """True if this record's segy_filename is no longer the currently
    active volume -- e.g. a later dashboard upload for a different well
    replaced it while this well's own background task was still running.
    """
    if not record.segy_filename:
        return False
    if not seismic_deps_available():
        return False
    try:
        from app.services import seismic_processor as sp

        active = sp.get_segy_volume()
        return active.path.name != record.segy_filename
    except Exception:  # noqa: BLE001
        return False


def get_field_overview() -> dict:
    """Cross-well summary combining petrophysical pay-zone quality with
    well-to-seismic tie/synthetic-seismogram confidence for every
    currently loaded well, in one call -- backs the agent's
    get_field_overview tool so a cross-well question (e.g. "which well
    has the best pay with a reliable tie") doesn't require looping
    get_zone_breakdown + get_well_seismic_tie + get_synthetic_seismogram
    once per well itself.

    Deliberately returns raw combined data per well, not a precomputed
    composite "best well" score -- weighing pay quality against tie
    confidence is exactly the judgment call a geophysicist (or the agent,
    per the system prompt's reasoning workflow) should make explicitly,
    not have hidden inside an invented formula.

    Tie/synthetic fields reuse get_tie_summary/get_synthetic_summary's
    existing cache-first behavior -- for a well never processed by the
    dashboard-upload pipeline, this falls back to a live computation (the
    same full frequency/polarity/shift search GET /tie/{well_id} runs),
    so a field-wide overview across several uncached wells can take a few
    seconds -- the same synchronous-per-well cost the app's existing
    batch tie endpoint (GET /tie/all) already accepts.
    """
    from app.services import well_service

    wells: list[dict] = []
    for summary in well_service.list_well_summaries():
        well_id = summary.well_id
        entry: dict = {
            "well_id": well_id,
            "footage_logged": summary.footage_logged,
            "net_pay_thickness": summary.net_pay_thickness,
            "avg_vsh": summary.avg_vsh,
            "avg_phie": summary.avg_phie,
            "avg_swe": summary.avg_swe,
            "pay_zone": None,
            "zone_error": None,
            "tie": None,
            "tie_error": None,
            "synthetic": None,
            "synthetic_error": None,
        }

        try:
            zones = well_service.get_well_zones(well_id)
            pay_row = next((z for z in zones.zones if z.zone_label == "Pay"), None)
            if pay_row is not None:
                entry["pay_zone"] = {
                    "thickness_m": pay_row.thickness,
                    "avg_phie": pay_row.avg_phie,
                    "avg_swe": pay_row.avg_swe,
                    "avg_vsh": pay_row.avg_vsh,
                    "n_samples": pay_row.n_samples,
                }
        except Exception as exc:  # noqa: BLE001 -- one well's zone lookup failing shouldn't drop it from the overview
            entry["zone_error"] = str(exc)

        if seismic_deps_available():
            tie = get_tie_summary(well_id)
            if "error" in tie:
                entry["tie_error"] = tie["error"]
            else:
                entry["tie"] = tie

            synthetic = get_synthetic_summary(well_id)
            if "error" in synthetic:
                entry["synthetic_error"] = synthetic["error"]
            else:
                entry["synthetic"] = synthetic
        else:
            entry["tie_error"] = "Seismic module unavailable."
            entry["synthetic_error"] = "Seismic module unavailable."

        wells.append(entry)

    return {"wells": wells}
