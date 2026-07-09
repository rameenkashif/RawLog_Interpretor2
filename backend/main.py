"""
main.py
-------
FastAPI application entry point. Run with:

    cd backend
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import tie
load_dotenv()  # picks up backend/.env (ANTHROPIC_API_KEY, etc.)

logger = logging.getLogger("uvicorn.error")

from app.routers import chat, dashboard, wells  # noqa: E402 (import after load_dotenv)

app = FastAPI(
    title="RawReservoirClassifier",
    summary="Multi-Well Petrophysical Interpretation Platform",
    description=(
        "Reads raw LAS well logs, computes standard petrophysical interpretation "
        "curves, and exposes an Anthropic Claude-powered petrophysics assistant."
    ),
    version="1.0.0",
)

# Frontend dev server origin(s). Add production origins via FRONTEND_ORIGINS env var
# (comma-separated) when deploying.
default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
extra_origins = os.environ.get("FRONTEND_ORIGINS", "")
origins = default_origins + [o.strip() for o in extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wells.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(tie.router)

# The seismic module depends on extra packages (segyio, scipy) that ship in
# requirements.txt but may not yet be installed in every environment (e.g.
# right after pulling this feature without re-running `pip install -r
# requirements.txt`). Importing it defensively here means a missing/broken
# seismic dependency only disables the seismic endpoints -- it can never
# take down wells/dashboard/chat, which used to all fail together if this
# import raised at module load time.
try:
    from app.routers import seismic

    app.include_router(seismic.router)
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "Seismic module failed to load and its endpoints will be unavailable "
        "(GET/POST /seismic/*). This is usually caused by a missing dependency -- "
        "run `pip install -r requirements.txt` (needs segyio + scipy) and restart. "
        "Underlying error: %s",
        exc,
    )

# Seismic Visualization (inline/crossline sections, time slices, well ties,
# amplitude spectra read directly off the raw SEG-Y volume) shares the same
# segyio dependency and gets the same defensive import treatment. It's also
# independent of the routers above -- a missing backend/data/seismic_raw/
# file only 404s its own endpoints per-request (SegyFileNotFoundError), not
# at import time.
try:
    from app.routers import seismic_viz

    app.include_router(seismic_viz.router)
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "Seismic Visualization module failed to load and its endpoints will be "
        "unavailable (GET /api/seismic/*). This is usually caused by a missing "
        "dependency -- run `pip install -r requirements.txt` (needs segyio) and "
        "restart. Underlying error: %s",
        exc,
    )

# Synthetic seismogram / well-tie module (unit-standardized well header QC,
# selectable density/wavelet, washout QC, persisted manual stretch/squeeze)
# -- reuses seismic_processor.SegyVolume, so it shares the same dependency
# and gets the same defensive import treatment.
try:
    from app.routers import synthetic

    app.include_router(synthetic.router)
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "Synthetic Seismogram module failed to load and its endpoints will be "
        "unavailable (GET/PUT/DELETE /api/synthetic/*). This is usually caused by a "
        "missing dependency -- run `pip install -r requirements.txt` (needs segyio) "
        "and restart. Underlying error: %s",
        exc,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
