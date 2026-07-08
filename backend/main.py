"""
main.py
-------
FastAPI application entry point. Run with:

    cd backend
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # picks up backend/.env (ANTHROPIC_API_KEY, etc.)

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
