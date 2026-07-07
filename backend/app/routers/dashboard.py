"""
routers/dashboard.py
---------------------
Aggregated multi-well statistics for the field-wide dashboard (section 4/6).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import DashboardSummary
from app.services import well_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary() -> DashboardSummary:
    """Field-wide summary: well count, total footage, average VSH/PHIE/SWE,
    and per-well summaries (used for the wells table + bar charts).
    """
    return well_service.get_dashboard_summary()
