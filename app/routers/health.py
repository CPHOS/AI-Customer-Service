"""
Health-check router.

GET /health — liveness probe.  Returns 200 once the pipeline is initialised.
Used by the frontend to poll readiness on load, and by load-balancers / k8s
readiness probes.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.chat import HealthResponse

router = APIRouter(tags=["utility"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service liveness and pipeline readiness."""
    pipeline = getattr(request.app.state, "pipeline", None)
    return HealthResponse(
        status="ok",
        pipeline="ready" if pipeline is not None else "not initialised",
    )
