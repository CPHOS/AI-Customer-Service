"""
FastAPI dependency-injection providers.

Import these with `Depends(...)` in route handlers to access shared state
without coupling route modules directly to `app.state`.
"""
from __future__ import annotations

from fastapi import Request

from app.sessions import SessionStore
from pipeline import Pipeline


def get_pipeline(request: Request) -> Pipeline:
    """Yield the global Pipeline instance stored on app state."""
    return request.app.state.pipeline


def get_session_store(request: Request) -> SessionStore:
    """Yield the global SessionStore instance stored on app state."""
    return request.app.state.session_store
