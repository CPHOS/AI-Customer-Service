"""
FastAPI application factory.

`create_app(settings?)` is the single entry-point that wires together:
  - lifespan  — pipeline build on startup
  - middleware — CORS, X-Request-ID, slowapi rate-limiting
  - routers   — /health, /chat
  - static    — serves the SPA under /static and / → index.html
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import Settings
from app.limiter import limiter
from app.middleware.request_id import RequestIDMiddleware
from app.routers import chat, health
from app.sessions import SessionStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR   = str(_PROJECT_ROOT / "static")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and return the configured FastAPI application.

    Args:
        settings: Pre-built Settings instance.  When *None*, Settings() is
                  instantiated from environment variables / .env.
    """
    if settings is None:
        settings = Settings()

    # ── Lifespan ──────────────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Import here to avoid circular imports and to keep main.py as the
        # canonical pipeline factory that the CLI also uses.
        from main import build_pipeline  # project-root main.py
        import config as root_config

        doc_paths = (
            [settings.abs_path(p) for p in settings.doc_paths.split()]
            if settings.doc_paths.strip()
            else None
        )

        if settings.debug_mode:
            root_config.DEBUG_MODE = True

        app.state.pipeline = await asyncio.to_thread(
            build_pipeline,
            doc_paths     = doc_paths,
            refs_dir      = settings.abs_path(settings.refs_dir),
            load_index    = settings.abs_path(settings.load_index),
            save_index    = settings.abs_path(settings.save_index) if settings.save_index else None,
            conv_log_path = settings.abs_path(settings.logs_dir),
            verbose       = settings.log_verbose,
        )
        yield
        # Nothing to tear down — the pipeline has no persistent connections.

    # ── App ───────────────────────────────────────────────────────────────────
    app = FastAPI(
        title       = "CPHOS AI Customer Service",
        description = "RAG + multi-agent conversational Q&A.  POST /chat to ask a question.",
        version     = "1.0.0",
        lifespan    = lifespan,
    )

    # ── Shared state (available to all routes via request.app.state) ──────────
    app.state.settings      = settings
    app.state.pipeline      = None          # replaced by lifespan
    app.state.session_store = SessionStore(
        ttl_seconds = settings.session_ttl,
        max_history = settings.session_max_history,
    )
    app.state.limiter = limiter             # required by SlowAPIMiddleware

    # ── Middleware (registered in reverse order — last added runs first) ───────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins_list,
        allow_credentials = False,
        allow_methods     = ["GET", "POST"],
        allow_headers     = ["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SlowAPIMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)

    # ── Static files + SPA index ──────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    return app
