"""
Typed settings for the CPHOS AI Customer Service backend.

All fields are read from environment variables (case-insensitive) or a .env
file at the project root.  pydantic-settings validates types at startup so
misconfigured deployments fail fast with clear error messages.

Environment variables (and their defaults):
    REFS_DIR            references/          Path to YAML reference files
    DOC_PATHS           ""                   Space-separated extra doc paths
    LOAD_INDEX          cphos.npz            Pre-built .npz index file
    SAVE_INDEX          (none)               Save built index to this path
    LOGS_DIR            logs                 Directory for per-session log files
    LOG_VERBOSE         false
    DEBUG_MODE          false                Print agent I/O on every turn

    HOST                0.0.0.0
    PORT                8000
    RELOAD              false                Uvicorn hot-reload (dev only)
    SHUTDOWN_TIMEOUT    5                    Graceful-shutdown seconds

    CORS_ORIGINS        *                    Comma-separated allowed origins
    RATE_LIMIT          30/minute            Per-IP chat rate limit

    SESSION_TTL         1800                 Session idle expiry (seconds)
    SESSION_MAX_HISTORY 20                   Max Q&A turns kept per session
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    refs_dir:      str        = "references/"
    doc_paths:     str        = ""           # space-separated list
    load_index:    str        = "cphos.npz"
    save_index:    str | None = None
    logs_dir:      str        = "logs"
    log_verbose:   bool       = False
    debug_mode:    bool       = False

    # ── Server ────────────────────────────────────────────────────────────────
    host:             str  = "0.0.0.0"
    port:             int  = 8000
    reload:           bool = False
    shutdown_timeout: int  = 5

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated.  For production set to specific origins, e.g.:
    #   CORS_ORIGINS=https://yourapp.com,https://admin.yourapp.com
    cors_origins: str = "*"

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # slowapi format: "<count>/<period>", e.g. "30/minute", "5/second"
    rate_limit: str = "30/minute"

    # ── Session ───────────────────────────────────────────────────────────────
    session_ttl:         int = 1800   # seconds of idle time before eviction
    session_max_history: int = 20     # max Q&A turns stored per session

    # ── Helpers ───────────────────────────────────────────────────────────────
    def abs_path(self, p: str) -> str:
        """Resolve *p* relative to the project root when not already absolute."""
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
