"""
Typed settings for the CPHOS AI Customer Service backend.

All fields are read from environment variables (case-insensitive) or a .env
file at the project root.  pydantic-settings validates types at startup so
misconfigured deployments fail fast with clear error messages.

Environment variables (and their defaults):
    REFS_DIR            references/docs      Path to YAML reference files
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
    CORS_ALLOW_CREDENTIALS false             Needed for cross-origin cookies
    RATE_LIMIT          30/minute            Per-IP chat rate limit

    SESSION_TTL         1800                 Session idle expiry (seconds)
    SESSION_MAX_HISTORY 20                   Max Q&A turns kept per session
    SESSION_BACKEND     memory               memory / redis
    REDIS_URL           (none)               Required for redis backend

    SESSION_COOKIE_NAME cphos_sid
    SESSION_COOKIE_SECURE false              true in production HTTPS
    SESSION_COOKIE_SAMESITE lax              lax / strict / none
    SESSION_COOKIE_DOMAIN (none)
    SESSION_COOKIE_PATH  /
    SESSION_ACCEPT_BODY_ID true              compatibility window toggle
    SESSION_RETURN_BODY_ID true              compatibility window toggle
"""
from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    refs_dir:      str        = "references/docs"
    doc_paths:     str        = ""           # space-separated list
    load_index:    str        = "cphos.npz"
    save_index:    str | None = None
    logs_dir:      str        = "logs"
    log_verbose:   bool       = False
    debug_mode:    bool       = False

    # ── Server ────────────────────────────────────────────────────────────────
    host:             str   = "0.0.0.0"
    port:             int   = 8000
    reload:           bool  = False
    shutdown_timeout: int   = 5
    pipeline_timeout: float = 120.0   # hard ceiling (seconds) per /chat request

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated.  For production set to specific origins, e.g.:
    #   CORS_ORIGINS=https://yourapp.com,https://admin.yourapp.com
    cors_origins: str = "*"
    cors_allow_credentials: bool = False

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # slowapi format: "<count>/<period>", e.g. "30/minute", "5/second"
    rate_limit: str = "30/minute"

    # ── Session ───────────────────────────────────────────────────────────────
    session_ttl:         int = 1800   # seconds of idle time before eviction
    session_max_history: int = 20     # max Q&A turns stored per session
    session_backend: str = "memory"  # memory / redis
    redis_url: str | None = None

    session_cookie_name: str = "cphos_sid"
    session_cookie_secure: bool = False
    session_cookie_samesite: str = "lax"
    session_cookie_domain: str | None = None
    session_cookie_path: str = "/"
    session_accept_body_id: bool = True
    session_return_body_id: bool = True

    # ── Helpers ───────────────────────────────────────────────────────────────
    def abs_path(self, p: str) -> str:
        """Resolve *p* relative to the project root when not already absolute."""
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("session_backend")
    @classmethod
    def _validate_session_backend(cls, v: str) -> str:
        vv = v.strip().lower()
        if vv not in {"memory", "redis"}:
            raise ValueError("session_backend must be 'memory' or 'redis'")
        return vv

    @field_validator("session_cookie_samesite")
    @classmethod
    def _validate_samesite(cls, v: str) -> str:
        vv = v.strip().lower()
        if vv not in {"lax", "strict", "none"}:
            raise ValueError("session_cookie_samesite must be lax/strict/none")
        return vv

    @model_validator(mode="after")
    def _validate_cookie_security(self) -> Settings:
        if self.session_cookie_samesite == "none" and not self.session_cookie_secure:
            raise ValueError("session_cookie_secure must be true when session_cookie_samesite=none")
        return self
