"""
CPHOS AI Customer Service — server entry point.

All configuration is read from environment variables or a .env file.
See app/config.py (Settings) for the complete list of options.

Usage
-----
# Recommended: let uvicorn manage the process
uvicorn server:app --host 0.0.0.0 --port 8000

# Or run directly (useful for debugging / quick start)
python server.py

Key environment variables
--------------------------
LOAD_INDEX      cphos.npz            Pre-built index file (skips re-embedding)
REFS_DIR        references/          YAML knowledge-base directory
RATE_LIMIT      30/minute            Per-IP request limit for POST /chat
CORS_ORIGINS    *                    Comma-separated allowed origins
SESSION_TTL     1800                 Session idle-expiry in seconds
"""
from __future__ import annotations

from app.config import Settings
from app.main import create_app

# Build settings once; create_app wires everything together.
_settings = Settings()
app = create_app(_settings)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host                      = _settings.host,
        port                      = _settings.port,
        reload                    = _settings.reload,
        timeout_graceful_shutdown = _settings.shutdown_timeout,
    )

