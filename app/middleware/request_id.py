"""
X-Request-ID middleware (pure ASGI).

Attaches a unique request identifier to every HTTP transaction:
  - If the incoming request already carries an ``X-Request-ID`` header (set by
    an upstream proxy or the client), that value is reused.
  - Otherwise a new UUID4 is generated.

The ID is stored on ``scope["state"]["request_id"]`` (accessible via
``request.state.request_id``) and echoed in the response ``X-Request-ID``
header so clients can correlate requests to log entries.

This is a lightweight pure-ASGI middleware — it does **not** buffer the
response body and therefore works correctly with streaming / SSE responses
(unlike Starlette's ``BaseHTTPMiddleware``).
"""
from __future__ import annotations

import uuid
from typing import Any, Callable


class RequestIDMiddleware:
    """Pure ASGI middleware for X-Request-ID propagation."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract existing header or generate a new UUID
        request_id = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                request_id = value.decode("latin-1")
                break
        if not request_id:
            request_id = str(uuid.uuid4())

        # Expose via request.state.request_id
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)
