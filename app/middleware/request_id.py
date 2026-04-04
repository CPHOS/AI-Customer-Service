"""
X-Request-ID middleware.

Attaches a unique request identifier to every HTTP transaction:
  - If the incoming request already carries an `X-Request-ID` header (set by
    an upstream proxy or the client), that value is reused.
  - Otherwise a new UUID4 is generated.

The ID is stored on `request.state.request_id` for use in route handlers and
logs, and is echoed back in the `X-Request-ID` response header so clients can
correlate requests to log entries.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
