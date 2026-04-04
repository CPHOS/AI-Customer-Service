"""
Chat router.

POST /chat — submit a question and receive an answer.

Features
--------
* Per-IP rate limiting via slowapi (default: 30 req/min, configurable).
* Session tracking: backend-issued HttpOnly cookie (`cphos_sid` by default).
* Migration compatibility: optional request-body `session_id` fallback.
* The server maintains per-session conversation history for future use.
* X-Request-ID propagation from RequestIDMiddleware for log correlation.
* Pipeline runs in a thread-pool worker so the async event-loop is never
  blocked by the synchronous LLM/embedding calls.
* In-flight request deduplication: if the same session already has a
  pipeline call running (e.g. browser refreshed), the new handler joins
  the existing task instead of starting a redundant pipeline run.
"""
import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.deps import get_pipeline, get_session_store
from app.limiter import limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.sessions import SessionStoreProtocol
from pipeline import Pipeline
from utils.logger import get_logger

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)

# In-flight request tracker: session_id → asyncio.Task[(answer, latency)]
# Used to deduplicate concurrent requests for the same session (e.g. page
# refresh while waiting for a response).
_inflight: dict[str, asyncio.Task] = {}


def _extract_client_ip(request: Request) -> str:
    """Extract best-effort real client IP from proxy headers or socket peer."""
    x_forwarded_for = request.headers.get("x-forwarded-for", "")
    if x_forwarded_for:
        # RFC 7239 style chain: client, proxy1, proxy2 ...
        first = x_forwarded_for.split(",", 1)[0].strip()
        if first:
            return first

    x_real_ip = request.headers.get("x-real-ip", "").strip()
    if x_real_ip:
        return x_real_ip

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request:  Request,              # must be a direct param for slowapi
    response: Response,
    req:      ChatRequest,
    pipeline: Pipeline      = Depends(get_pipeline),
    sessions: SessionStoreProtocol = Depends(get_session_store),
) -> ChatResponse:
    """
    Ask the AI assistant a question.

    Session resolution priority:
    1) Cookie sid (authoritative)
    2) Request-body session_id (compatibility window only)
    3) Fresh server-issued sid
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The pipeline is not ready yet. Please retry shortly.",
        )

    settings = request.app.state.settings

    # ── Session resolution (cookie first, request-body fallback for migration) ─
    cookie_sid = request.cookies.get(settings.session_cookie_name)
    body_sid = req.session_id if settings.session_accept_body_id else None
    incoming_sid = cookie_sid or body_sid
    resolved = sessions.resolve(incoming_sid)
    session = resolved.session

    if resolved.state == "created":
        logger.info("Session created sid=%r source=%r", session.session_id, req.source)
    elif resolved.state == "reissued":
        logger.warning(
            "Session reissued old_sid=%r new_sid=%r source=%r",
            incoming_sid,
            session.session_id,
            req.source,
        )

    # Use session ID as the stable user identity so all turns are grouped
    user_id = session.session_id
    client_ip = _extract_client_ip(request)

    # ── Deduplicate in-flight requests for the same session ──────────────────
    sid = session.session_id
    existing_task = _inflight.get(sid)

    if existing_task is not None and not existing_task.done():
        # Another handler is already running the pipeline for this session
        # (e.g. the browser refreshed while waiting). Join it instead of
        # starting a redundant pipeline run.
        logger.info("Dedup: joining in-flight pipeline task for session %s", sid)
        try:
            answer, latency = await asyncio.shield(existing_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="In-flight request failed. Please try again.",
            )
    else:
        # No in-flight request — start the pipeline in a background task so it
        # survives handler cancellation (client disconnect on refresh).
        async def _run_pipeline() -> tuple[str, float]:
            t0 = time.monotonic()
            ans: str = await asyncio.to_thread(
                pipeline.answer,
                req.question,
                user_id=user_id,
                source=req.source,
                client_ip=client_ip,
            )
            lat = round(time.monotonic() - t0, 3)
            sessions.add_turn(sid, req.question, ans)
            return ans, lat

        task = asyncio.create_task(_run_pipeline())
        _inflight[sid] = task
        task.add_done_callback(lambda _: _inflight.pop(sid, None))

        try:
            answer, latency = await asyncio.shield(task)
        except asyncio.CancelledError:
            # Handler cancelled (client disconnected) but the task keeps running
            # so a subsequent request from the same session can still join it.
            raise

    # ── Cookie issue / refresh ───────────────────────────────────────────────
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session.session_id,
        max_age=settings.session_ttl,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        domain=settings.session_cookie_domain,
        path=settings.session_cookie_path,
    )

    return ChatResponse(
        answer     = answer,
        session_id = session.session_id if settings.session_return_body_id else "",
        latency_s  = latency,
    )


@router.post("/chat/reset")
async def reset_chat_session(request: Request, response: Response) -> dict[str, str]:
    """Reset browser session by deleting the session cookie.

    The server-side session record is left for TTL eviction; this endpoint only
    removes the browser's cookie so the next /chat call gets a fresh session.
    """
    settings = request.app.state.settings
    response.delete_cookie(
        key=settings.session_cookie_name,
        domain=settings.session_cookie_domain,
        path=settings.session_cookie_path,
    )
    return {"status": "ok"}
