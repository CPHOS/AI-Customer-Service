"""
Chat router.

POST /chat — submit a question and receive an answer.

Features
--------
* Per-IP rate limiting via slowapi (default: 30 req/min, configurable).
* Session tracking: the client sends back `session_id` on subsequent turns;
  the server maintains per-session conversation history for future use.
* X-Request-ID propagation from RequestIDMiddleware for log correlation.
* Pipeline runs in a thread-pool worker so the async event-loop is never
  blocked by the synchronous LLM/embedding calls.
"""
import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.deps import get_pipeline, get_session_store
from app.limiter import limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.sessions import SessionStore
from pipeline import Pipeline

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request:  Request,              # must be a direct param for slowapi
    req:      ChatRequest,
    pipeline: Pipeline      = Depends(get_pipeline),
    sessions: SessionStore  = Depends(get_session_store),
) -> ChatResponse:
    """
    Ask the AI assistant a question.

    **session_id** — omit on the first request; include the value returned by
    the server on all subsequent requests to maintain conversation context.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The pipeline is not ready yet. Please retry shortly.",
        )

    # ── Session ───────────────────────────────────────────────────────────────
    session = sessions.get_or_create(req.session_id)

    # ── Client identity (used for logging; respect reverse-proxy headers) ─────
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )
    # Use session ID as the stable user identity so all turns are grouped
    user_id = session.session_id

    # ── Call pipeline in thread (synchronous, may block for several seconds) ──
    t0 = time.monotonic()
    answer: str = await asyncio.to_thread(
        pipeline.answer,
        req.question,
        user_id = user_id,
        source  = req.source,
    )
    latency = round(time.monotonic() - t0, 3)

    # ── Persist turn to session history ───────────────────────────────────────
    sessions.add_turn(session.session_id, req.question, answer)

    return ChatResponse(
        answer     = answer,
        session_id = session.session_id,
        latency_s  = latency,
    )
