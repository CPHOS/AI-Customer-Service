"""
Thread-safe in-memory session store.

Each session tracks:
  - a unique session_id (UUID4)
  - timestamps (created / last active) for TTL eviction
  - conversation history: ordered list of (question, answer) turns

Sessions are evicted lazily on the next `get_or_create` call once they have
been idle for longer than `ttl_seconds`.  For production deployments with
multiple workers you should replace this with a Redis-backed store.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import TypedDict


class Turn(TypedDict):
    question: str
    answer: str


@dataclass
class Session:
    session_id:  str
    created_at:  float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    history:     list[Turn] = field(default_factory=list)


class SessionStore:
    """Thread-safe in-memory session store with idle-TTL eviction.

    Args:
        ttl_seconds:  Seconds of inactivity before a session is evicted.
        max_history:  Maximum number of Q&A turns kept per session.
                      Older turns are dropped when the limit is exceeded.
    """

    def __init__(self, ttl_seconds: int = 1800, max_history: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self.ttl = ttl_seconds
        self.max_history = max_history

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str | None) -> Session:
        """Return the existing session for *session_id*, or create a new one.

        If *session_id* is None or unknown, a fresh session is created with a
        new UUID.  Expired sessions are evicted on each call.
        """
        with self._lock:
            self._evict_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_active = time.monotonic()
                return session
            new_id = session_id if session_id else str(uuid.uuid4())
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Append a completed Q&A turn to the session history."""
        with self._lock:
            if session_id not in self._sessions:
                return
            session = self._sessions[session_id]
            session.history.append(Turn(question=question, answer=answer))
            # Keep only the most recent `max_history` turns
            if len(session.history) > self.max_history:
                session.history = session.history[-self.max_history :]

    def get_history(self, session_id: str) -> list[Turn]:
        """Return a snapshot of the conversation history for *session_id*."""
        with self._lock:
            session = self._sessions.get(session_id)
            return list(session.history) if session else []

    @property
    def active_count(self) -> int:
        """Number of currently live (non-expired) sessions."""
        with self._lock:
            self._evict_expired()
            return len(self._sessions)

    # ── Private ───────────────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Remove sessions that have been idle for longer than `ttl_seconds`.

        Must be called with `_lock` held.
        """
        now = time.monotonic()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self.ttl
        ]
        for sid in expired:
            del self._sessions[sid]
