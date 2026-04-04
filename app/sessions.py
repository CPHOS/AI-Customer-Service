"""Session storage layer.

This module provides:
1) A storage protocol used by route handlers (`SessionStoreProtocol`)
2) A secure in-memory implementation (`InMemorySessionStore`)
3) An optional Redis implementation (`RedisSessionStore`)

Key security behavior:
- Session IDs are always generated server-side via `secrets.token_urlsafe`
- Unknown / invalid client-provided IDs are NOT reused; a fresh ID is issued
"""
from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, Protocol, TypedDict


SID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


class Turn(TypedDict):
    question: str
    answer: str


@dataclass
class Session:
    session_id:  str
    created_at:  float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    history:     list[Turn] = field(default_factory=list)


@dataclass
class SessionResolution:
    session: Session
    state: Literal["created", "reused", "reissued"]


class SessionStoreProtocol(Protocol):
    ttl: int
    max_history: int

    def resolve(self, session_id: str | None) -> SessionResolution:
        ...

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        ...

    def get_history(self, session_id: str) -> list[Turn]:
        ...

    @property
    def active_count(self) -> int:
        ...


def generate_session_id() -> str:
    """Generate a high-entropy server-side session ID."""
    return secrets.token_urlsafe(24)


def is_valid_session_id(session_id: str | None) -> bool:
    if not session_id:
        return False
    return bool(SID_RE.match(session_id))


class InMemorySessionStore:
    """Thread-safe in-memory session store with idle-TTL eviction."""

    def __init__(self, ttl_seconds: int = 1800, max_history: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self.ttl = ttl_seconds
        self.max_history = max_history

    def resolve(self, session_id: str | None) -> SessionResolution:
        """Resolve incoming session id to an active session.

        Reuse when valid+exists, otherwise create/reissue a new server ID.
        """
        with self._lock:
            self._evict_expired()

            if is_valid_session_id(session_id) and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_active = time.monotonic()
                return SessionResolution(session=session, state="reused")

            if session_id:
                # Client supplied an unknown/invalid id: reissue for safety.
                new_id = generate_session_id()
                session = Session(session_id=new_id)
                self._sessions[new_id] = session
                return SessionResolution(session=session, state="reissued")

            new_id = generate_session_id()
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return SessionResolution(session=session, state="created")

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                return
            session = self._sessions[session_id]
            session.history.append(Turn(question=question, answer=answer))
            if len(session.history) > self.max_history:
                session.history = session.history[-self.max_history :]

    def get_history(self, session_id: str) -> list[Turn]:
        with self._lock:
            session = self._sessions.get(session_id)
            return list(session.history) if session else []

    @property
    def active_count(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._sessions)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self.ttl
        ]
        for sid in expired:
            del self._sessions[sid]


class RedisSessionStore:
    """Redis-backed session store for multi-worker deployments."""

    def __init__(self, redis_url: str, ttl_seconds: int = 1800, max_history: int = 20) -> None:
        try:
            import redis
        except Exception as exc:
            raise RuntimeError("Redis backend selected but 'redis' package is not installed") from exc

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds
        self.max_history = max_history

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def resolve(self, session_id: str | None) -> SessionResolution:
        now = time.time()

        if is_valid_session_id(session_id):
            key = self._key(session_id)
            if self._redis.exists(key):
                self._redis.hset(key, mapping={"last_active": str(now)})
                self._redis.expire(key, self.ttl)
                return SessionResolution(
                    session=Session(
                        session_id=session_id,
                        created_at=float(self._redis.hget(key, "created_at") or now),
                        last_active=now,
                        history=self.get_history(session_id),
                    ),
                    state="reused",
                )

        new_id = generate_session_id()
        key = self._key(new_id)
        self._redis.hset(
            key,
            mapping={
                "created_at": str(now),
                "last_active": str(now),
                "history": "[]",
            },
        )
        self._redis.expire(key, self.ttl)
        return SessionResolution(
            session=Session(session_id=new_id, created_at=now, last_active=now, history=[]),
            state="reissued" if session_id else "created",
        )

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        if not is_valid_session_id(session_id):
            return
        key = self._key(session_id)
        raw = self._redis.hget(key, "history")
        if raw is None:
            return
        history = json.loads(raw)
        history.append({"question": question, "answer": answer})
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        self._redis.hset(
            key,
            mapping={"history": json.dumps(history, ensure_ascii=False), "last_active": str(time.time())},
        )
        self._redis.expire(key, self.ttl)

    def get_history(self, session_id: str) -> list[Turn]:
        if not is_valid_session_id(session_id):
            return []
        raw = self._redis.hget(self._key(session_id), "history")
        if not raw:
            return []
        try:
            return list(json.loads(raw))
        except Exception:
            return []

    @property
    def active_count(self) -> int:
        # O(N) scan; sufficient for simple observability endpoint/logging.
        return sum(1 for _ in self._redis.scan_iter(match="session:*", count=1000))


def create_session_store(*, backend: str, ttl_seconds: int, max_history: int, redis_url: str | None) -> SessionStoreProtocol:
    """Factory for session store backend selection.

    Args:
        backend: "memory" or "redis"
        ttl_seconds: session idle TTL
        max_history: max turns per session
        redis_url: required when backend == "redis"
    """
    if backend == "redis":
        if not redis_url:
            raise RuntimeError("SESSION_BACKEND=redis requires REDIS_URL")
        return RedisSessionStore(redis_url=redis_url, ttl_seconds=ttl_seconds, max_history=max_history)
    return InMemorySessionStore(ttl_seconds=ttl_seconds, max_history=max_history)
