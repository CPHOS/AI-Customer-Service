"""
Structured logging utility.

Three components:

1. ``get_logger(name)``
   Standard Python logger that writes INFO/WARNING/ERROR lines to stderr
   and, if a session log context is active, to the current session's .log
   file as well.  Import via::

       from utils.logger import get_logger
       logger = get_logger(__name__)

2. ``dbg(label, text)``
   Centralised debug printer.  Reads ``config.DEBUG_MODE``; no-op when False.
   Import via::

       from utils.logger import dbg
       dbg("Classifier output", f"category={category!r}")

3. ``ConversationLogger``
   Per-session conversation recorder.  Every completed Q&A turn is written
   as two files under *logs_dir*:

   * ``<session_id>.jsonl`` -- one JSON line per turn (structured, for analysis)
   * ``<session_id>.log``   -- full pipeline log for this session: every
                              INFO/WARNING/ERROR emitted during ``answer()``
                              (classify, retrieve, verify, turn summary, etc.),
                              written in the same timestamp+level format as the
                              console logger.

   ``session_log_context(user_id)`` is a context manager that must be entered
   around each ``pipeline.answer()`` call.  It routes all Python logger output
   from the current thread to that session's ``.log`` file.  Multiple
   concurrent sessions in different threads each get their own file with no
   interleaving.

   Schema per JSONL record::

       {
         "ts":         "2026-04-03T11:10:03",
         "user_id":    "abc-123",
         "source":     "api",
         "category":   "B",
         "question":   "...",
         "reply":      "...",
         "latency_s":  2.34
       }
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ── Request-ID correlation ────────────────────────────────────────────────────
# Populated by the ``request_log_context`` context-manager (called from the
# chat router) so that every log line emitted during a request carries the
# ``X-Request-ID`` for easy correlation in aggregated log stores.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="",
)


class _RequestIDFilter(logging.Filter):
    """Inject ``%(request_id)s`` into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get("")  # type: ignore[attr-defined]
        return True


_FMT      = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(request_id)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_request_id_filter = _RequestIDFilter()

_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
_console_handler.setLevel(logging.WARNING)   # silent by default


class _ThreadRoutingHandler(logging.Handler):
    """Routes log records to a per-thread FileHandler.

    A single module-level instance is attached to every logger.  Callers
    register a FileHandler for the current thread; records emitted on that
    thread are forwarded to the registered handler.  Records on unregistered
    threads are silently dropped (they still reach the console handler).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._handlers: dict[int, logging.FileHandler] = {}
        self._lock = threading.Lock()

    def register(self, thread_id: int, handler: logging.FileHandler) -> None:
        with self._lock:
            self._handlers[thread_id] = handler

    def unregister(self, thread_id: int) -> None:
        with self._lock:
            self._handlers.pop(thread_id, None)

    def emit(self, record: logging.LogRecord) -> None:
        handler = self._handlers.get(threading.current_thread().ident)
        if handler:
            handler.emit(record)


_routing_handler = _ThreadRoutingHandler()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger backed by the shared console and routing handlers.

    Calling this multiple times with the same *name* returns the same logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addFilter(_request_id_filter)
        logger.addHandler(_console_handler)
        logger.addHandler(_routing_handler)
        logger.setLevel(logging.DEBUG)   # logger passes everything; handlers filter
        logger.propagate = False
    return logger


@contextmanager
def request_log_context(request_id: str) -> Iterator[None]:
    """Set the X-Request-ID for all log records emitted within this context.

    Usage (in an async route handler or middleware)::

        with request_log_context(request.state.request_id):
            ...  # all log lines include the request_id
    """
    token = _request_id_var.set(request_id)
    try:
        yield
    finally:
        _request_id_var.reset(token)


def dbg(label: str, text: str) -> None:
    """Print a debug block to stderr (and active session log) when ``config.DEBUG_MODE`` is True.

    This is the single, centralised debug printer used across the whole
    codebase.  Import once and call anywhere::

        from utils.logger import dbg
        dbg("Classifier output", f"category={category!r}")

    Output is gated by ``config.DEBUG_MODE`` so no call-site ``if debug:``
    guards are needed.
    """
    import config  # local import avoids circular dependency at module load time
    if not config.DEBUG_MODE:
        return
    _dbg_logger = logging.getLogger("debug")
    if not _dbg_logger.handlers:
        _dbg_logger.addHandler(_console_handler)
        _dbg_logger.addHandler(_routing_handler)
        _dbg_logger.setLevel(logging.DEBUG)
        _dbg_logger.propagate = False
    border = "─" * 60
    _dbg_logger.debug("%s\n[DEBUG] %s\n%s\n%s", border, label, text, border)


class ConversationLogger:
    """Per-session conversation recorder.

    For every completed Q&A turn, two files are written under *logs_dir*:

    * ``<session_id>.jsonl`` -- structured JSON line (for analysis / replay)
    * ``<session_id>.log``   -- full pipeline log: all INFO/WARNING/ERROR records
                               emitted during the ``answer()`` call (classify,
                               retrieve, verify steps plus the turn summary),
                               in the same timestamp+level format as the console.

    Use ``session_log_context(user_id)`` to activate per-thread routing of
    Python logger output to the session's .log file.

    Args:
        logs_dir: Directory where per-session files are created.
                  Created (with parents) if absent.
        verbose:  When True, elevate console handler to INFO level.
    """

    def __init__(
        self,
        logs_dir: str | Path = "logs",
        verbose:  bool = False,
    ) -> None:
        self._sessions_dir = Path(logs_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

        # -- console handler: INFO when verbose, otherwise silent
        _console_handler.setLevel(logging.INFO if verbose else logging.WARNING)

        self._sys_logger = get_logger(__name__)

    @staticmethod
    def _safe_id(user_id: str) -> str:
        """Sanitise *user_id* to a safe filename stem."""
        return (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))[:128]
            or "anonymous"
        )

    @contextmanager
    def session_log_context(self, user_id: str) -> Iterator[None]:
        """Context manager: route all pipeline log records to this session's .log file.

        Wrap each ``pipeline.answer()`` call with this so that INFO/WARNING/ERROR
        records (classify, retrieve, verify, turn summary ...) are written to
        ``<logs_dir>/<session_id>.log`` in addition to stderr.

        Thread-safe: concurrent sessions each route to their own file.
        """
        safe_id  = self._safe_id(user_id)
        log_file = self._sessions_dir / f"{safe_id}.log"
        handler  = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        handler.setLevel(logging.DEBUG)
        thread_id = threading.current_thread().ident
        _routing_handler.register(thread_id, handler)
        try:
            yield
        finally:
            _routing_handler.unregister(thread_id)
            handler.close()

    def record(
        self,
        *,
        question:  str,
        reply:     str,
        user_id:   str = "anonymous",
        source:    str = "cli",
        category:  str = "?",
        latency_s: float = 0.0,
        **extra: Any,
    ) -> None:
        """Write one conversation turn to the per-session JSONL file.

        The .log file is written automatically by the Python logger via
        ``session_log_context``; this method only writes the structured
        JSONL record.

        Args:
            question:  The user's question text.
            reply:     The final AI reply sent to the user.
            user_id:   Session / user identifier (used as the filename stem).
            source:    Channel name: ``"cli"`` / ``"api"`` / ``"wechat"`` / etc.
            category:  Topic letter from the Classifier (A-G).
            latency_s: Wall-clock seconds from question received to reply sent.
            **extra:   Additional key-value pairs included in the JSONL record.
        """
        ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        rec: dict[str, Any] = {
            "ts":        ts_iso,
            "user_id":   user_id,
            "source":    source,
            "category":  category,
            "question":  question,
            "reply":     reply,
            "latency_s": round(latency_s, 3),
        }
        rec.update(extra)

        safe_id    = self._safe_id(user_id)
        jsonl_file = self._sessions_dir / f"{safe_id}.jsonl"
        try:
            with jsonl_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._sys_logger.error("ConversationLogger: failed to write .jsonl: %s", exc)
