"""
Structured logging utility.

Two components:

1. ``get_logger(name)``
   Standard Python logger that writes INFO/WARNING/ERROR lines to stderr.
   Import via::

       from utils.logger import get_logger
       logger = get_logger(__name__)

2. ``ConversationLogger``
   Append-only JSONL conversation recorder.  Every completed Q&A turn is
   written as one JSON line to a file (default: ``conversations.jsonl``).

   Schema per record::

       {
         "ts":         "2026-04-03T11:10:03",   # ISO-8601 UTC timestamp
         "user_id":    "张老师",                 # arbitrary user identifier
         "source":     "cli",                   # channel: "cli" / "wechat" /
                                                #   "wecom" (企业微信) / etc.
         "category":   "B",                     # classifier topic letter
         "question":   "怎么添加学生信息？",
         "reply":      "请在小程序中点击…",
         "latency_s":  2.34                     # wall-clock seconds for answer()
       }

   Usage::

       from utils.logger import ConversationLogger
       conv_log = ConversationLogger()           # uses default path
       conv_log.record(
           user_id="张老师",
           source="cli",
           category="B",
           question="怎么添加学生信息？",
           reply="请在小程序中…",
           latency_s=2.34,
       )

   Designed for future WeChat / 企业微信 (WeCom) integration: pass the
   WeChat nickname or WeCom user-id string as ``user_id`` and set
   ``source="wechat"`` or ``source="wecom"``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FMT      = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
_console_handler.setLevel(logging.WARNING)   # silent by default

# Set by ConversationLogger.__init__; None until then.
_file_handler: logging.FileHandler | None = None


def get_logger(name: str) -> logging.Logger:
    """Return a named logger backed by the shared console (and file) handler(s).

    Calling this multiple times with the same *name* returns the same logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_console_handler)
        if _file_handler is not None:
            logger.addHandler(_file_handler)
        logger.setLevel(logging.DEBUG)   # logger passes everything; handlers filter
        logger.propagate = False
    return logger


class ConversationLogger:
    """Append-only JSONL conversation recorder.

    Each call to ``record()`` appends one JSON line to *path*.  The file is
    opened and closed per write so no data is lost if the process crashes.

    Args:
        path: Path to the JSONL file.  Created (with parent dirs) if absent.
    """

    def __init__(
        self,
        path:     str | Path = "logs/conversations.jsonl",
        verbose:  bool = False,
        log_file: str | Path | None = None,
    ) -> None:
        global _file_handler

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # ── console handler: INFO when verbose, otherwise silent ──────────────
        _console_handler.setLevel(logging.INFO if verbose else logging.WARNING)

        # ── file handler: always INFO+ regardless of verbose ──────────────────
        # Single process (e.g. WeCom server): one shared log file is fine —
        # Python's FileHandler uses a lock, so threads never interleave lines.
        # Multi-process (e.g. stress_test): append PID to avoid cross-process
        # corruption.  We detect this by checking whether we were spawned by
        # another Python process that is also running main.py.
        if log_file:
            _log_path = Path(log_file)
        else:
            parent_pid = os.getppid()
            is_subprocess = parent_pid != os.getpid() and Path(f"/proc/{parent_pid}").exists() if sys.platform == "linux" else False
            # Simpler cross-platform heuristic: if stdin is not a tty we are
            # likely a subprocess (stress_test pipes stdin).
            is_subprocess = not sys.stdin.isatty()
            if is_subprocess:
                _log_path = self._path.parent / f"{self._path.stem}_{os.getpid()}.log"
            else:
                _log_path = self._path.with_suffix(".log")
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(_log_path, encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        _file_handler.setLevel(logging.INFO)

        # Back-fill: attach the file handler to loggers already created before
        # ConversationLogger was instantiated (module-level get_logger() calls).
        for existing in logging.Logger.manager.loggerDict.values():
            if isinstance(existing, logging.Logger) and _console_handler in existing.handlers:
                if _file_handler not in existing.handlers:
                    existing.addHandler(_file_handler)

        self._sys_logger = get_logger(__name__)

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
        """Write one conversation turn to the JSONL file.

        Args:
            question:  The user's question text.
            reply:     The final AI reply sent to the user.
            user_id:   User identifier.  For CLI sessions this is the value
                       passed via ``--user``.  For WeChat / WeCom integrations
                       pass the nickname or WeCom userid here.
            source:    Channel name.  Suggested values:
                       ``"cli"`` / ``"wechat"`` / ``"wecom"`` / ``"api"``
            category:  The topic letter from the Classifier (A–G).
            latency_s: Wall-clock seconds from question received to reply sent.
            **extra:   Any additional key-value pairs to include in the record
                       (e.g. ``wechat_room="CPHOS技术组工作群"``).
        """
        record: dict[str, Any] = {
            "ts":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id":   user_id,
            "source":    source,
            "category":  category,
            "question":  question,
            "reply":     reply,
            "latency_s": round(latency_s, 3),
        }
        record.update(extra)

        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._sys_logger.error("ConversationLogger: failed to write record: %s", exc)

