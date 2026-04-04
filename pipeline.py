"""
Pipeline orchestrator.

Wires together the three agents (Classifier → Executor → Verifier), an
optional Critic for dual-path selection, and the document Retriever.
All orchestration is explicit Python code — no third-party agent or
workflow framework is used.

Flow for each call to ``answer()``:

  Single-path mode (enable_dual_path=False or no Critic configured):
    1. Classifier decides topic category (A–F) or out-of-scope (G).
       - G  → return a polite decline immediately.
    2. Retrieve with section-hint boost from the detected topic.
    3. Executor generates a candidate answer.
    4. Verifier validates → polish and return, or retry.
    5. After max_retries exhausted → return an escalation message.

  Dual-path mode (enable_dual_path=True, Critic configured):
    1–2. Same as above, also retrieve without section-hint (general path).
    3.  Both retrieval paths run the Executor IN PARALLEL via a
        ThreadPoolExecutor (improvement over the serial dual-path in
        CPHOS_AIReplyer_Playground).
    4.  Critic selects the better of the two candidate answers.
    5.  Verifier validates the winner → polish and return, or retry.
    6.  After max_retries exhausted → return an escalation message.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Generator

from agents.classifier import ClassifierAgent
from agents.critic     import CriticAgent
from agents.executor   import ExecutorAgent
from agents.verifier   import VerifierAgent
from rag.retriever     import Retriever
from utils.logger      import ConversationLogger, dbg, get_logger

logger = get_logger(__name__)

# ── Fallback messages ─────────────────────────────────────────────────────────
_OUT_OF_SCOPE_REPLY = (
    "很抱歉，您的问题超出了我目前的知识范围，建议联系人工客服获取帮助。\n"
    "(Sorry, this question is outside my current knowledge. "
    "Please contact human customer service.)"
)

_EXHAUSTED_REPLY = (
    "很抱歉，我暂时无法为您提供满意的回答，请联系人工客服。\n"
    "(I was unable to produce a satisfactory answer after several attempts. "
    "Please contact human customer service.)"
)


class Pipeline:
    """Classifier → [dual-path Executor] → Critic → Verifier pipeline.

    Args:
        classifier:       ClassifierAgent instance.
        executor:         ExecutorAgent instance.
        verifier:         VerifierAgent instance.
        retriever:        Retriever instance (already indexed with documents).
        critic:           Optional CriticAgent.  When provided and
                          *enable_dual_path* is True, dual-path mode is active.
        top_k:            Number of document chunks to retrieve per query.
        max_retries:      Maximum number of Executor/Verifier retry cycles.
        enable_dual_path: Enable dual-path parallel execution + Critic.
                          Automatically disabled if no Critic is provided.
        conv_logger:      Optional ConversationLogger.  When provided, every
                          completed Q&A turn is recorded to the JSONL file.
    """

    def __init__(
        self,
        classifier:       ClassifierAgent,
        executor:         ExecutorAgent,
        verifier:         VerifierAgent,
        retriever:        Retriever,
        critic:           CriticAgent | None = None,
        top_k:            int = 5,
        max_retries:      int = 3,
        enable_dual_path: bool = True,
        conv_logger:      ConversationLogger | None = None,
    ) -> None:
        self.classifier       = classifier
        self.executor         = executor
        self.verifier         = verifier
        self.retriever        = retriever
        self.critic           = critic
        self.top_k            = top_k
        self.max_retries      = max_retries
        self.enable_dual_path = enable_dual_path and critic is not None
        self.conv_logger      = conv_logger

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        question:  str,
        user_id:   str = "anonymous",
        source:    str = "cli",
        client_ip: str = "unknown",
    ) -> str:
        """Process *question* and return the final reply string.

        Args:
            question: The user's question.
            user_id:  Caller identity string logged with every turn.
                      For CLI sessions this comes from ``--user``.
                      For WeChat / WeCom integrations pass the nickname or
                      WeCom userid (e.g. ``"张老师"`` / ``"zhanglaoshi"``).
            source:   Channel tag recorded in the conversation log.
                      Suggested values: ``"cli"`` / ``"wechat"`` / ``"wecom"``
            client_ip: Client IP address for audit and troubleshooting logs.

        Debug output is controlled globally via ``config.DEBUG_MODE``
        (set ``DEBUG_MODE=true`` in .env or pass ``--debug`` on the CLI).
        """
        _t0 = time.monotonic()

        _ctx = (
            self.conv_logger.session_log_context(user_id)
            if self.conv_logger else nullcontext()
        )
        with _ctx:
            return self._answer_inner(question, user_id, source, client_ip, _t0)

    # ── Streaming public API ──────────────────────────────────────────────────

    # SSE event types:
    #   ("status", {"step": ..., "message": ...})  — pipeline progress
    #   ("token",  {"text": ...})                   — streamed answer chunk
    # The caller is responsible for sending the final "done" event after
    # collecting the full answer.

    StreamEvent = tuple[str, dict]

    def answer_stream(
        self,
        question:  str,
        user_id:   str = "anonymous",
        source:    str = "cli",
        client_ip: str = "unknown",
    ) -> Generator[StreamEvent, None, None]:
        """Like :meth:`answer`, but yields ``(event_type, data)`` tuples.

        The preceding pipeline steps (classify, retrieve, execute, verify) run
        synchronously.  Only the final *summarize* LLM call is streamed so
        that tokens reach the client as they are generated.
        """
        _t0 = time.monotonic()

        _ctx = (
            self.conv_logger.session_log_context(user_id)
            if self.conv_logger else nullcontext()
        )
        with _ctx:
            yield from self._answer_stream_inner(question, user_id, source, client_ip, _t0)

    def _answer_stream_inner(
        self,
        question: str,
        user_id:  str,
        source:   str,
        client_ip: str,
        _t0:      float,
    ) -> Generator[StreamEvent, None, None]:
        """Inner streaming implementation, runs inside session_log_context."""
        def fmt(text: str) -> str:
            return text.replace("**", "")

        def _record(reply: str, category: str = "?") -> None:
            latency = time.monotonic() - _t0
            logger.info(
                "━━ Turn ━━ user=%r source=%r ip=%r category=%r latency=%.2fs\n"
                "  Q: %s\n  A: %s",
                user_id, source, client_ip, category, latency,
                question, reply,
            )
            if self.conv_logger:
                self.conv_logger.record(
                    question=question, reply=reply, user_id=user_id,
                    source=source, category=category, latency_s=latency,
                    client_ip=client_ip,
                )

        # ── Step 1: Classify ──────────────────────────────────────────────────
        yield ("status", {"step": "classifying", "message": "正在分析问题…"})
        logger.info("━━ Question ━━ user=%r ip=%r  %s", user_id, client_ip, question)
        category = self.classifier.classify(question)
        dbg("Classifier output", f"category={category!r}")

        if not self.classifier.is_in_scope(category):
            logger.info("Question classified out-of-scope (G). Declining.")
            yield ("token", {"text": _OUT_OF_SCOPE_REPLY})
            _record(_OUT_OF_SCOPE_REPLY, category)
            return

        section_hint = self.classifier.section_hint(category)
        logger.info("Classified as %r → section_hint=%r", category, section_hint)

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        yield ("status", {"step": "retrieving", "message": "正在检索知识库…"})
        chunks_primary = self.retriever.query(
            question, top_k=self.top_k, section_hint=section_hint,
        )
        dbg(
            f"RAG primary ({len(chunks_primary)} chunk(s), hint={section_hint!r})",
            "\n---\n".join(chunks_primary) if chunks_primary else "(empty)",
        )

        # ── Step 3: Dual-path parallel execution ─────────────────────────────
        yield ("status", {"step": "generating", "message": "正在生成回答…"})
        if self.enable_dual_path:
            chunks_general = self.retriever.query(question, top_k=self.top_k)
            dbg(
                f"RAG general ({len(chunks_general)} chunk(s))",
                "\n---\n".join(chunks_general) if chunks_general else "(empty)",
            )
            logger.info("Dual-path: launching parallel Executor calls.")
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(self.executor.execute, question, chunks_primary)
                fut_b = pool.submit(self.executor.execute, question, chunks_general)
            answer_a = fut_a.result()
            answer_b = fut_b.result()
            dbg("Executor Path A (section-hinted)", answer_a)
            dbg("Executor Path B (general)", answer_b)
            best_candidate: str | None = self.critic.choose_better(question, answer_a, answer_b)
            dbg("Critic selection", best_candidate)
            logger.info("Critic selected answer (Path %s).", "A" if best_candidate == answer_a else "B")
        else:
            best_candidate = None

        # ── Step 4: Verify → retry loop ───────────────────────────────────────
        previous_failures: list[tuple[str, str]] = []

        for attempt in range(self.max_retries + 1):
            if attempt == 0 and best_candidate is not None:
                raw_answer = best_candidate
            else:
                raw_answer = self.executor.execute(
                    question, chunks_primary, previous_failures or None,
                )
            dbg(f"Executor (attempt {attempt + 1})", raw_answer)

            is_valid, reason = self.verifier.verify(
                question, raw_answer,
                current_iter=attempt, total_iter=self.max_retries,
            )
            dbg(f"Verifier (attempt {attempt + 1})", f"valid={is_valid}  reason={reason}")

            if is_valid:
                # ── Stream the summarize step ─────────────────────────────────
                yield ("status", {"step": "polishing", "message": "正在优化回答…"})
                full_parts: list[str] = []
                for chunk in self.verifier.summarize_stream(question, raw_answer):
                    cleaned = fmt(chunk)
                    if cleaned:
                        full_parts.append(cleaned)
                        yield ("token", {"text": cleaned})
                _record("".join(full_parts), category)
                return

            previous_failures.append((raw_answer, reason))
            logger.warning(
                "Attempt %d/%d invalid. Reason: %s",
                attempt + 1, self.max_retries + 1, reason,
            )

        # ── Step 5: Exhausted ─────────────────────────────────────────────────
        logger.error("All %d attempts exhausted. Escalating.", self.max_retries + 1)
        yield ("token", {"text": _EXHAUSTED_REPLY})
        _record(_EXHAUSTED_REPLY, category)

    def _answer_inner(
        self,
        question: str,
        user_id:  str,
        source:   str,
        client_ip: str,
        _t0:      float,
    ) -> str:
        """Inner implementation of answer(), always runs inside session_log_context."""
        def fmt(text: str) -> str:
            """Strip Markdown bold markers (**) from the reply before sending."""
            return text.replace("**", "")

        def _done(reply: str, category: str = "?") -> str:
            """Record the turn and return the reply."""
            latency = time.monotonic() - _t0
            logger.info(
                "━━ Turn ━━ user=%r source=%r ip=%r category=%r latency=%.2fs\n"
                "  Q: %s\n"
                "  A: %s",
                user_id, source, client_ip, category, latency,
                question,
                reply,
            )
            if self.conv_logger:
                self.conv_logger.record(
                    question=question,
                    reply=reply,
                    user_id=user_id,
                    source=source,
                    category=category,
                    latency_s=latency,
                    client_ip=client_ip,
                )
            return reply

        # ── Step 1: Classify → topic category ────────────────────────────────
        logger.info("━━ Question ━━ user=%r ip=%r  %s", user_id, client_ip, question)
        category = self.classifier.classify(question)
        dbg("Classifier output", f"category={category!r}")

        if not self.classifier.is_in_scope(category):
            logger.info("Question classified out-of-scope (G). Declining.")
            return _done(_OUT_OF_SCOPE_REPLY, category)

        section_hint = self.classifier.section_hint(category)
        logger.info("Classified as %r → section_hint=%r", category, section_hint)

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        chunks_primary = self.retriever.query(
            question, top_k=self.top_k, section_hint=section_hint
        )
        # print(f"Retrieved {len(chunks_primary)} chunk(s) with section hint {section_hint!r}.")
        # print("Chunks:\n" + "\n---\n".join(chunks_primary) if chunks_primary else "(empty)")
        dbg(
            f"RAG primary ({len(chunks_primary)} chunk(s), hint={section_hint!r})",
            "\n---\n".join(chunks_primary) if chunks_primary else "(empty)",
        )

        # ── Step 3: Dual-path parallel execution ─────────────────────────────
        if self.enable_dual_path:
            chunks_general = self.retriever.query(question, top_k=self.top_k)
            dbg(
                f"RAG general ({len(chunks_general)} chunk(s))",
                "\n---\n".join(chunks_general) if chunks_general else "(empty)",
            )

            logger.info("Dual-path: launching parallel Executor calls.")
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(self.executor.execute, question, chunks_primary)
                fut_b = pool.submit(self.executor.execute, question, chunks_general)
            answer_a = fut_a.result()
            answer_b = fut_b.result()
            dbg("Executor Path A (section-hinted)", answer_a)
            dbg("Executor Path B (general)", answer_b)

            # Critic picks the better pre-candidate
            best_candidate: str | None = self.critic.choose_better(question, answer_a, answer_b)
            dbg("Critic selection", best_candidate)
            logger.info("Critic selected answer (Path %s).", "A" if best_candidate == answer_a else "B")
        else:
            best_candidate = None

        # ── Step 4: Verify → retry loop ───────────────────────────────────────
        # Critic winner (if any) is validated first; retries generate via Executor.
        previous_failures: list[tuple[str, str]] = []

        for attempt in range(self.max_retries + 1):
            if attempt == 0 and best_candidate is not None:
                raw_answer = best_candidate
            else:
                raw_answer = self.executor.execute(
                    question, chunks_primary, previous_failures or None
                )
            dbg(f"Executor (attempt {attempt + 1})", raw_answer)

            is_valid, reason = self.verifier.verify(
                question, raw_answer,
                current_iter=attempt,
                total_iter=self.max_retries,
            )
            dbg(f"Verifier (attempt {attempt + 1})", f"valid={is_valid}  reason={reason}")

            if is_valid:
                final = self.verifier.summarize(question, raw_answer)
                dbg("Verifier summarize", final)
                return _done(fmt(final), category)

            previous_failures.append((raw_answer, reason))
            logger.warning(
                "Attempt %d/%d invalid. Reason: %s",
                attempt + 1, self.max_retries + 1, reason,
            )

        # ── Step 5: Exhausted ─────────────────────────────────────────────────
        logger.error("All %d attempts exhausted. Escalating.", self.max_retries + 1)
        return _done(_EXHAUSTED_REPLY, category)
