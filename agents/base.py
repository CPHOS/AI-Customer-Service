"""
Base agent class.

All LLM calls route through this class exclusively via the official
openai Python library. No third-party routing or agent frameworks are used.
"""
from __future__ import annotations
import re
import time
from typing import Generator

import openai

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """Wraps an OpenAI chat model with retry logic.

    Third-party dependencies introduced here:
        - openai  (official OpenAI Python SDK)
    """

    _DEFAULT_MAX_ATTEMPTS = 5
    _DEFAULT_RETRY_SLEEP  = 2.0

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        *,
        max_attempts: int | None = None,
        retry_sleep: float | None = None,
    ) -> None:
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=openai.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0),
        )
        self.model   = model
        self._max_attempts = max_attempts if max_attempts is not None else self._DEFAULT_MAX_ATTEMPTS
        self._retry_sleep  = retry_sleep if retry_sleep is not None else self._DEFAULT_RETRY_SLEEP

    def ask_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat-completion request and return the reply text.

        Retries up to _MAX_ATTEMPTS times on transient errors (rate-limit,
        network, server errors) with a fixed sleep between attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = self._extract_content(response)
                return content
            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "%s rate-limited (attempt %d/%d). Sleeping %.0fs…",
                    self.__class__.__name__, attempt + 1, self._max_attempts, self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
            except openai.APIStatusError as exc:
                last_exc = exc
                logger.warning(
                    "%s API error %s (attempt %d/%d): %s",
                    self.__class__.__name__, exc.status_code, attempt + 1, self._max_attempts, exc.message,
                )
                time.sleep(self._retry_sleep)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s unexpected error (attempt %d/%d): %s",
                    self.__class__.__name__, attempt + 1, self._max_attempts, exc,
                )
                time.sleep(self._retry_sleep)

        raise RuntimeError(
            f"LLM call failed after {self._max_attempts} attempts. "
            f"Last error: {last_exc}"
        )

    def ask_llm_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Yield content chunks from a streaming chat-completion request.

        Retries the full request on transient errors.  ``<think>…</think>``
        blocks are silently discarded (same semantics as ``ask_llm``).
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                yield from self._filter_think_stream(stream)
                return
            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "%s stream rate-limited (attempt %d/%d). Sleeping %.0fs…",
                    self.__class__.__name__, attempt + 1, self._max_attempts, self._retry_sleep,
                )
                time.sleep(self._retry_sleep)
            except openai.APIStatusError as exc:
                last_exc = exc
                logger.warning(
                    "%s stream API error %s (attempt %d/%d): %s",
                    self.__class__.__name__, exc.status_code, attempt + 1, self._max_attempts, exc.message,
                )
                time.sleep(self._retry_sleep)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s stream unexpected error (attempt %d/%d): %s",
                    self.__class__.__name__, attempt + 1, self._max_attempts, exc,
                )
                time.sleep(self._retry_sleep)
        raise RuntimeError(
            f"LLM stream failed after {self._max_attempts} attempts. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _filter_think_stream(stream) -> Generator[str, None, None]:
        """Yield content deltas, silently dropping ``<think>…</think>`` regions."""
        in_think = False
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            text = delta.content if delta else None
            if not text:
                continue
            if in_think:
                if "</think>" in text:
                    in_think = False
                    rest = text.split("</think>", 1)[1]
                    if rest:
                        yield rest
                continue
            if "<think>" in text:
                in_think = True
                before = text.split("<think>", 1)[0]
                if before:
                    yield before
                continue
            yield text

    # ── Response extraction ────────────────────────────────────────────────────

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    @classmethod
    def _extract_content(cls, response) -> str:
        """Extract usable text from a chat-completion response.

        Handles reasoning / thinking models that may place output in extra
        fields (e.g. OpenRouter ``reasoning``) or wrap chain-of-thought in
        ``<think>…</think>`` tags inside the content field.
        """
        msg = response.choices[0].message
        content = msg.content

        # Fallback: reasoning models may return content=None with a separate field
        if content is None:
            extra = getattr(msg, "model_extra", None) or {}
            content = extra.get("reasoning") or extra.get("reasoning_content")

        if content is None:
            raise ValueError("LLM returned empty content (None)")

        # Strip <think>…</think> blocks emitted by some reasoning models
        content = cls._THINK_RE.sub("", content).strip()

        if not content:
            raise ValueError("LLM returned empty content after stripping think blocks")

        return content
