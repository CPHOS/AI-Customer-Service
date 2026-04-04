"""
Base agent class.

All LLM calls route through this class exclusively via the official
openai Python library. No third-party routing or agent frameworks are used.
"""
from __future__ import annotations

import time

import openai

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """Wraps an OpenAI chat model with retry logic.

    Third-party dependencies introduced here:
        - openai  (official OpenAI Python SDK)
    """

    _MAX_ATTEMPTS = 20
    _RETRY_SLEEP  = 5.0

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=openai.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0),
        )
        self.model   = model

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
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError as exc:
                last_exc = exc
                logger.warning(
                    "%s rate-limited (attempt %d/%d). Sleeping %.0fs…",
                    self.__class__.__name__, attempt + 1, self._MAX_ATTEMPTS, self._RETRY_SLEEP,
                )
                time.sleep(self._RETRY_SLEEP)
            except openai.APIStatusError as exc:
                last_exc = exc
                logger.warning(
                    "%s API error %s (attempt %d/%d): %s",
                    self.__class__.__name__, exc.status_code, attempt + 1, self._MAX_ATTEMPTS, exc.message,
                )
                time.sleep(self._RETRY_SLEEP)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s unexpected error (attempt %d/%d): %s",
                    self.__class__.__name__, attempt + 1, self._MAX_ATTEMPTS, exc,
                )
                time.sleep(self._RETRY_SLEEP)

        raise RuntimeError(
            f"LLM call failed after {self._MAX_ATTEMPTS} attempts. "
            f"Last error: {last_exc}"
        )
