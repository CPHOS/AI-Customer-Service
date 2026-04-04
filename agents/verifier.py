"""
Verifier agent.

Two responsibilities:
    1. verify()   — judge whether a candidate answer is valid.
    2. summarize() — polish a valid answer into the final user-facing reply.

Scoring logic mirrors CHOPS: INVALID answers with a high enough score are
promoted to VALID as retry iterations increase (leniency decay).

Orchestrated directly in code; no agent framework is used.
"""
from __future__ import annotations

from agents.base import BaseAgent

_VERIFY_SYSTEM = """\
You are an answer quality judge for a customer service system.
Given a question and a candidate answer, decide whether the answer is valid.

An answer is VALID if it:
  - Directly addresses the question, OR
  - Honestly says it does not know and suggests contacting human customer service.

An answer is INVALID if it:
  - Is clearly irrelevant to the question, OR
  - Contains fabricated facts that contradict common sense.

Reply in EXACTLY this format (one line, nothing else):
  VALID,<score 1-10>|||<brief reason>
  or
  INVALID,<score 1-10>|||<brief reason>

Examples:
  Q: How do I submit answer sheets?
  A: You need to be approved first before you can submit anything.
  → VALID,8|||Correct: unapproved users cannot submit.

  Q: Why was my account not approved?
  A: The weather is nice today.
  → INVALID,1|||Completely irrelevant answer.

  Q: How do I reset my password?
  A: I cannot find information about this. Please contact human customer service.
  → VALID,7|||Honest acknowledgement with appropriate referral.
"""

_SUMMARIZE_SYSTEM = """\
You are a customer service reply polisher.
Rewrite the answer below in a friendly, natural tone suitable for sending to a user.
Do NOT change the factual content. Keep it concise.
Respond in the same language as the original answer.
"""


class VerifierAgent(BaseAgent):
    """Validates and polishes candidate answers."""

    def verify(
        self,
        question: str,
        answer: str,
        current_iter: int = 0,
        total_iter: int = 3,
    ) -> tuple[bool, str]:
        """Return (is_valid, reason).

        Leniency promotion: if the LLM marks an answer INVALID but the score
        is high enough relative to the retry index, it is promoted to VALID.
        This prevents infinite rejection loops on borderline answers.
        """
        messages = [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    "Judgement:"
                ),
            },
        ]
        raw = self.ask_llm(messages, temperature=0.0, max_tokens=128)

        try:
            status_part, reason = raw.split("|||", 1)
            reason = reason.strip()
            is_invalid = "INVALID" in status_part.upper()

            if not is_invalid:
                return True, reason

            # Leniency: promote if score + leniency_bonus > threshold
            try:
                score = int(status_part.split(",")[1].strip())
                leniency_bonus = (current_iter / max(total_iter, 1)) * 3.0
                if score + leniency_bonus > 6:
                    return True, "Promoted by leniency."
            except (IndexError, ValueError):
                pass

            return False, reason

        except Exception:
            # Malformed response — accept to avoid infinite rejection loop
            return True, "Judgement parsing failed; accepted."

    def summarize(self, question: str, answer: str) -> str:
        """Polish a validated answer into the final user-facing reply."""
        messages = self._summarize_messages(question, answer)
        return self.ask_llm(messages, temperature=0.5)

    def summarize_stream(self, question: str, answer: str):
        """Streaming variant of :meth:`summarize` — yields content chunks."""
        messages = self._summarize_messages(question, answer)
        yield from self.ask_llm_stream(messages, temperature=0.5)

    @staticmethod
    def _summarize_messages(question: str, answer: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Raw answer: {answer}\n"
                    "Polished reply:"
                ),
            },
        ]
