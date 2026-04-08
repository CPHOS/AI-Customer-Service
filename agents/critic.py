"""
Critic agent.

Selects the better of two candidate answers using LLM judgement.
Adapted from ``choose_better_ans()`` in CPHOS_AIReplyer_Playground,
but as a proper class to match the agent architecture of this project.

The Critic is called after dual-path execution has produced two independent
candidate answers.  It acts as a lightweight pre-filter before the Verifier
does the final quality gate, reducing the chance that the Verifier receives
a weaker answer.

Orchestrated directly in code; no agent framework is used.
"""
from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = """\
You are an answer quality critic for a CPHOS (Chinese Physics Olympiad S) customer service system.

You will be given a user question and two candidate answers (A1 and A2).
Choose the BETTER answer using these rules:

1. Prefer an answer that DIRECTLY addresses the specific question asked.
2. Prefer an answer that gives CONCRETE, actionable information over vague generalities.
3. If one answer honestly says "I don't know, please contact support" and the
   other gives relevant information, prefer the one with relevant information.
4. If both are equally vague or unhelpful, prefer the one that is shorter and
   more honest about its limitations.

Reply with EXACTLY one character: "1" (A1 is better) or "2" (A2 is better).
Nothing else — no explanation, no punctuation.
"""


class CriticAgent(BaseAgent):
    """Selects the better of two candidate answers."""

    def choose_better(self, question: str, answer_a: str, answer_b: str) -> tuple[str, str, str]:
        """Return ``(chosen_answer, raw_llm_response, choice_label)``.

        *choice_label* is ``"A"`` or ``"B"`` indicating which path was selected.

        Falls back to *answer_a* (path A) if the LLM response cannot be parsed.

        Args:
            question: The original user question.
            answer_a: First candidate answer (from section-hinted retrieval).
            answer_b: Second candidate answer (from general retrieval).
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"A1: {answer_a}\n\n"
                    f"A2: {answer_b}\n\n"
                    "Better answer (1 or 2):"
                ),
            },
        ]
        raw = self.ask_llm(messages, temperature=0.0, max_tokens=4)

        # Find the last occurrence of "1" and "2" in the response,
        # matching the parse logic used in Playground's choose_better_ans().
        i1 = raw.rfind("1")
        i2 = raw.rfind("2")

        if i2 != -1 and (i1 == -1 or i2 > i1):
            return answer_b, raw, "B"
        return answer_a, raw, "A"  # default: prefer section-hinted answer
