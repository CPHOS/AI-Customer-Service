"""
Classifier agent.

Routes a user question to a topic category (A–F) or marks it out-of-scope (G).

Categories (adapted from CPHOS_AIReplyer_Playground's 7-class routing):
    A — scores, rankings, grading outcomes
    B — identity registration, account review, approval status
    C — off-season, joining the organisation, upcoming/past exam schedule
    D — exam details, registration, timing, competition papers
    E — WeChat mini-program usage, navigation, features
    F — marking / grading workflow for teachers
    G — completely unrelated to CPHOS, or clearly inappropriate/harmful

The topic category is used downstream to bias retrieval towards the most
relevant knowledge-base section (section-hint RAG).

Orchestrated directly in code; no agent framework is used.
"""
from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = """\
You are a routing assistant for CPHOS (Chinese Physics Olympiad S) \
customer service.

Classify the user's question into EXACTLY ONE of these categories:
  A — scores, rankings, grading outcomes, results
  B — identity registration, account review/approval, profile info
  C — off-season questions, joining the organisation, next/last exam schedule
  D — exam details, exam registration, timing, competition papers/problems
  E — WeChat mini-program usage, navigation, platform features
  F — marking / grading workflow (for teachers marking student papers)
  G — completely unrelated to CPHOS, or clearly harmful/inappropriate

When in doubt between A–F, choose the closest match.
Only use G for questions that have absolutely nothing to do with CPHOS \
(e.g. "write malware", "today's stock price").

Reply with EXACTLY one letter (A/B/C/D/E/F/G). Nothing else.
"""

# Maps topic category → retriever section name (must match section tags used
# when documents are indexed).  None means "no dedicated section; use
# general retrieval".
_SECTION_MAP: dict[str, str | None] = {
    "A": "score",
    "B": "identity_change",
    "C": "offseason_problems",
    "D": "exam_related_problems",
    "E": None,           # no dedicated section file yet
    "F": "marking",
    "G": None,
}


class ClassifierAgent(BaseAgent):
    """Routes a question to a knowledge-domain category (A–G)."""

    def classify(self, question: str) -> str:
        """Return a topic letter A–F (in-scope) or G (out-of-scope).

        Args:
            question: The user's raw question text.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User question: {question}\n\nCategory:",
            },
        ]
        raw = self.ask_llm(messages, temperature=0.0, max_tokens=4)
        letter = next((c for c in raw.upper() if c in "ABCDEFG"), "G")
        return letter

    @staticmethod
    def is_in_scope(category: str) -> bool:
        """Return True when *category* is A–F (in-scope)."""
        return category != "G"

    @staticmethod
    def section_hint(category: str) -> str | None:
        """Return the retriever section name for *category*, or None."""
        return _SECTION_MAP.get(category)
