"""
Classifier agent.

Routes a user question to a topic category (A–F), marks it out-of-scope (G),
or identifies it as a time-sensitive query (H) that requires live web data.

Categories:
    A — scores, rankings, grading outcomes
    B — identity registration, account review, approval status
    C — off-season, joining the organisation, upcoming/past exam schedule
    D — exam details, registration, timing, competition papers
    E — WeChat mini-program usage, navigation, features
    F — marking / grading workflow for teachers
    G — completely unrelated to CPHOS, or clearly inappropriate/harmful
    H — time-sensitive: latest announcements, upcoming exam dates, recent
        events, news that changes over time and is NOT in a static FAQ

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
  H — time-sensitive questions that require the LATEST information from \
the CPHOS official website, such as: "最近一次联考是什么时候", \
"有什么新通知", "最新的活动是什么", "下一次考试", or any question \
about recent/upcoming announcements, events, or schedules whose answer \
changes over time and would NOT be in a static FAQ

When in doubt between A–F, choose the closest match.
Only use G for questions that have absolutely nothing to do with CPHOS \
(e.g. "write malware", "today's stock price").
Use H ONLY when the question clearly requires up-to-date, time-sensitive \
information that a static knowledge base cannot answer.

Reply with EXACTLY one letter (A/B/C/D/E/F/G/H). Nothing else.
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
    "H": None,           # time-sensitive → web fetch, no RAG section
}


class ClassifierAgent(BaseAgent):
    """Routes a question to a knowledge-domain category (A–G)."""

    def classify(self, question: str) -> tuple[str, str]:
        """Return ``(category_letter, raw_llm_response)``.

        Args:
            question: The user's raw question text.

        Returns:
            A 2-tuple where the first element is the normalised topic letter
            (A–G) and the second is the raw LLM output (for logging/debugging).
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User question: {question}\n\nCategory:",
            },
        ]
        raw = self.ask_llm(messages, temperature=0.0, max_tokens=4)
        letter = next((c for c in raw.upper() if c in "ABCDEFGH"), "G")
        return letter, raw

    @staticmethod
    def is_in_scope(category: str) -> bool:
        """Return True when *category* is A–F or H (in-scope)."""
        return category != "G"

    @staticmethod
    def section_hint(category: str) -> str | None:
        """Return the retriever section name for *category*, or None."""
        return _SECTION_MAP.get(category)
