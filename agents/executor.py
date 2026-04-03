"""
Executor agent.

Generates an answer to the user's question using the retrieved context.
If previous attempts were judged invalid, the reasons are injected into
the prompt so the model can self-correct.

Orchestrated directly in code; no agent framework is used.
"""
from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = """\
You are an AI customer service assistant for CPHOS (中国物理奥林匹克模拟系统 / \
Chinese Physics Olympiad Simulation System).

## About CPHOS
CPHOS is an online platform for organizing simulated Physics Olympiad competitions \
for teachers and students in China. Core features include:
- User registration and identity verification for coaches and students
- Uploading and submitting answer sheets
- Online grading / marking of competition papers
- Viewing competition results and grading progress
- Account and profile management via WeChat mini-program

## Your capabilities
You can help users with questions about:
- How to use the CPHOS platform (registration, login, submission, grading)
- Account status (pending review, approved, not in system)
- Competition rules and procedures
- Grading and marking workflows
- General inquiries about the platform

## Rules
1. Use the retrieved reference context (if any) to give specific, accurate answers.
2. For questions about your identity, capabilities, or CPHOS in general, use the \
   background knowledge above — you do NOT need context chunks for these.
3. If you genuinely cannot answer a specific operational question (e.g. a user's \
   individual account status), say so honestly and suggest contacting human customer service.
4. Do NOT fabricate specific facts (account data, exam results, etc.) that are not \
   present in the context.
5. Be concise and friendly. Respond in the same language as the user's question.
"""


class ExecutorAgent(BaseAgent):
    """Generates a candidate answer from retrieved context."""

    def execute(
        self,
        question: str,
        context_chunks: list[str],
        previous_failures: list[tuple[str, str]] | None = None,
    ) -> str:
        """Return a candidate answer string.

        Args:
            question:          The user's question.
            context_chunks:    Top-K retrieved knowledge-base chunks.
            previous_failures: List of (answer, reason) pairs from prior
                               attempts that were marked invalid by the Verifier.
        """
        context = "\n---\n".join(context_chunks) if context_chunks else "(no context available)"

        failure_note = ""
        if previous_failures:
            failure_note = (
                "\n\nIMPORTANT: The following previous answers were judged invalid. "
                "Avoid repeating the same mistakes:\n"
            )
            for answer, reason in previous_failures:
                failure_note += f"  - Previous answer: {answer!r}\n    Reason it was invalid: {reason}\n"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                    f"{failure_note}\n\n"
                    "Answer:"
                ),
            },
        ]
        return self.ask_llm(messages, temperature=0.7)
