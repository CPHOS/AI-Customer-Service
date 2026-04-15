"""
Executor agent.

Generates an answer to the user's question using the retrieved context.
If previous attempts were judged invalid, the reasons are injected into
the prompt so the model can self-correct.

Orchestrated directly in code; no agent framework is used.
"""
from __future__ import annotations

from agents.base import BaseAgent
from utils.web_fetch import fetch_page as _fetch_page, ALLOWED_PAGES

_SYSTEM_PROMPT = """\
You are an AI customer service assistant for CPHOS (Chinese Physics Olympiad S).

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


# ── Web-fetch dedicated prompt (used only for category H) ────────────────────

_WEB_SYSTEM_PROMPT = """\
You are an AI customer service assistant for CPHOS (Chinese Physics Olympiad S).

The user is asking a time-sensitive question whose answer must come from the
CPHOS official website.  You MUST call the `fetch_page` tool to retrieve the
relevant page before answering.

Available pages you can fetch:
  - notification  — 联考通知 (exam/competition announcements)
  - organization  — 关于我们 (about CPHOS)
  - resource      — 资料下载 (downloadable materials)
  - events        — 往期活动 (past events and activities)

Strategy:
1. Decide which page(s) are most likely to contain the answer.
2. Call `fetch_page` with the appropriate page_key.
3. Read the returned content carefully.
4. If the first page does not contain the answer, try another relevant page.
5. Compose a clear, concise answer based on the fetched content.
6. Respond in the same language as the user's question.

Do NOT fabricate any dates, names, or specifics. Only use information from
the fetched page content.
"""

# ── Tool definition (OpenAI function-calling spec) ───────────────────────────

_page_keys_desc = ", ".join(f'"{k}" ({v})' for k, v in ALLOWED_PAGES.items())

_FETCH_PAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch the plain-text content of a CPHOS official web page. "
            "Only call this when the provided knowledge-base context is "
            "insufficient to answer the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page_key": {
                    "type": "string",
                    "enum": list(ALLOWED_PAGES.keys()),
                    "description": (
                        "Which CPHOS page to fetch. Available: "
                        + _page_keys_desc
                    ),
                },
            },
            "required": ["page_key"],
        },
    },
}

_TOOLS = [_FETCH_PAGE_TOOL]

# ── Tool executor (maps tool name → local function) ──────────────────────

def _run_fetch_page(args: dict) -> str:
    page_key = args.get("page_key", "")
    return _fetch_page(page_key)

_TOOL_EXECUTOR = {"fetch_page": _run_fetch_page}


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

    def execute_with_web(
        self,
        question: str,
    ) -> str:
        """Answer a time-sensitive question by fetching CPHOS web pages.

        Unlike :meth:`execute`, this method does NOT receive RAG context.
        Instead it provides the LLM with the ``fetch_page`` tool so it can
        pull live data from the CPHOS website.
        """
        messages = [
            {"role": "system", "content": _WEB_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Use the fetch_page tool to find the answer on the CPHOS "
                    "official website, then reply to the user."
                ),
            },
        ]
        return self.ask_llm_with_tools(
            messages,
            tools=_TOOLS,
            tool_executor=_TOOL_EXECUTOR,
            temperature=0.7,
        )
