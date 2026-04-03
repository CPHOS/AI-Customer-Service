"""
Central configuration for the AI Customer Service system.
All values can be overridden via environment variables or a .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM backend settings ─────────────────────────────────────────────────────
OPENAI_API_KEY: str       = os.environ.get("OPENROUTER_API_KEY", "")
LLM_BASE_URL:   str | None = os.environ.get("OPENROUTER_BASE_URL") or None

# ── Model selection (per-agent) ───────────────────────────────────────────────
# Cheap model for routing, criticism & verification; heavier model for answer generation.
CLASSIFIER_MODEL: str = os.environ.get("OPENROUTER_CLASSIFIER_MODEL", "openai/gpt-4o-mini")
EXECUTOR_MODEL:   str = os.environ.get("OPENROUTER_EXECUTOR_MODEL",   "openai/gpt-4o")
VERIFIER_MODEL:   str = os.environ.get("OPENROUTER_VERIFIER_MODEL",   "openai/gpt-4o-mini")
CRITIC_MODEL:     str = os.environ.get("OPENROUTER_CRITIC_MODEL",     "openai/gpt-4o-mini")
EMBEDDING_MODEL:  str = os.environ.get("OPENROUTER_EMBEDDING_MODEL",  "openai/text-embedding-3-small")

# ── RAG settings ──────────────────────────────────────────────────────────────
CHUNK_WORD_LENGTH: int = int(os.environ.get("OPENROUTER_CHUNK_WORD_LENGTH", "150"))
TOP_K_CHUNKS:      int = int(os.environ.get("OPENROUTER_TOP_K_CHUNKS",      "5"))

# ── Pipeline settings ─────────────────────────────────────────────────────────
MAX_RETRIES: int = int(os.environ.get("OPENROUTER_MAX_RETRIES", "3"))

# When True, run two retrieval paths (section-hinted + general) in parallel and
# use the Critic to select the better answer before the Verifier validates it.
# Set to "false" to disable and use the simpler single-path flow.
ENABLE_DUAL_PATH: bool = os.environ.get("OPENROUTER_ENABLE_DUAL_PATH", "true").lower() != "false"
