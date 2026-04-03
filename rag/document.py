"""
Document loading and chunking utilities.

Supported formats: .txt, .md, .pdf (PDF requires PyMuPDF), .yml/.yaml.

YAML format (from CPHOS_AIReplyer_Playground knowledge base):
    content:
      - Q: "..."
        A: "..."
Each Q&A pair is treated as one self-contained chunk, which gives much
better retrieval precision than fixed-size word splitting.

Third-party dependencies introduced here:
    - PyMuPDF / fitz  (optional — only needed for PDF loading)
      Install: pip install PyMuPDF
    - PyYAML / yaml   (optional — only needed for YAML loading)
      Install: pip install PyYAML
"""
from __future__ import annotations

import os
import re
from typing import List


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(text: str, word_length: int = 150) -> List[str]:
    """Split *text* into fixed-size word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), word_length):
        chunk = " ".join(words[i : i + word_length]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ── Per-format loaders ────────────────────────────────────────────────────────

def load_text_file(path: str, word_length: int = 150) -> List[str]:
    """Load a plain-text or Markdown file and return word chunks."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return _split_into_chunks(_clean(text), word_length)


def load_pdf_file(path: str, word_length: int = 150) -> List[str]:
    """Load a PDF file and return word chunks.

    Requires PyMuPDF (``pip install PyMuPDF``).
    Raises ImportError with a friendly message if the package is absent.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required to load PDF files. "
            "Install it with:  pip install PyMuPDF"
        ) from exc

    doc = fitz.open(path)
    pages: List[str] = []
    for page in doc:
        text = page.get_text("text")
        cleaned = _clean(text)
        if cleaned:
            pages.append(cleaned)
    doc.close()

    full_text = " ".join(pages)
    return _split_into_chunks(full_text, word_length)


def load_yaml_file(path: str) -> List[str]:
    """Load a YAML Q&A knowledge base and return one chunk per Q&A pair.

    Expected YAML structure::

        content:
          - Q: "How do I submit my answer sheet?"
            A: "Open the mini-program, tap Submit…"

    Each pair is formatted as ``"Q: <question>\\nA: <answer>"`` so the
    embedding captures both the question and its answer for accurate retrieval.

    Requires PyYAML (``pip install PyYAML``).
    Raises ImportError with a friendly message if the package is absent.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML knowledge-base files. "
            "Install it with:  pip install PyYAML"
        ) from exc

    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            import warnings
            warnings.warn(
                f"Skipping {path!r}: YAML parse error — {exc}",
                stacklevel=2,
            )
            return []

    content = data.get("content", []) if isinstance(data, dict) else []
    chunks: List[str] = []
    for item in content:
        q = str(item.get("Q", "")).strip()
        a = str(item.get("A", "")).strip()
        if q and a:
            chunks.append(f"Q: {q}\nA: {a}")
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def load_documents(paths: List[str], word_length: int = 150) -> List[str]:
    """Load and chunk one or more .txt, .md, .pdf, .yml, or .yaml files.

    Args:
        paths:       List of file paths to load.
        word_length: Target number of words per chunk (ignored for YAML files,
                     which use one chunk per Q&A pair).

    Returns:
        A flat list of text chunks ready for indexing.
    """
    chunks: List[str] = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            chunks.extend(load_pdf_file(path, word_length))
        elif ext in (".txt", ".md"):
            chunks.extend(load_text_file(path, word_length))
        elif ext in (".yml", ".yaml"):
            chunks.extend(load_yaml_file(path))
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}' for file: {path}\n"
                "Supported formats: .txt  .md  .pdf  .yml  .yaml"
            )
    return chunks
