"""
Embedding-based document retriever.

Uses OpenAI's Embeddings API to encode documents and queries, then finds
the nearest neighbours via cosine similarity computed in-memory with NumPy.

Supports *section-tagged* indexing: each chunk can be tagged with a section
name (e.g. "marking", "identity_change") at index time.  At query time an
optional ``section_hint`` boosts scores for chunks in that section, so the
most topically relevant knowledge surfaces first without completely excluding
other sections (graceful degradation when the section is sparse).

No external vector database or third-party retrieval framework is used.

Third-party dependencies introduced here:
    - openai  (official OpenAI Python SDK — embeddings endpoint)
    - numpy   (vector arithmetic for cosine similarity)
"""
from __future__ import annotations

import numpy as np
import openai

# Multiplicative score boost applied to chunks whose section matches the hint.
# 1.2 means "20 % preference" — strong enough to surface matching chunks while
# still allowing high-quality chunks from other sections to rank above poor
# section-matches.
_SECTION_BOOST = 1.2


class Retriever:
    """In-memory embedding retriever backed by OpenAI Embeddings API.

    Usage::

        retriever = Retriever(api_key="sk-...", embedding_model="text-embedding-3-small")
        # For OpenRouter, also pass: base_url="https://openrouter.ai/api/v1"
        retriever.add_documents(chunks, section="marking")  # index with section tag
        top_chunks = retriever.query(question, section_hint="marking")  # boosted retrieval
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._client         = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or {},
        )
        self.embedding_model = embedding_model
        self._chunks:     list[str]            = []
        self._sections:   list[str]            = []        # parallel to _chunks
        self._embeddings: np.ndarray | None    = None  # shape (N, D)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_documents(
        self,
        chunks:     list[str],
        batch_size: int = 100,
        section:    str = "default",
    ) -> None:
        """Embed *chunks* and append them to the in-memory index.

        Args:
            chunks:     List of text chunks to index.
            batch_size: Number of texts to embed per API call.
            section:    Section tag for all chunks in this batch (e.g. the
                        source filename stem).  Used for section-hint boosting
                        at query time.
        """
        if not chunks:
            return
        new_vecs = self._embed(chunks, batch_size)
        self._chunks.extend(chunks)
        self._sections.extend([section] * len(chunks))
        if self._embeddings is None:
            self._embeddings = new_vecs
        else:
            self._embeddings = np.vstack([self._embeddings, new_vecs])

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Call the Embeddings API in batches; return a float32 matrix."""
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch    = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            all_vecs.extend(item.embedding for item in response.data)
        return np.array(all_vecs, dtype=np.float32)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        text:         str,
        top_k:        int = 5,
        section_hint: str | None = None,
    ) -> list[str]:
        """Return the *top_k* chunks most similar to *text*.

        Args:
            text:         Query string.
            top_k:        Maximum number of chunks to return.
            section_hint: When provided, chunks from this section receive a
                          score boost (``_SECTION_BOOST``), so they rank higher
                          than same-similarity chunks from other sections.
                          Falls back gracefully to global results when the
                          section has too few chunks.

        Returns an empty list if the index is empty.
        """
        if self._embeddings is None or not self._chunks:
            return []

        response  = self._client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)

        scores = self._cosine_similarity(query_vec, self._embeddings)

        # Apply section boost when a hint is provided
        if section_hint and self._sections:
            boost = np.array(
                [_SECTION_BOOST if s == section_hint else 1.0 for s in self._sections],
                dtype=np.float32,
            )
            scores = scores * boost

        top_k   = min(top_k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self._chunks[i] for i in top_idx]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between one vector and a matrix of row vectors."""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        m_norm = matrix   / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return m_norm @ q_norm

    # ── Persistence helpers (optional) ────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the index to a .npz file."""
        if self._embeddings is None:
            raise ValueError("Index is empty; nothing to save.")
        np.savez_compressed(
            path,
            embeddings=self._embeddings,
            chunks=np.array(self._chunks, dtype=object),
            sections=np.array(self._sections, dtype=object),
        )

    def load(self, path: str) -> None:
        """Load a previously saved index from a .npz file."""
        data             = np.load(path, allow_pickle=True)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._chunks     = list(data["chunks"])
        # Backward-compatible: old indexes saved without section tags.
        if "sections" in data:
            self._sections = list(data["sections"])
        else:
            self._sections = ["default"] * len(self._chunks)
