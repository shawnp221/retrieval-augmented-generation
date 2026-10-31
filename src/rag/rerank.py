"""Step 6 - cross-encoder reranking.

A bi-encoder (embeddings) scores query and chunk independently; a cross-encoder reads them
TOGETHER, which is far more accurate but too slow for 100M chunks - hence we only rerank
the ~100 fused candidates.
"""
from __future__ import annotations

import re
from typing import Protocol, Sequence

from rag.models import Chunk, ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int) -> list[ScoredChunk]: ...


class CohereReranker:
    def __init__(self, api_key: str | None, model: str = "rerank-v3.5") -> None:
        import cohere
        self.model = model
        self._client = cohere.ClientV2(api_key=api_key)

    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        resp = self._client.rerank(
            model=self.model, query=query, documents=[c.text for c in chunks], top_n=top_n
        )
        return [ScoredChunk(chunks[r.index], r.relevance_score) for r in resp.results]


class BGEReranker:
    """Open-source option, runs locally (GPU recommended)."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
        batch_size: int = 16, max_length: int = 1024,
    ) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks], batch_size=self.batch_size)
        ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        return [ScoredChunk(c, float(s)) for c, s in ranked[:top_n]]


class LexicalOverlapReranker:
    """Offline stand-in for tests/demos: fraction of query terms present in the chunk."""
    _TOKEN = re.compile(r"[a-z0-9$\-]+")

    def rerank(self, query: str, chunks: Sequence[Chunk], top_n: int) -> list[ScoredChunk]:
        q = set(self._TOKEN.findall(query.lower()))
        scored = [
            ScoredChunk(c, len(q & set(self._TOKEN.findall(c.text.lower()))) / max(len(q), 1))
            for c in chunks
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_n]
