"""Steps 4-5 - filtered hybrid search with Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from rag.access import User, can_see, visible_teams
from rag.embedding import Embedder
from rag.models import Chunk
from rag.stores.base import KeywordStore, VectorStore


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, top_n: int | None = None
) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1.

    Only ranks matter, so BM25 scores and cosine similarities (which live on different
    scales) can be combined without normalisation. A chunk near the top of BOTH lists
    outranks one that is first in only one. Ties keep first-seen order (stable sort).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_n] if top_n else fused


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    rrf_score: float
    vector_rank: int | None    # 1-based rank in the vector list, None if absent
    keyword_rank: int | None


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        *,
        top_k: int = 50,
        rrf_k: int = 60,
        max_candidates: int = 100,
    ) -> None:
        self.embedder, self.vectors, self.keywords = embedder, vector_store, keyword_store
        self.top_k, self.rrf_k, self.max_candidates = top_k, rrf_k, max_candidates
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _vector_search(self, query: str, teams: list[str]):
        return self.vectors.search(self.embedder.embed_query(query), teams, self.top_k)

    def retrieve(self, query: str, user: User) -> list[Candidate]:
        teams = visible_teams(user)                       # step 4: filter...
        if not query.strip():
            return []
        # ...pushed down into BOTH databases, run in parallel (step 5)
        vec_future = self._pool.submit(self._vector_search, query, teams)
        kw_future = self._pool.submit(self.keywords.search, query, teams, self.top_k)
        vec_ids = [h.chunk_id for h in vec_future.result()]
        kw_ids = [h.chunk_id for h in kw_future.result()]

        fused = reciprocal_rank_fusion([vec_ids, kw_ids], k=self.rrf_k, top_n=self.max_candidates)
        chunks = self.keywords.fetch([cid for cid, _ in fused])

        vec_rank = {cid: i for i, cid in enumerate(vec_ids, 1)}
        kw_rank = {cid: i for i, cid in enumerate(kw_ids, 1)}
        return [
            Candidate(chunks[cid], score, vec_rank.get(cid), kw_rank.get(cid))
            for cid, score in fused
            if cid in chunks and can_see(chunks[cid], user)   # defence in depth
        ]
