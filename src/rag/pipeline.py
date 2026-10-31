from __future__ import annotations

from dataclasses import dataclass

from rag.access import User
from rag.generation import Answer, AnswerGenerator
from rag.models import ScoredChunk
from rag.rerank import Reranker
from rag.retrieval import Candidate, HybridRetriever


@dataclass(frozen=True)
class RagResult:
    answer: Answer
    candidates: list[Candidate]      # after filter + hybrid search + RRF (<= 100)
    reranked: list[ScoredChunk]      # the top-N actually sent to the LLM


class RagPipeline:
    def __init__(
        self, retriever: HybridRetriever, reranker: Reranker, generator: AnswerGenerator,
        final_top_k: int = 5,
    ) -> None:
        self.retriever, self.reranker, self.generator = retriever, reranker, generator
        self.final_top_k = final_top_k

    def ask(self, question: str, user: User) -> RagResult:
        candidates = self.retriever.retrieve(question, user)
        reranked = self.reranker.rerank(question, [c.chunk for c in candidates], self.final_top_k)
        answer = self.generator.answer(question, [s.chunk for s in reranked])
        return RagResult(answer, candidates, reranked)
