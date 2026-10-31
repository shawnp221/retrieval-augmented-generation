"""Builds concrete components from Settings (keeps scripts and tests free of wiring)."""
from __future__ import annotations

from rag.config import Settings
from rag.embedding import Embedder, HashingEmbedder, OpenAIEmbedder
from rag.generation import AnswerGenerator, OpenAIChat
from rag.pipeline import RagPipeline
from rag.rerank import BGEReranker, CohereReranker, LexicalOverlapReranker, Reranker
from rag.retrieval import HybridRetriever
from rag.stores.es_store import ElasticKeywordStore
from rag.stores.qdrant_store import QdrantVectorStore


def build_embedder(s: Settings) -> Embedder:
    if s.embedder_backend == "fake":
        return HashingEmbedder(s.embedding_dim)
    return OpenAIEmbedder(s.openai_api_key, s.embedding_model, s.embedding_dim)


def build_vector_store(s: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(s.qdrant_url, s.qdrant_collection, s.embedding_dim)


def build_keyword_store(s: Settings) -> ElasticKeywordStore:
    return ElasticKeywordStore(s.es_url, s.es_index, shards=s.es_shards, replicas=s.es_replicas)


def build_reranker(s: Settings) -> Reranker:
    if s.reranker_backend == "bge":
        return BGEReranker(s.bge_rerank_model)
    if s.reranker_backend == "overlap":
        return LexicalOverlapReranker()
    return CohereReranker(s.cohere_api_key, s.cohere_rerank_model)


def build_retriever(s: Settings) -> HybridRetriever:
    return HybridRetriever(
        build_embedder(s), build_vector_store(s), build_keyword_store(s),
        top_k=s.retrieval_top_k, rrf_k=s.rrf_k, max_candidates=2 * s.retrieval_top_k,
    )


def build_pipeline(s: Settings) -> RagPipeline:
    return RagPipeline(
        build_retriever(s), build_reranker(s),
        AnswerGenerator(OpenAIChat(s.openai_api_key, s.llm_model)),
        final_top_k=s.final_top_k,
    )
