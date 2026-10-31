from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    cohere_api_key: str | None

    embedder_backend: str          # openai | fake
    embedding_model: str
    embedding_dim: int

    reranker_backend: str          # cohere | bge | overlap
    cohere_rerank_model: str
    bge_rerank_model: str
    llm_model: str

    qdrant_url: str
    qdrant_collection: str
    es_url: str
    es_index: str
    es_shards: int
    es_replicas: int

    chunk_max_tokens: int
    retrieval_top_k: int           # per retriever (vector, keyword)
    rrf_k: int
    final_top_k: int               # chunks sent to the LLM

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:  # pragma: no cover
            pass
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            cohere_api_key=os.environ.get("COHERE_API_KEY") or None,
            embedder_backend=_env("EMBEDDER_BACKEND", "openai"),
            embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=int(_env("EMBEDDING_DIM", "1536")),
            reranker_backend=_env("RERANKER_BACKEND", "cohere"),
            cohere_rerank_model=_env("COHERE_RERANK_MODEL", "rerank-v3.5"),
            bge_rerank_model=_env("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
            qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=_env("QDRANT_COLLECTION", "chunks"),
            es_url=_env("ES_URL", "http://localhost:9200"),
            es_index=_env("ES_INDEX", "chunks"),
            es_shards=int(_env("ES_SHARDS", "1")),
            es_replicas=int(_env("ES_REPLICAS", "0")),
            chunk_max_tokens=int(_env("CHUNK_MAX_TOKENS", "500")),
            retrieval_top_k=int(_env("RETRIEVAL_TOP_K", "50")),
            rrf_k=int(_env("RRF_K", "60")),
            final_top_k=int(_env("FINAL_TOP_K", "5")),
        )
