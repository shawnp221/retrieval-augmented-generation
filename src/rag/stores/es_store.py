"""Step 3b - keyword index (Elasticsearch, BM25).

`text.exact` uses a whitespace tokenizer + lowercase + edge-punctuation stripping so
identifiers like "INV-12345" survive as ONE token (the standard analyzer would split them
into "inv" and "12345"). Elasticsearch is also the source of truth for chunk text; the
vector DB only returns IDs.
"""
from __future__ import annotations

from typing import Sequence

from elasticsearch import Elasticsearch, helpers

from rag.models import Chunk, Hit


class ElasticKeywordStore:
    def __init__(
        self,
        url: str,
        index: str,
        *,
        shards: int = 1,
        replicas: int = 0,
        client: Elasticsearch | None = None,
    ) -> None:
        self.index_name, self.shards, self.replicas = index, shards, replicas
        self._es = client or Elasticsearch(url, request_timeout=60, max_retries=3, retry_on_timeout=True)

    # ------------------------------------------------------------------ lifecycle
    def ensure_index(self, recreate: bool = False) -> None:
        exists = self._es.indices.exists(index=self.index_name)
        if exists and recreate:
            self._es.indices.delete(index=self.index_name)
            exists = False
        if exists:
            return
        self._es.indices.create(
            index=self.index_name,
            settings={
                "number_of_shards": self.shards,
                "number_of_replicas": self.replicas,
                "refresh_interval": "5s",
                "analysis": {
                    "filter": {
                        "strip_edge_punct": {
                            "type": "pattern_replace",
                            "pattern": "^[^\\p{L}\\p{N}]+|[^\\p{L}\\p{N}]+$",
                            "replacement": "",
                        },
                        "non_empty": {"type": "length", "min": 1},
                    },
                    "analyzer": {
                        "id_aware": {
                            "type": "custom",
                            "tokenizer": "whitespace",
                            "filter": ["lowercase", "strip_edge_punct", "non_empty"],
                        }
                    },
                },
            },
            mappings={
                "dynamic": "strict",
                "properties": {
                    "text": {"type": "text", "fields": {"exact": {"type": "text", "analyzer": "id_aware"}}},
                    "title": {"type": "text"},
                    "doc_id": {"type": "keyword"},
                    "source_url": {"type": "keyword", "index": False},
                    "allowed_teams": {"type": "keyword"},
                    "position": {"type": "integer"},
                    "token_count": {"type": "integer"},
                },
            },
        )

    def set_bulk_mode(self, enabled: bool) -> None:
        self._es.indices.put_settings(
            index=self.index_name,
            settings={"index": {
                "refresh_interval": "-1" if enabled else "5s",
                "number_of_replicas": 0 if enabled else self.replicas,
            }},
        )

    def refresh(self) -> None:
        self._es.indices.refresh(index=self.index_name)

    def count(self) -> int:
        return self._es.count(index=self.index_name)["count"]

    # ------------------------------------------------------------------ write
    def index(self, chunks: Sequence[Chunk]) -> None:
        actions = (
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": c.chunk_id,
                "_source": {
                    "text": c.text, "title": c.title, "doc_id": c.doc_id,
                    "source_url": c.source_url, "allowed_teams": list(c.allowed_teams),
                    "position": c.position, "token_count": c.token_count,
                },
            }
            for c in chunks
        )
        helpers.bulk(self._es, actions, chunk_size=1000, raise_on_error=True)

    # ------------------------------------------------------------------ read
    def search(self, query: str, teams: Sequence[str], limit: int) -> list[Hit]:
        resp = self._es.search(
            index=self.index_name,
            size=limit,
            source=False,
            query={"bool": {
                "filter": [{"terms": {"allowed_teams": list(teams)}}],
                "must": [{"multi_match": {
                    "query": query,
                    "fields": ["text", "text.exact^2", "title^2"],
                    "type": "most_fields",
                }}],
            }},
        )
        return [Hit(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]

    def fetch(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        resp = self._es.mget(index=self.index_name, ids=list(chunk_ids))
        out: dict[str, Chunk] = {}
        for doc in resp["docs"]:
            if not doc.get("found"):
                continue
            s = doc["_source"]
            out[doc["_id"]] = Chunk(
                chunk_id=doc["_id"], doc_id=s["doc_id"], position=s["position"],
                text=s["text"], token_count=s["token_count"], title=s.get("title", ""),
                source_url=s.get("source_url", ""), allowed_teams=tuple(s["allowed_teams"]),
            )
        return out
