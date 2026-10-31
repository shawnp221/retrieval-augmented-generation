"""Step 3a - vector index (Qdrant).

Memory layout (100M chunks x 1536 dims):
  * original float32 vectors   ~614 GB  -> on DISK   (vectors on_disk=True)
  * binary-quantized vectors   ~19 GB   -> in RAM    (always_ram=True, 1 bit per dim)
Search flow: Hamming-distance scan over the RAM copy -> oversample candidates ->
rescore those few with the full-precision vectors read from disk.

The payload only holds what filtering needs (allowed_teams, doc_id). Chunk text lives in
Elasticsearch, so we do not store ~250 GB of text twice.
"""
from __future__ import annotations

from typing import Sequence

from qdrant_client import QdrantClient, models

from rag.models import Chunk, Hit


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        dim: int,
        *,
        hnsw_ef: int = 128,
        oversampling: float = 3.0,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection, self.dim = collection, dim
        self.hnsw_ef, self.oversampling = hnsw_ef, oversampling
        self._client = client or QdrantClient(url=url, prefer_grpc=True, timeout=60)

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self._client.collection_exists(self.collection)
        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE, on_disk=True
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            on_disk_payload=True,
        )
        # Create the payload index BEFORE loading data so HNSW is built filter-aware.
        self._client.create_payload_index(
            self.collection, field_name="allowed_teams",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=list(v),
                payload={"doc_id": c.doc_id, "allowed_teams": list(c.allowed_teams)},
            )
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: Sequence[float], teams: Sequence[str], limit: int) -> list[Hit]:
        result = self._client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=models.Filter(must=[
                models.FieldCondition(key="allowed_teams", match=models.MatchAny(any=list(teams)))
            ]),
            limit=limit,
            search_params=models.SearchParams(
                hnsw_ef=self.hnsw_ef,
                quantization=models.QuantizationSearchParams(
                    rescore=True, oversampling=self.oversampling
                ),
            ),
            with_payload=False,
            with_vectors=False,
        )
        return [Hit(str(p.id), p.score) for p in result.points]

    def set_bulk_mode(self, enabled: bool) -> None:
        """During a bulk load, defer HNSW building; afterwards let the optimizer build it."""
        self._client.update_collection(
            self.collection,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0 if enabled else 20_000),
        )

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count
