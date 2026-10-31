"""Glue for steps 1-3: chunks -> embeddings -> both indexes."""
from __future__ import annotations

import logging
import time
from typing import Iterable

from rag.embedding import Embedder
from rag.models import Chunk
from rag.stores.base import KeywordStore, VectorStore
from rag.utils import batched

log = logging.getLogger(__name__)


class Indexer:
    """Streams chunks in large batches so embedding calls can run concurrently.

    Writes are idempotent (deterministic chunk IDs), so a crashed run can simply be
    restarted. Elasticsearch is written first: if the vector write then fails, the chunk
    is still keyword-searchable and the retry repairs the gap.
    """

    def __init__(
        self, embedder: Embedder, vector_store: VectorStore, keyword_store: KeywordStore,
        batch_size: int = 2048,
    ) -> None:
        self.embedder, self.vectors, self.keywords = embedder, vector_store, keyword_store
        self.batch_size = batch_size

    def index(self, chunks: Iterable[Chunk], *, bulk: bool = False) -> int:
        if bulk:
            self.vectors.set_bulk_mode(True)
            self.keywords.set_bulk_mode(True)
        total, started = 0, time.monotonic()
        try:
            for batch in batched(chunks, self.batch_size):
                embeddings = self.embedder.embed_documents([c.text for c in batch])
                self.keywords.index(batch)
                self.vectors.upsert(batch, embeddings)
                total += len(batch)
                rate = total / max(time.monotonic() - started, 1e-9)
                log.info("indexed %d chunks (%.0f chunks/s)", total, rate)
        finally:
            if bulk:
                self.vectors.set_bulk_mode(False)
                self.keywords.set_bulk_mode(False)
            self.keywords.refresh()
        return total
