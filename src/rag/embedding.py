"""Step 2 - embeddings.

`OpenAIEmbedder` is the real implementation. `HashingEmbedder` is a deterministic, offline
bag-of-words stand-in so the rest of the pipeline can be tested without an API key
(it has no semantic understanding - never use it for real retrieval).
"""
from __future__ import annotations

import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

from rag.utils import batched


class Embedder(Protocol):
    dim: int
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str | None,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        batch_size: int = 256,      # 256 x ~500 tokens = ~128k tokens, under the per-request cap
        max_workers: int = 8,
    ) -> None:
        import openai
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

        self.model, self.dim = model, dim
        self.batch_size, self.max_workers = batch_size, max_workers
        self._client = openai.OpenAI(api_key=api_key, max_retries=0)  # retries handled below
        retryable = (
            openai.RateLimitError, openai.APIConnectionError,
            openai.APITimeoutError, openai.InternalServerError,
        )
        self._call = retry(
            retry=retry_if_exception_type(retryable),
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(8),
            reraise=True,
        )(self._request)

    def _request(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts, dimensions=self.dim)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batches = list(batched(texts, self.batch_size))
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = pool.map(self._call, batches)      # map preserves input order
            return [vec for batch in results for vec in batch]

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]


class HashingEmbedder:
    _TOKEN = re.compile(r"[a-z0-9$\-]+")

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
