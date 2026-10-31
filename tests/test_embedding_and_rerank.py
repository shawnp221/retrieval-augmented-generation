import math

from conftest import make_chunk
from rag.embedding import HashingEmbedder
from rag.rerank import LexicalOverlapReranker


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(64)
    v1, v2 = e.embed_query("refund limit INV-12345"), e.embed_query("refund limit INV-12345")
    assert v1 == v2 and len(v1) == 64 and abs(math.hypot(*v1) - 1) < 1e-9


def test_overlap_reranker_puts_the_answer_first():
    chunks = [make_chunk("x", "holiday calendar"), make_chunk("y", "refund limit is $50 per invoice")]
    out = LexicalOverlapReranker().rerank("refund limit invoice", chunks, top_n=1)
    assert out[0].chunk.chunk_id == "y"
