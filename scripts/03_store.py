"""STEP 3 - create both indexes and load the sample documents.
Needs `docker compose up -d` (+ OPENAI_API_KEY or EMBEDDER_BACKEND=fake).

    python scripts/03_store.py --reset
"""
import argparse

from _common import DATA
from rag.chunking import chunk_documents
from rag.config import Settings
from rag.factory import build_embedder, build_keyword_store, build_vector_store
from rag.indexing import Indexer
from rag.io import read_documents

p = argparse.ArgumentParser()
p.add_argument("--reset", action="store_true", help="drop and recreate collection/index")
args = p.parse_args()

s = Settings.from_env()
vectors, keywords = build_vector_store(s), build_keyword_store(s)
vectors.ensure_collection(recreate=args.reset)
keywords.ensure_index(recreate=args.reset)

chunks = chunk_documents(read_documents(DATA), max_tokens=s.chunk_max_tokens)
n = Indexer(build_embedder(s), vectors, keywords).index(chunks)

print(f"\nindexed {n} chunks | qdrant={vectors.count()} | elasticsearch={keywords.count()}")
assert vectors.count() == keywords.count() == n, "the two indexes are out of sync"
print("OK: both indexes hold the same chunk IDs")
