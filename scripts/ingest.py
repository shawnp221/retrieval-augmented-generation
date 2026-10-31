"""Bulk-ingest a JSONL corpus (one {"doc_id","text","title","source_url","allowed_teams"} per line).

    python scripts/ingest.py corpus.jsonl --workers 8 --bulk --reset

Chunking runs in a process pool; embedding is concurrent and rate-limit aware; both stores
are written idempotently, so a crashed run can simply be restarted.
"""
import argparse

from _common import *  # noqa: F401,F403  (configures logging)
from rag.chunking import chunk_documents
from rag.config import Settings
from rag.factory import build_embedder, build_keyword_store, build_vector_store
from rag.indexing import Indexer
from rag.io import read_documents

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--workers", type=int, default=4, help="chunking processes")
p.add_argument("--bulk", action="store_true", help="defer HNSW build / ES refresh during load")
p.add_argument("--reset", action="store_true")
args = p.parse_args()

s = Settings.from_env()
vectors, keywords = build_vector_store(s), build_keyword_store(s)
vectors.ensure_collection(recreate=args.reset)
keywords.ensure_index(recreate=args.reset)

chunks = chunk_documents(read_documents(args.path), max_tokens=s.chunk_max_tokens, workers=args.workers)
n = Indexer(build_embedder(s), vectors, keywords).index(chunks, bulk=args.bulk)
print(f"done: {n} chunks | qdrant={vectors.count()} | elasticsearch={keywords.count()}")
