"""STEP 6 - rerank the fused candidates. Needs COHERE_API_KEY, or RERANKER_BACKEND=bge|overlap.

    python scripts/06_rerank.py
"""
import argparse

from _common import DEFAULT_QUESTION, snippet, user_from_team
from rag.config import Settings
from rag.factory import build_reranker, build_retriever

p = argparse.ArgumentParser()
p.add_argument("query", nargs="?", default=DEFAULT_QUESTION)
p.add_argument("--team", default="sales")
args = p.parse_args()

s = Settings.from_env()
candidates = build_retriever(s).retrieve(args.query, user_from_team(args.team))
order_before = {c.chunk.chunk_id: i for i, c in enumerate(candidates, 1)}
reranked = build_reranker(s).rerank(args.query, [c.chunk for c in candidates], s.final_top_k)

print(f"query: {args.query!r}\n\n{'new#':>4} {'was#':>5} {'score':>7}  title / text")
for i, r in enumerate(reranked, 1):
    print(f"{i:>4} {order_before[r.chunk.chunk_id]:>5} {r.score:7.3f}  "
          f"{r.chunk.title:30} {snippet(r.chunk.text, 45)}")
