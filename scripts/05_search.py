"""STEP 5 - hybrid search + RRF. Shows where each chunk ranked in each list.

    python scripts/05_search.py
"""
import argparse

from _common import DEFAULT_QUESTION, snippet, user_from_team
from rag.config import Settings
from rag.factory import build_retriever

p = argparse.ArgumentParser()
p.add_argument("query", nargs="?", default=DEFAULT_QUESTION)
p.add_argument("--team", default="sales")
p.add_argument("--show", type=int, default=10)
args = p.parse_args()

candidates = build_retriever(Settings.from_env()).retrieve(args.query, user_from_team(args.team))

print(f"query: {args.query!r}  team={args.team}  candidates={len(candidates)}\n")
print(f"{'rrf':>7} {'vec#':>5} {'kw#':>5}  title / text")
for c in candidates[: args.show]:
    print(f"{c.rrf_score:7.4f} {str(c.vector_rank or '-'):>5} {str(c.keyword_rank or '-'):>5}  "
          f"{c.chunk.title:30} {snippet(c.chunk.text, 45)}")
