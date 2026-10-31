"""END-TO-END: filter -> hybrid search -> RRF -> rerank -> grounded answer.

    python scripts/ask.py --team sales "What is the refund limit for invoice INV-12345?"
"""
import argparse

from _common import DEFAULT_QUESTION, user_from_team
from rag.config import Settings
from rag.factory import build_pipeline

p = argparse.ArgumentParser()
p.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
p.add_argument("--team", default="sales")
args = p.parse_args()

result = build_pipeline(Settings.from_env()).ask(args.question, user_from_team(args.team))
print(f"\nQ: {args.question}  (team={args.team})")
print(f"retrieved {len(result.candidates)} candidates -> reranked to {len(result.reranked)}\n")
print(f"A: {result.answer.text}\n")
for c in result.answer.citations:
    print(f"  [{c.index}] {c.title}  {c.source_url}")
print(f"\ngrounded={result.answer.grounded}")
