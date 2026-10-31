"""STEP 4 - team filter. Shows raw hits from each store for a given team.

    python scripts/04_filter.py --team sales
    python scripts/04_filter.py --team hr        # same query, different visible chunks
"""
import argparse

from _common import DEFAULT_QUESTION, snippet, user_from_team
from rag.access import can_see, visible_teams
from rag.config import Settings
from rag.factory import build_embedder, build_keyword_store, build_vector_store

p = argparse.ArgumentParser()
p.add_argument("query", nargs="?", default=DEFAULT_QUESTION)
p.add_argument("--team", default="sales")
args = p.parse_args()

s = Settings.from_env()
user, teams = user_from_team(args.team), visible_teams(user_from_team(args.team))
vectors, keywords = build_vector_store(s), build_keyword_store(s)

vec_hits = vectors.search(build_embedder(s).embed_query(args.query), teams, 10)
kw_hits = keywords.search(args.query, teams, 10)
chunks = keywords.fetch([h.chunk_id for h in vec_hits + kw_hits])

print(f"query: {args.query!r}\nteam={args.team}  -> filter allowed_teams IN {teams}\n")
for name, hits in (("VECTOR", vec_hits), ("KEYWORD", kw_hits)):
    print(f"--- {name} ({len(hits)} hits)")
    for h in hits:
        c = chunks[h.chunk_id]
        print(f"  {h.score:8.3f}  {str(list(c.allowed_teams)):12} {c.title:32} {snippet(c.text, 50)}")

assert all(can_see(chunks[h.chunk_id], user) for h in vec_hits + kw_hits)
print("\nOK: no chunk outside the user's teams was returned by either store")
