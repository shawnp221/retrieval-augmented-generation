"""STEP 2 - embeddings. Needs OPENAI_API_KEY (or EMBEDDER_BACKEND=fake).

    python scripts/02_embedding.py
"""
import math

from rag.config import Settings
from rag.factory import build_embedder

settings = Settings.from_env()
embedder = build_embedder(settings)

texts = [
    "Sales reps may approve refunds of up to $50 per invoice.",
    "The money-back policy lets representatives return up to fifty dollars.",
    "Employees receive 16 weeks of paid parental leave.",
]
vecs = embedder.embed_documents(texts)
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))

print(f"backend={settings.embedder_backend} model={settings.embedding_model} dim={len(vecs[0])}")
print(f"refund vs money-back (similar meaning): {cos(vecs[0], vecs[1]):.3f}")
print(f"refund vs parental leave (unrelated)  : {cos(vecs[0], vecs[2]):.3f}")
assert len(vecs[0]) == settings.embedding_dim
