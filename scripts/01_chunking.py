"""STEP 1 - chunking. Needs no services and no API keys.

    python scripts/01_chunking.py
    python scripts/01_chunking.py --file my.txt --max-tokens 500
"""
import argparse

from _common import DATA
from rag.chunking import SentenceChunker
from rag.io import read_documents
from rag.models import Document

p = argparse.ArgumentParser()
p.add_argument("--file", help="plain-text file to chunk (default: sample docs repeated)")
p.add_argument("--max-tokens", type=int, default=500)
p.add_argument("--overlap", type=int, default=0)
args = p.parse_args()

if args.file:
    text = open(args.file, encoding="utf-8").read()
else:
    base = " ".join(d.text for d in read_documents(DATA))
    text = " ".join([base] * 12)          # long enough to produce several chunks

chunker = SentenceChunker(max_tokens=args.max_tokens, overlap_sentences=args.overlap)
chunks = chunker.chunk(Document(doc_id="demo", text=text))

print(f"{len(chunks)} chunks (max_tokens={args.max_tokens})\n")
for c in chunks:
    print(f"#{c.position:<3} tokens={c.token_count:<4} start={c.text[:45]!r:50} end={c.text[-30:]!r}")

assert all(c.token_count <= args.max_tokens for c in chunks), "token budget exceeded"
assert all(c.text.rstrip()[-1] in ".!?\"')]" for c in chunks[:-1]), "a chunk ends mid-sentence"
print("\nOK: every chunk respects the token budget and ends on a sentence boundary")
