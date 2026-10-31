from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from rag.models import Document


def read_documents(path: str | Path) -> Iterator[Document]:
    """Stream documents from a JSONL file (one object per line) without loading it all.

    Expected keys: doc_id, text, and optionally title, source_url, allowed_teams (list).
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            teams = tuple(t.strip().lower() for t in row.get("allowed_teams", ["all"]))
            yield Document(
                doc_id=row["doc_id"],
                text=row["text"],
                title=row.get("title", ""),
                source_url=row.get("source_url", ""),
                allowed_teams=teams,
            )
