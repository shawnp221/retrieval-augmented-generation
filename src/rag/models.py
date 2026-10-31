"""Plain data types shared by every pipeline stage."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

_NAMESPACE = uuid.UUID("6f1c2b8e-5d0a-4c3e-9b7a-1f2e3d4c5b6a")


def make_chunk_id(doc_id: str, position: int) -> str:
    """Deterministic ID -> re-ingesting the same document overwrites instead of duplicating.

    A UUID is used because Qdrant point IDs must be ints or UUIDs; the same ID is the
    Elasticsearch `_id`, which is what lets us join the two indexes.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{position}"))


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    source_url: str = ""
    allowed_teams: tuple[str, ...] = ("all",)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    position: int
    text: str
    token_count: int
    title: str = ""
    source_url: str = ""
    allowed_teams: tuple[str, ...] = ("all",)


@dataclass(frozen=True, slots=True)
class Hit:
    """One result of a single retriever (vector OR keyword), already ordered by rank."""
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
