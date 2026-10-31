"""Step 1 - sentence-aware chunking.

Greedy packing: sentences are appended to the current chunk until adding the next one would
exceed `max_tokens`; then the chunk is closed and a new one starts. A sentence is therefore
never split - with ONE exception: a single sentence longer than `max_tokens` (e.g. a log
line or a table dump) is hard-split on token boundaries, because it cannot fit otherwise.

Token budgets are the sum of per-sentence token counts. Re-tokenising the joined text can
differ by a few tokens at the seams, which is irrelevant against the embedding model's
8191-token limit.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator, Protocol, Sequence

from rag.models import Chunk, Document, make_chunk_id
from rag.utils import batched


# --------------------------------------------------------------------------- tokenizers
class Tokenizer(Protocol):
    def count_many(self, texts: Sequence[str]) -> list[int]: ...
    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]: ...


class TiktokenTokenizer:
    """cl100k_base is the encoding used by OpenAI's text-embedding-3-* models."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(ids) for ids in self._enc.encode_ordinary_batch(list(texts))]

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        ids = self._enc.encode_ordinary(text)
        return [self._enc.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


# --------------------------------------------------------------------------- sentences
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
    "corp", "e.g", "i.e", "fig", "no", "approx", "dept", "u.s", "u.k",
}
_PARAGRAPH = re.compile(r"\n\s*\n")
_CANDIDATE = re.compile(r"[.!?]+[\"')\]]*\s+")
_PRECEDING_WORD = re.compile(r"([A-Za-z.]+)$")
_SENTENCE_STARTERS = set("\"'([{-*\u2022")


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    nxt = text[match.end():match.end() + 1]
    if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in _SENTENCE_STARTERS):
        return False  # "approx. five" / "e.g. this" - next word is lowercase
    if match.group().startswith(".") and not match.group().startswith(".."):
        word = _PRECEDING_WORD.search(text[max(0, match.start() - 20):match.start()])
        if word:
            w = word.group(1).lower()
            if w in _ABBREVIATIONS or (len(w) == 1 and w.isalpha()):
                return False  # "Dr. Smith", "J. Smith"
    return True


def split_sentences(text: str) -> list[str]:
    """Fast rule-based splitter (paragraph breaks are hard boundaries).

    Swap in pysbd or spaCy if you need higher precision; the chunker only needs
    `list[str]` back.
    """
    sentences: list[str] = []
    for paragraph in _PARAGRAPH.split(text.strip()):
        paragraph = paragraph.strip()
        start = 0
        for m in _CANDIDATE.finditer(paragraph):
            if _is_boundary(paragraph, m):
                s = paragraph[start:m.end()].strip()
                if s:
                    sentences.append(s)
                start = m.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------- chunker
class SentenceChunker:
    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        max_tokens: int = 500,
        overlap_sentences: int = 0,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.tokenizer = tokenizer or TiktokenTokenizer()
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc: Document) -> list[Chunk]:
        groups = self._pack(self._pieces(doc.text))
        return [
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, position),
                doc_id=doc.doc_id,
                position=position,
                text=" ".join(text for text, _ in group),
                token_count=sum(n for _, n in group),
                title=doc.title,
                source_url=doc.source_url,
                allowed_teams=doc.allowed_teams,
            )
            for position, group in enumerate(groups)
        ]

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        pieces: list[tuple[str, int]] = []
        for sentence, n in zip(sentences, self.tokenizer.count_many(sentences)):
            if n <= self.max_tokens:
                pieces.append((sentence, n))
            else:  # the only case where a sentence is split
                parts = self.tokenizer.split_by_tokens(sentence, self.max_tokens)
                pieces.extend(zip(parts, self.tokenizer.count_many(parts)))
        return pieces

    def _pack(self, pieces: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        groups: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []
        used = 0
        for piece in pieces:
            n = piece[1]
            if current and used + n > self.max_tokens:
                groups.append(current)
                current = self._overlap_tail(current, incoming=n)
                used = sum(t for _, t in current)
            current.append(piece)
            used += n
        if current:
            groups.append(current)
        return groups

    def _overlap_tail(self, closed: list[tuple[str, int]], incoming: int) -> list[tuple[str, int]]:
        tail = closed[-self.overlap_sentences:] if self.overlap_sentences else []
        while tail and sum(t for _, t in tail) + incoming > self.max_tokens:
            tail = tail[1:]
        return list(tail)


# --------------------------------------------------------------------------- scale-out
_worker: SentenceChunker | None = None


def _init_worker(max_tokens: int, overlap: int) -> None:
    global _worker
    _worker = SentenceChunker(max_tokens=max_tokens, overlap_sentences=overlap)


def _chunk_one(doc: Document) -> list[Chunk]:
    assert _worker is not None
    return _worker.chunk(doc)


def chunk_documents(
    docs: Iterable[Document],
    *,
    max_tokens: int = 500,
    overlap_sentences: int = 0,
    workers: int = 1,
    block_size: int = 5_000,
) -> Iterator[Chunk]:
    """Lazily turn a stream of documents into a stream of chunks.

    Memory stays bounded: with workers > 1 documents are fed to the process pool in blocks
    (ProcessPoolExecutor.map would otherwise submit all 10M documents up front).
    """
    if workers <= 1:
        chunker = SentenceChunker(max_tokens=max_tokens, overlap_sentences=overlap_sentences)
        for doc in docs:
            yield from chunker.chunk(doc)
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(max_tokens, overlap_sentences)
    ) as pool:
        for block in batched(docs, block_size):
            for chunks in pool.map(_chunk_one, block, chunksize=50):
                yield from chunks
