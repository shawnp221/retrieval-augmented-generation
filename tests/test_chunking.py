from rag.chunking import SentenceChunker, split_sentences
from rag.models import Document


def _doc(text):
    return Document(doc_id="d1", text=text)


def _numbered(n):  # each sentence is exactly 10 "tokens"
    return " ".join(f"This is sentence number {i} with some words in it." for i in range(n))


def test_no_sentence_is_cut(word_tokenizer):
    text = _numbered(200)
    chunks = SentenceChunker(word_tokenizer, max_tokens=50).chunk(_doc(text))
    assert len(chunks) == 40
    for c in chunks:
        assert c.text.startswith("This is sentence")
        assert c.text.endswith("in it.")
    assert " ".join(c.text for c in chunks) == text          # lossless without overlap


def test_token_budget_respected(word_tokenizer):
    chunks = SentenceChunker(word_tokenizer, max_tokens=37).chunk(_doc(_numbered(100)))
    assert all(c.token_count <= 37 for c in chunks)


def test_abbreviations_and_decimals_do_not_split():
    s = split_sentences("Dr. Smith approved the refund. It was $3.50 in total, e.g. a small one. Done!")
    assert s == ["Dr. Smith approved the refund.", "It was $3.50 in total, e.g. a small one.", "Done!"]


def test_paragraph_break_is_a_boundary():
    assert split_sentences("First line without period\n\nSecond paragraph.") == [
        "First line without period", "Second paragraph."]


def test_oversized_sentence_is_hard_split(word_tokenizer):
    long_sentence = " ".join(["word"] * 120) + "."
    chunks = SentenceChunker(word_tokenizer, max_tokens=50).chunk(_doc(long_sentence))
    assert [c.token_count for c in chunks] == [50, 50, 20]


def test_overlap_repeats_last_sentence(word_tokenizer):
    chunks = SentenceChunker(word_tokenizer, max_tokens=30, overlap_sentences=1).chunk(_doc(_numbered(9)))
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(b.text)[0] == split_sentences(a.text)[-1]
        assert b.token_count <= 30


def test_ids_are_deterministic_and_positioned(word_tokenizer):
    a = SentenceChunker(word_tokenizer, max_tokens=20).chunk(_doc(_numbered(10)))
    b = SentenceChunker(word_tokenizer, max_tokens=20).chunk(_doc(_numbered(10)))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert [c.position for c in a] == list(range(len(a)))


def test_empty_document(word_tokenizer):
    assert SentenceChunker(word_tokenizer).chunk(_doc("   ")) == []
