import pytest

from rag.models import Chunk


class WordTokenizer:
    """1 whitespace-separated word == 1 token. Keeps chunking tests offline and exact."""

    def count_many(self, texts):
        return [len(t.split()) for t in texts]

    def split_by_tokens(self, text, max_tokens):
        w = text.split()
        return [" ".join(w[i:i + max_tokens]) for i in range(0, len(w), max_tokens)]


@pytest.fixture
def word_tokenizer():
    return WordTokenizer()


def make_chunk(cid: str, text: str = "text", teams=("sales",)) -> Chunk:
    return Chunk(cid, "doc", 0, text, len(text.split()), title=cid, source_url=f"https://x/{cid}",
                 allowed_teams=tuple(teams))
