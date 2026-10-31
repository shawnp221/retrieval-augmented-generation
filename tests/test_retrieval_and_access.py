from conftest import make_chunk
from rag.access import PUBLIC_LABEL, User, can_see, visible_teams
from rag.embedding import HashingEmbedder
from rag.models import Hit
from rag.retrieval import HybridRetriever


class FakeVectors:
    def __init__(self, ids): self.ids, self.seen_teams = ids, None
    def search(self, vector, teams, limit):
        self.seen_teams = list(teams)
        return [Hit(i, 1.0) for i in self.ids][:limit]


class FakeKeywords:
    def __init__(self, ids, chunks): self.ids, self.chunks, self.seen_teams = ids, chunks, None
    def search(self, query, teams, limit):
        self.seen_teams = list(teams)
        return [Hit(i, 1.0) for i in self.ids][:limit]
    def fetch(self, ids): return {i: self.chunks[i] for i in ids if i in self.chunks}


def test_visible_teams_always_include_public():
    assert visible_teams(User.of("u", "Sales")) == [PUBLIC_LABEL, "sales"]
    assert visible_teams(User.of("u")) == [PUBLIC_LABEL]


def test_can_see():
    sales = User.of("u", "sales")
    assert can_see(make_chunk("a", teams=("sales",)), sales)
    assert can_see(make_chunk("b", teams=("all",)), sales)
    assert not can_see(make_chunk("c", teams=("hr",)), sales)


def test_retriever_pushes_filter_down_fuses_and_rechecks_acl():
    chunks = {c: make_chunk(c) for c in ("a", "b", "c")}
    chunks["leak"] = make_chunk("leak", teams=("hr",))        # simulates an index/ACL bug
    vectors, keywords = FakeVectors(["a", "b", "leak"]), FakeKeywords(["b", "c"], chunks)
    r = HybridRetriever(HashingEmbedder(8), vectors, keywords, top_k=50)

    out = r.retrieve("refund limit", User.of("u", "sales"))

    assert vectors.seen_teams == keywords.seen_teams == ["all", "sales"]
    assert [c.chunk.chunk_id for c in out] == ["b", "a", "c"]   # b is in both lists -> first
    assert (out[0].vector_rank, out[0].keyword_rank) == (2, 1)
    assert "leak" not in [c.chunk.chunk_id for c in out]        # defence in depth


def test_blank_query_returns_nothing():
    r = HybridRetriever(HashingEmbedder(8), FakeVectors(["a"]), FakeKeywords(["a"], {}), top_k=5)
    assert r.retrieve("   ", User.of("u", "sales")) == []
