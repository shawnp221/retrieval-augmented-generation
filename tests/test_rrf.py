from rag.retrieval import reciprocal_rank_fusion


def test_chunk_high_in_both_lists_wins():
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "y"]], k=60)
    assert fused[0][0] == "y"


def test_known_ordering():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [cid for cid, _ in fused] == ["b", "a", "d", "c"]


def test_score_formula():
    (cid, score), = reciprocal_rank_fusion([["a"], ["a"]], k=60)
    assert cid == "a" and abs(score - 2 / 61) < 1e-12


def test_top_n_and_duplicates_within_a_list():
    fused = reciprocal_rank_fusion([["a", "a", "b"], ["c"]], k=60, top_n=2)
    assert len(fused) == 2
    assert dict(reciprocal_rank_fusion([["a", "a"]]))["a"] == 1 / 61   # counted once


def test_fifty_plus_fifty_gives_up_to_hundred():
    v = [f"v{i}" for i in range(50)]
    k = [f"k{i}" for i in range(50)]
    assert len(reciprocal_rank_fusion([v, k])) == 100
    assert len(reciprocal_rank_fusion([v, v])) == 50
