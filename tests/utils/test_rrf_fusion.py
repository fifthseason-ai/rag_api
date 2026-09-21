"""Reciprocal Rank Fusion — the scorer that fuses the dense and keyword arms.

`reciprocal_rank_fusion` runs on EVERY hybrid `/query` (HYBRID_SEARCH_ENABLED defaults
True) yet had no test. Its whole reason to exist is that dense (cosine DISTANCE, lower is
better) and keyword (ts_rank_cd, higher is better) live on incompatible scales, so it must
fuse by RANK, never by score. These tests pin that contract and the arithmetic, dedup and
truncation around it — each with a control that reddens if the property is broken.
"""
from langchain_core.documents import Document

from app.config import RRF_K
from app.services.hybrid_search import reciprocal_rank_fusion, _fusion_key


def _doc(text, **md):
    return Document(page_content=text, metadata=md)


def _order(fused):
    return [d.page_content for d, _s in fused]


def test_fuses_by_rank_not_by_score():
    """THE contract: only rank matters, never the incompatible dense/keyword score scales.

    Discriminating fixture (rank-fusion and score-fusion give DIFFERENT orders):
      * `both`   is rank 0 in dense AND rank 1 in keyword -> two rank contributions;
      * `hidense` is rank 0 in keyword only, but with a HUGE ts_rank score.
    By RANK, `both` (present in two lists) outranks `hidense`. By raw SCORE, `hidense`'s
    huge number would dominate and win. So this reddens the moment fusion uses score."""
    both = _doc("both")
    hidense = _doc("hidense")
    filler = _doc("filler")
    dense = [(both, 0.01), (filler, 0.9)]        # both = dense rank 0
    keyword = [(hidense, 1000.0), (both, 1.0)]   # hidense = keyword rank 0 (huge score)

    fused = reciprocal_rank_fusion([dense, keyword], k=3, rrf_k=60)
    order = _order(fused)
    # By rank: both = 1/61 + 1/62 ; hidense = 1/61  -> both first. A score-based fuser
    # would put hidense (1000.0) first.
    assert order[0] == "both", order
    assert order.index("both") < order.index("hidense"), order


def test_rrf_arithmetic_is_one_over_rrfk_plus_rank_plus_one():
    """Exact RRF score for a single list: rank is 0-indexed with a +1."""
    x, y = _doc("X"), _doc("Y")
    fused = reciprocal_rank_fusion([[(x, 0.1), (y, 0.2)]], k=2, rrf_k=60)
    scores = {d.page_content: s for d, s in fused}
    assert abs(scores["X"] - 1.0 / (60 + 0 + 1)) < 1e-12, scores
    assert abs(scores["Y"] - 1.0 / (60 + 1 + 1)) < 1e-12, scores


def test_a_doc_in_both_lists_sums_and_beats_a_single_list_top_rank():
    """Overlap is rewarded: a doc present in BOTH lists (even mid-rank in each) sums its
    contributions and outranks a doc that is #1 in only one list. Control: if the fuser
    failed to SUM overlaps (e.g. last-wins), 'shared' would not win."""
    shared = _doc("shared")
    solo = _doc("solo")
    other = _doc("other")
    # shared at rank 1 in both lists; solo at rank 0 in list one only.
    list1 = [(solo, 0.1), (shared, 0.2)]
    list2 = [(other, 0.1), (shared, 0.2)]
    fused = reciprocal_rank_fusion([list1, list2], k=3, rrf_k=60)

    order = _order(fused)
    assert order[0] == "shared", order
    # shared appears exactly once (deduped), not twice.
    assert order.count("shared") == 1, order
    shared_score = dict((d.page_content, s) for d, s in fused)["shared"]
    assert abs(shared_score - (1.0 / 62 + 1.0 / 62)) < 1e-12, shared_score


def test_fusion_key_dedups_by_digest_then_by_content():
    """_fusion_key prefers an explicit metadata digest and falls back to content hash."""
    d1 = _doc("different text one", digest="same-digest")
    d2 = _doc("different text two", digest="same-digest")
    assert _fusion_key(d1) == _fusion_key(d2), "same digest must be one identity"

    c1 = _doc("identical content")
    c2 = _doc("identical content")
    assert _fusion_key(c1) == _fusion_key(c2), "same content must be one identity"

    assert _fusion_key(_doc("alpha")) != _fusion_key(_doc("beta")), "distinct content"

    # Two chunks with the same digest fuse into a single ranked entry.
    fused = reciprocal_rank_fusion([[(d1, 0.1)], [(d2, 0.2)]], k=5)
    assert len(fused) == 1, fused


def test_truncates_to_k_by_fused_score():
    docs = [_doc(c) for c in "ABCDE"]
    ranked = [(d, 0.1 * i) for i, d in enumerate(docs)]
    fused = reciprocal_rank_fusion([ranked], k=2)
    assert len(fused) == 2, fused
    # best-first: the two earliest ranks survive.
    assert _order(fused) == ["A", "B"], _order(fused)


def test_empty_and_single_list():
    assert reciprocal_rank_fusion([], k=3) == []
    assert reciprocal_rank_fusion([[]], k=3) == []
    a, b = _doc("A"), _doc("B")
    fused = reciprocal_rank_fusion([[(a, 0.1), (b, 0.2)]], k=5)
    assert _order(fused) == ["A", "B"], "a single list must preserve its rank order"


def test_scores_are_descending():
    docs = [(_doc(c), 0.1) for c in "ABCD"]
    fused = reciprocal_rank_fusion([docs], k=4)
    scores = [s for _d, s in fused]
    assert scores == sorted(scores, reverse=True), scores
