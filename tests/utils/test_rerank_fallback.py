"""The reranker's fallback + mapping contract (RERANK_ENABLED defaults True).

`rerank()` runs on every `/query` after hybrid retrieval and calls Cohere-via-Bedrock.
It is best-effort: on disable, empty input, provider failure, an empty provider result,
or out-of-range indices it must fall back to the pre-rerank order and NEVER drop the
answer or 500. On success it must reorder by the provider's indices and map the relevance
scores. None of this was tested. The only external call (`_rerank_sync`) is monkeypatched
so no AWS is touched; each test carries the control that reddens if the property breaks.
"""
import pytest
from langchain_core.documents import Document

from app.services import reranker
from app.services.reranker import rerank


def _cands(*texts):
    return [(Document(page_content=t), 0.1 * i) for i, t in enumerate(texts)]


async def test_empty_candidates_returns_empty(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    assert await rerank("q", [], top_n=5) == []


async def test_disabled_preserves_pre_rerank_order(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)
    cands = _cands("a", "b", "c")

    def _must_not_call(*a, **k):
        raise AssertionError("provider called while disabled")
    monkeypatch.setattr(reranker, "_rerank_sync", _must_not_call)

    out = await rerank("q", cands, top_n=2)
    assert [d.page_content for d, _ in out] == ["a", "b"], out


async def test_provider_failure_falls_back_and_is_logged(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)

    def _boom(query, documents, top_n):
        raise RuntimeError("bedrock down")
    monkeypatch.setattr(reranker, "_rerank_sync", _boom)

    with caplog.at_level(logging.WARNING):
        out = await rerank("q", _cands("a", "b", "c"), top_n=2)
    assert [d.page_content for d, _ in out] == ["a", "b"], "a failed rerank must not drop the answer"
    assert any("[rerank] failed" in r.getMessage() for r in caplog.records), \
        "a rerank failure must be surfaced in the log"


async def test_empty_provider_result_falls_back(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "_rerank_sync", lambda q, d, n: [])
    out = await rerank("q", _cands("a", "b", "c"), top_n=2)
    assert [d.page_content for d, _ in out] == ["a", "b"], out


async def test_success_reorders_by_index_and_maps_scores(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    # Provider says candidate #2 is most relevant, then #0.
    monkeypatch.setattr(
        reranker, "_rerank_sync",
        lambda q, d, n: [{"index": 2, "relevanceScore": 0.91}, {"index": 0, "relevanceScore": 0.42}],
    )
    out = await rerank("q", _cands("a", "b", "c"), top_n=2)
    assert [d.page_content for d, _ in out] == ["c", "a"], out
    assert [round(s, 2) for _d, s in out] == [0.91, 0.42], out


async def test_out_of_range_index_is_filtered_then_falls_back(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    # An index past the candidate list must never IndexError; if nothing valid
    # survives, fall back to pre-rerank order.
    monkeypatch.setattr(reranker, "_rerank_sync", lambda q, d, n: [{"index": 99, "relevanceScore": 0.9}])
    out = await rerank("q", _cands("a", "b", "c"), top_n=2)
    assert [d.page_content for d, _ in out] == ["a", "b"], out


async def test_top_n_is_clamped_to_the_candidate_count(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)
    cands = _cands("a", "b")
    # top_n larger than the pool must not over-read.
    assert len(await rerank("q", cands, top_n=10)) == 2
    # top_n <= 0 is clamped up to at least 1 (max(1, ...)).
    assert len(await rerank("q", cands, top_n=0)) == 1
