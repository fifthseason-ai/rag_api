"""The test suite cannot reach the real Bedrock rerank provider.

MEASURED HAZARD (RV-122 on #106, 2026-09-23): `rerank()` is on by default
(`RERANK_ENABLED` defaults True) and several suites drive a query route without stubbing
it -- `tests/test_main.py` and `tests/test_entitlement_routes.py` stub nothing at all. A
plain `pytest` run therefore reached the REAL Bedrock Rerank endpoint. The reviewer only
noticed because their container had no network; on a host with egress the call goes out.

With dummy credentials the call is rejected before any inference, so it is not billed --
but "probably not billed" is not the standard. Under the standing no-paid-calls hold a
suite must not be ABLE to call a paid provider, and one that stays free only because the
credentials happen to be wrong is one real `.env` away from spending money.

The guard lives in `tests/conftest.py` as an autouse fixture that blocks
`reranker._get_client`. This file is its CONTROL: without these, the guard is a claim.

Why `_get_client` and not `_rerank_sync`: a test that stubs `_rerank_sync` never reaches
the client, so the suites that legitimately exercise rerank keep working untouched. Test 3
below pins exactly that, so a future "tighten the guard" change cannot quietly break them.
"""
import pytest
from langchain_core.documents import Document

from app.services import reranker


def _candidates(*texts):
    return [(Document(page_content=t), 0.1 * i) for i, t in enumerate(texts)]


def test_the_real_rerank_client_cannot_be_built():
    """THE GUARD. If this ever passes silently, the suite can bill the account."""
    with pytest.raises(RuntimeError) as excinfo:
        reranker._get_client()
    assert "blocked" in str(excinfo.value), excinfo.value


async def test_an_unstubbed_rerank_degrades_instead_of_calling_out(monkeypatch):
    """THE HAZARD SCENARIO: rerank enabled, nothing stubbed -- exactly what test_main.py
    and test_entitlement_routes.py do. It must fall back to the pre-rerank order rather
    than open a socket, and it must not raise into the route."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    candidates = _candidates("a", "b", "c")

    out = await reranker.rerank("q", candidates, top_n=2)

    assert [doc.page_content for doc, _score in out] == ["a", "b"], out


async def test_a_stubbed_rerank_still_produces_reranked_output(monkeypatch):
    """NO COLLATERAL DAMAGE: the guard must not disable rerank for the suites that
    legitimately exercise it -- they stub `_rerank_sync`, which never reaches the client."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    monkeypatch.setattr(
        reranker,
        "_rerank_sync",
        lambda query, documents, top_n: [
            {"index": 2, "relevanceScore": 0.9},
            {"index": 0, "relevanceScore": 0.4},
        ],
    )

    out = await reranker.rerank("q", _candidates("a", "b", "c"), top_n=2)

    assert [doc.page_content for doc, _score in out] == ["c", "a"], out
