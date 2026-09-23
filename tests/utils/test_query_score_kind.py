"""F-QUERY-SCORE-KIND-ON-THE-WIRE, hermetic half: each `/query*` hit declares what its score MEANS.

PACKET 1 found (program-control 5e49dfe) that the second element of every `[document, score]`
pair means a cosine DISTANCE in dense mode, an RRF fused score in the DEFAULT hybrid mode and a
provider relevance after a successful rerank -- with nothing on the wire saying which -- while
the consumer read all of them as a distance. Under the default configuration that sorts this
service's best chunks LAST. Contract: P06-5 addendum 2 (`score_kind` + `score_direction`,
top-level on the document, additive; absent/null = UNKNOWN).

This file pins the DECLARATION at every place it is decided or could fall off, without a
database:
  * `_dense_kind` declares only store types it can identify, never a guess;
  * `rerank` declares `rerank_relevance` only on success, and EVERY fallback keeps its input's
    kind (a slice is a plain list -- the easiest place for a kind to vanish);
  * `_authorized_only` rebuilds the list and must carry the kind across;
  * all three routes put both fields on the wire, top-level, with the pair shape unchanged;
  * an undeclared list (a stub, a future caller) goes out as null/null -- UNKNOWN, stated.

The real-pgvector half -- each mode driven for real, the declared kind checked against the
ARITHMETIC that produced the number -- is `test_query_score_kind_real_pg.py`.
"""
import datetime
import os

import jwt
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from langchain_core.documents import Document

os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes import document_routes as dr  # noqa: E402
from app.services import reranker  # noqa: E402
from app.services.score_kind import (  # noqa: E402
    COSINE_DISTANCE,
    DIRECTION,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    RERANK_RELEVANCE,
    RRF,
    VECTOR_SEARCH_SCORE,
    ScoredHits,
    kind_of,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector  # noqa: E402
from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector  # noqa: E402
from langchain_community.vectorstores.pgvector import DistanceStrategy  # noqa: E402
from main import app  # noqa: E402


# -- the closed set -------------------------------------------------------------------------

def test_the_closed_set_and_its_directions():
    """The contract's table, pinned: four kinds, and which way is better for each."""
    assert DIRECTION == {
        COSINE_DISTANCE: LOWER_IS_BETTER,
        RRF: HIGHER_IS_BETTER,
        RERANK_RELEVANCE: HIGHER_IS_BETTER,
        VECTOR_SEARCH_SCORE: HIGHER_IS_BETTER,
    }
    assert (COSINE_DISTANCE, RRF, RERANK_RELEVANCE, VECTOR_SEARCH_SCORE) == (
        "cosine_distance", "rrf", "rerank_relevance", "vector_search_score")


def test_a_kind_outside_the_closed_set_is_refused():
    with pytest.raises(ValueError):
        ScoredHits([], "similarity")


def test_a_plain_list_is_unknown_not_guessed():
    assert kind_of([(Document(page_content="x"), 0.5)]) is None
    assert ScoredHits([(Document(page_content="x"), 0.5)], RRF) == [(Document(page_content="x"), 0.5)], \
        "ScoredHits must still BE the list every existing caller compares against"


# -- where the dense kind is decided ---------------------------------------------------------

def _bare(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


@pytest.mark.parametrize("strategy, expected", [
    (DistanceStrategy.COSINE, COSINE_DISTANCE),
    (DistanceStrategy.EUCLIDEAN, None),
    (DistanceStrategy.MAX_INNER_PRODUCT, None),
])
def test_pgvector_declares_cosine_distance_only_for_the_cosine_strategy(strategy, expected):
    """The consumer's calibration (relevance = 1 - distance, its 0.45 floor) is cosine-shaped;
    a euclidean or inner-product number declared as `cosine_distance` would be a lie."""
    assert dr._dense_kind(_bare(AsyncPgVector, _distance_strategy=strategy)) == expected


def test_atlas_declares_its_vector_search_score():
    assert dr._dense_kind(_bare(AtlasMongoVector)) == VECTOR_SEARCH_SCORE


def test_an_unidentified_store_is_unknown():
    class FakeStore:
        _distance_strategy = DistanceStrategy.COSINE  # looks right; is not a PGVector
    assert dr._dense_kind(FakeStore()) is None


# -- rerank: success declares relevance, every fallback keeps its input's kind ---------------

def _cands(kind, *texts):
    return ScoredHits([(Document(page_content=t), 0.01 * (i + 1)) for i, t in enumerate(texts)], kind)


async def test_rerank_success_declares_rerank_relevance(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "_rerank_sync",
                        lambda q, d, n: [{"index": 1, "relevanceScore": 0.9}, {"index": 0, "relevanceScore": 0.4}])
    out = await reranker.rerank("q", _cands(RRF, "a", "b"), top_n=2)
    assert kind_of(out) == RERANK_RELEVANCE
    assert [(d.page_content, s) for d, s in out] == [("b", 0.9), ("a", 0.4)], \
        "declared rerank_relevance, so the numbers must BE the provider's relevance"


@pytest.mark.parametrize("input_kind", [RRF, COSINE_DISTANCE, None])
@pytest.mark.parametrize("path", ["provider_raises", "provider_empty", "disabled"])
async def test_every_rerank_fallback_keeps_the_candidates_kind(monkeypatch, input_kind, path):
    """A failed rerank returns the CANDIDATES' numbers, so it must declare the candidates' kind.
    The default region (us-east-1) does not host the model, so `provider_raises` is the path a
    default deployment takes on every /query."""
    monkeypatch.setattr(reranker, "RERANK_ENABLED", path != "disabled")

    def _raise(q, d, n):
        raise RuntimeError("bedrock unavailable in this region")
    provider = {"provider_raises": _raise, "provider_empty": lambda q, d, n: [],
                "disabled": _raise}[path]
    monkeypatch.setattr(reranker, "_rerank_sync", provider)

    cands = _cands(input_kind, "a", "b", "c")
    out = await reranker.rerank("q", cands, top_n=2)
    assert kind_of(out) == input_kind, (path, input_kind, kind_of(out))
    assert list(out) == list(cands)[:2], "a fallback must return the candidates' own numbers"


async def test_rerank_of_nothing_keeps_the_kind(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    out = await reranker.rerank("q", ScoredHits([], RRF), top_n=3)
    assert out == [] and kind_of(out) == RRF


# -- the entitlement post-filter rebuilds the list and must carry the kind -------------------

def test_authorized_only_carries_the_kind_across():
    docs = ScoredHits([(Document(page_content="mine", metadata={"user_id": "uA"}), 0.03),
                       (Document(page_content="theirs", metadata={"user_id": "uB"}), 0.02)], RRF)
    out = dr._authorized_only(docs, ["uA"])
    assert [d.page_content for d, _ in out] == ["mine"]
    assert kind_of(out) == RRF, "dropping a pair must not drop what the remaining scores mean"


# -- the wire, on every query route ----------------------------------------------------------

def _auth():
    payload = {"id": "userA", "tid": "tenantA", "ent": ["userA"], "act": ["read"],
               "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)}
    return {"Authorization": "Bearer " + jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")}


_ROUTES = {
    "query": lambda c: c.post("/query", json={"query": "q", "file_id": "f1", "k": 2}, headers=_auth()),
    "query_by_entity": lambda c: c.post("/query/userA", json={"query": "q", "k": 2}, headers=_auth()),
    "query_multiple": lambda c: c.post("/query_multiple", json={"query": "q", "file_ids": ["f1"], "k": 2},
                                       headers=_auth()),
}


def _client(monkeypatch, returned):
    async def fake_retrieve(*_a, **_k):
        return returned
    monkeypatch.setattr(dr, "_retrieve_documents", fake_retrieve)
    monkeypatch.setattr(dr, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
    return TestClient(app)


def _hits(kind):
    meta = {"file_id": "f1", "user_id": "userA", "page": 3}
    pairs = [(Document(page_content="best", metadata=dict(meta)), 0.0323),
             (Document(page_content="next", metadata=dict(meta)), 0.0161)]
    return ScoredHits(pairs, kind) if kind else pairs


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("kind", [RRF, COSINE_DISTANCE, RERANK_RELEVANCE, VECTOR_SEARCH_SCORE])
def test_every_route_declares_the_kind_on_the_wire(monkeypatch, route, kind):
    r = _ROUTES[route](_client(monkeypatch, _hits(kind)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 2
    for hit in body:
        assert isinstance(hit, list) and len(hit) == 2, "the pair shape must not change: %r" % (hit,)
        doc, score = hit
        assert doc["score_kind"] == kind and doc["score_direction"] == DIRECTION[kind], doc
        assert "score_kind" not in doc["metadata"] and "score_direction" not in doc["metadata"], \
            "the declaration describes the RESPONSE; it must not be written into the chunk's metadata"
        assert doc["metadata"]["page"] == 3, "declaring the kind must not disturb the chunk metadata"
    assert [h[1] for h in body] == [0.0323, 0.0161], "the numbers themselves are unchanged"
    assert [h[0]["page_content"] for h in body] == ["best", "next"], "the order is unchanged"


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_an_undeclared_result_goes_out_as_unknown(monkeypatch, route):
    """A list built outside the declaring pipeline must say UNKNOWN (null/null), never a guess --
    and must still be served exactly as before."""
    r = _ROUTES[route](_client(monkeypatch, _hits(None)))
    assert r.status_code == 200, r.text
    for doc, _score in r.json():
        assert doc["score_kind"] is None and doc["score_direction"] is None, doc
