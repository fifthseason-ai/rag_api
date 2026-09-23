"""F-QUERY-SCORE-KIND-ON-THE-WIRE, real-pgvector half: the DECLARED kind matches the ARITHMETIC.

Each retrieval mode is driven for real, through every query route, on a pgvector store with
`document_tsv` (so hybrid genuinely fuses), and the numbers on the wire are recomputed here
INDEPENDENTLY of the code under test:

  * cosine_distance -- 1 - cos(query, row) from the fixed vector table below;
  * rrf             -- sum of 1/(RRF_K + rank + 1) over the two arms' orders, where each arm's
                       order is MEASURED by calling that arm directly (never by calling
                       `reciprocal_rank_fusion`, which is the thing being checked);
  * rerank_relevance -- the numbers the (faked, deterministic) provider returned.

So a declaration that says one kind while the pipeline produced another reddens, and so does a
kind that is right while the numbers came from somewhere else. The two fallbacks a DEFAULT
deployment actually takes are pinned too: hybrid configured but the keyword arm failing (a DB
without `document_tsv`) must declare `cosine_distance`; rerank enabled but the provider failing
(the default region does not host the model) must declare the fallback's `rrf`.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so these provably ran.
Fixture shape follows tests/utils/test_entitlement_fused.py (same init + pool handling).
"""
import asyncio
import datetime
import math
import os

import jwt
import psycopg2
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from langchain_core.documents import Document

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the score-kind modes",
)

_SECRET = "test-secret-score-kind"
_COLLECTION = "score_kind"

TERM = "vornakite4"  # the query; in R1 once and R3 twice, nowhere else

R1 = f"{TERM} anchor ledger note"          # nearest vector, keyword match
R2 = "cash forecast narrative for the office"  # mid vector, NO keyword match
R3 = f"{TERM} {TERM} reconciliation checklist"  # far vector, strongest keyword match

_VECTORS = {
    TERM: [1.0, 0.0, 0.0],
    R1: [1.0, 0.0, 0.0],
    R2: [0.6, 0.8, 0.0],
    R3: [0.0, 1.0, 0.0],
}

ENT = "uA"
FILE = "fA"


class _TableEmb:
    """Fixed text -> vector table. An unknown text raises: fixture drift must be loud."""

    def embed_documents(self, texts):
        return [list(_VECTORS[t]) for t in texts]

    def embed_query(self, text):
        return list(_VECTORS[text])


def _cos_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _tok():
    os.environ["JWT_SECRET"] = _SECRET
    payload = {"id": "caller", "tid": "tA", "ent": [ENT], "act": ["read"],
               "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)}
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _real_post_init(self):
    """Run the REAL pgvector init that conftest no-ops for the session."""
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
    if self.create_extension:
        self.create_vector_extension()
    EmbeddingStore, CollectionStore = _get_embedding_collection_store(
        self._embedding_length, use_jsonb=self.use_jsonb
    )
    self.CollectionStore = CollectionStore
    self.EmbeddingStore = EmbeddingStore
    self.create_tables_if_not_exists()
    self.create_collection()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture()
def astore(monkeypatch):
    """Real pgvector + document_tsv, both arms pointed at it. Hybrid on, rerank OFF by default."""
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services import reranker
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="sync")
    _real_post_init(store)
    rows = [R1, R2, R3]
    store.add_documents(
        [Document(page_content=t, metadata={"file_id": FILE, "user_id": ENT, "tenant_id": "tA"})
         for t in rows],
        ids=[f"s{i}" for i in range(len(rows))],
    )
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        c.commit()

    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)

    a = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="async")
    _real_post_init(a)
    monkeypatch.setattr(dr, "vector_store", a)
    monkeypatch.setattr("app.config.vector_store", a, raising=False)
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(dr, "RERANK_ENABLED", False)
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)

    real_kw = dr.keyword_search

    async def _kw_own_pool(*args, **kwargs):
        # One asyncpg pool per call: TestClient runs each request on a fresh loop
        # (F-HYBRID-COMPOSE, the measured deadlock).
        try:
            return await real_kw(*args, **kwargs)
        finally:
            await PSQLDatabase.close_pool()
    monkeypatch.setattr(dr, "keyword_search", _kw_own_pool)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield a

    if PSQLDatabase.pool is not None:
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


_ROUTES = {
    "query": lambda c: c.post("/query", json={"query": TERM, "file_id": FILE, "k": 5}, headers=_tok()),
    "query_by_entity": lambda c: c.post(f"/query/{ENT}", json={"query": TERM, "k": 5}, headers=_tok()),
    "query_multiple": lambda c: c.post("/query_multiple", json={"query": TERM, "file_ids": [FILE], "k": 5},
                                       headers=_tok()),
}


def _wire(route):
    import main
    r = _ROUTES[route](TestClient(main.app))
    assert r.status_code == 200, r.text
    hits = r.json()
    assert hits, "no hits: the fixture did not reach the route"
    kinds = {(h[0]["score_kind"], h[0]["score_direction"]) for h in hits}
    assert len(kinds) == 1, "one response must carry ONE kind: %r" % kinds
    (kind, direction), = kinds
    return kind, direction, [(h[0]["page_content"], h[1]) for h in hits]


def _measured_arm_orders():
    """Each arm's order, measured by calling the arm itself on the same store."""
    from app.routes import document_routes as dr
    from app.services.database import PSQLDatabase
    from app.services.hybrid_search import keyword_search  # the arm itself, not the route's wrapper

    dense = [d.page_content for d, _ in dr.vector_store.similarity_search_with_score_by_vector(
        _VECTORS[TERM], k=5, filter={"file_id": {"$eq": FILE}})]

    async def _kw():
        try:
            return await keyword_search(TERM, k=5, filters={"file_id": FILE, "user_id": ENT})
        finally:
            await PSQLDatabase.close_pool()
    keyword = [d.page_content for d, _ in _run(_kw())]
    return dense, keyword


def _rrf_expected(dense, keyword, rrf_k):
    fused = {}
    for order in (dense, keyword):
        for rank, text in enumerate(order):
            fused[text] = fused.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused


@needs_pg
def test_premise_the_two_arms_disagree_so_fusion_is_observable(astore):
    """If the arms agreed, RRF and dense would give the same ORDER and a wrong declaration could
    hide behind a right order. Measured, not assumed."""
    dense, keyword = _measured_arm_orders()
    assert dense[:3] == [R1, R2, R3], dense
    assert keyword and R2 not in keyword and set(keyword) == {R1, R3}, keyword


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_hybrid_declares_rrf_and_the_numbers_are_rrf(astore, route):
    from app.config import RRF_K
    dense, keyword = _measured_arm_orders()
    expected = _rrf_expected(dense, keyword, RRF_K)

    kind, direction, hits = _wire(route)
    assert (kind, direction) == ("rrf", "higher_is_better"), (kind, direction)
    for text, score in hits:
        assert math.isclose(score, expected[text], rel_tol=0, abs_tol=1e-6), (text, score, expected[text])
    assert [s for _t, s in hits] == sorted((s for _t, s in hits), reverse=True), \
        "declared higher_is_better, so the producer's best-first order must be descending"


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_dense_only_declares_cosine_distance_and_the_numbers_are_distances(astore, monkeypatch, route):
    from app.routes import document_routes as dr
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", False)

    kind, direction, hits = _wire(route)
    assert (kind, direction) == ("cosine_distance", "lower_is_better"), (kind, direction)
    for text, score in hits:
        want = _cos_distance(_VECTORS[TERM], _VECTORS[text])
        assert math.isclose(score, want, rel_tol=0, abs_tol=1e-5), (text, score, want)
    assert [s for _t, s in hits] == sorted(s for _t, s in hits)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_hybrid_whose_keyword_arm_fails_declares_what_ran_dense(astore, monkeypatch, route):
    """The DEFAULT configuration on a DB without `document_tsv`: hybrid is configured, dense RAN.
    Declaring the configuration (rrf) here would be exactly the lie this card removes."""
    from app.routes import document_routes as dr

    async def _no_tsv(*_a, **_k):
        raise RuntimeError('column "document_tsv" does not exist')
    monkeypatch.setattr(dr, "keyword_search", _no_tsv)

    kind, direction, hits = _wire(route)
    assert (kind, direction) == ("cosine_distance", "lower_is_better"), (kind, direction)
    for text, score in hits:
        assert math.isclose(score, _cos_distance(_VECTORS[TERM], _VECTORS[text]), rel_tol=0, abs_tol=1e-5)


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_a_successful_rerank_declares_relevance_and_the_numbers_are_the_providers(astore, monkeypatch, route):
    from app.routes import document_routes as dr
    from app.services import reranker
    monkeypatch.setattr(dr, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    given = {}

    def _provider(query, documents, top_n):
        order = list(range(len(documents)))[::-1][:top_n]   # deterministic reversal
        out = [{"index": i, "relevanceScore": round(0.9 - 0.2 * n, 4)} for n, i in enumerate(order)]
        given.update({documents[r["index"]]: r["relevanceScore"] for r in out})
        return out
    monkeypatch.setattr(reranker, "_rerank_sync", _provider)

    kind, direction, hits = _wire(route)
    assert (kind, direction) == ("rerank_relevance", "higher_is_better"), (kind, direction)
    assert given, "the provider was never called: not the rerank path"
    assert {t: s for t, s in hits} == {t: given[t] for t, _s in hits}, (hits, given)
    # RV-118 note 1: the label and the numbers can both be right while the list is served in the
    # wrong ORDER; a relevance must also be a relevance.
    scores = [s for _t, s in hits]
    assert scores == sorted(scores, reverse=True), "declared higher_is_better, served %r" % scores
    assert all(0.0 <= s <= 1.0 for s in scores), "rerank_relevance outside [0, 1]: %r" % scores


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_a_failed_rerank_declares_the_fallbacks_kind_and_its_numbers(astore, monkeypatch, route):
    """A rerank provider that fails (credentials, region, an outage) falls back to the pre-rerank
    list: the numbers are the pre-rerank RRF numbers, so is the kind, and so is the ORDER.

    (The config comment says the default region us-east-1 does not host the model. MEASURED
    2026-09-23T21:47Z on a real 18eca4c build: cohere.rerank-v3-5:0 SUCCEEDED in us-east-1. So
    this fallback is not the default path; the success path is -- see the test above.)"""
    from app.config import RRF_K
    from app.routes import document_routes as dr
    from app.services import reranker
    monkeypatch.setattr(dr, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
    called = []

    def _down(query, documents, top_n):
        called.append(1)
        raise RuntimeError("model not hosted in this region")
    monkeypatch.setattr(reranker, "_rerank_sync", _down)

    dense, keyword = _measured_arm_orders()
    expected = _rrf_expected(dense, keyword, RRF_K)
    kind, direction, hits = _wire(route)
    assert called, "the provider was never called: not the rerank path"
    assert (kind, direction) == ("rrf", "higher_is_better"), (kind, direction)
    for text, score in hits:
        assert math.isclose(score, expected[text], rel_tol=0, abs_tol=1e-6), (text, score, expected[text])
    scores = [s for _t, s in hits]
    assert scores == sorted(scores, reverse=True), (
        "declared higher_is_better, but the fallback served %r (RV-118 M6: right label, right "
        "numbers, worst-first order)" % scores)
