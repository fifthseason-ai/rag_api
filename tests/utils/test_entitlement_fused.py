"""F-ENTITLEMENT-FUSED: a caller entitled to NO documents gets [] from EVERY retrieval
entry point -- dense-only, keyword-only, fused, reranked -- on every query route, on a
real pgvector store, with no existence leak.

WHAT IS PROVEN
--------------
The caller holds a VALID entitlement (ent=[uC]) for an entity that owns zero rows. The
store holds rows of two other entities (uA, uB) that are the nearest vectors AND the
strongest keyword matches for the query, so any entitlement filter missing from any arm
would surface them. For each of the four retrieval modes x three query routes the answer
must be 200 `[]` -- not 403/404, not a count, not an error body -- and it must be the SAME
bytes as a query that matches nothing at all (no existence oracle).

Two layers are tested separately, because they are separate defences:

  1. The ARM filters (`_hybrid_or_dense_search`): each arm must apply the user_id filter
     itself. `test_each_arm_applies_the_entitlement_filter` calls the producer directly,
     so a dropped filter on EITHER arm reddens it even though the routes post-filter.
  2. The ROUTE post-filter (`_authorized_only`): every route re-filters the result to the
     token entitlement. /query and /query_multiple always did; /query/{entity_id} did NOT
     (F-HYBRID-COMPOSE finding) and gained it in this change (INTEGRATION decision
     2026-09-21). `test_route_post_filter_*` drives a leaking arm through the real routes.

Controls (run by hand, recorded in the checkpoint): drop user_id from the dense arm, then
from the keyword arm -> the producer test reddens per arm; with an arm leaking,
/query/{entity_id} leaked a foreign row on main (red-first) and returns [] with the
post-filter; removing the post-filter from /query/{entity_id} reddens the guard again.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so these provably ran.
"""
import asyncio
import datetime
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
    reason="RAG_TEST_PG_DSN not set: no pgvector for the entitlement-on-every-arm path",
)

_SECRET = "test-secret-entitlement-fused"
_COLLECTION = "entitlement_fused"

TERM = "zelquorin7"  # appears in every foreign row, nowhere else
NOTHING = "nothing matches this phrase"  # query with no keyword hit anywhere

OWN_A = f"{TERM} {TERM} ledger note for entity A"
OWN_A2 = "entity A cash forecast narrative"
OWN_B = f"{TERM} {TERM} {TERM} private memo of entity B"

_VECTORS = {
    TERM: [1.0, 0.0, 0.0],
    NOTHING: [0.0, 0.0, 1.0],
    OWN_A: [1.0, 0.0, 0.0],
    OWN_A2: [0.9, 0.4, 0.0],
    OWN_B: [1.0, 0.05, 0.0],
}

ENT_A, ENT_B, ENT_EMPTY = "uA", "uB", "uC"
FILE_A, FILE_B = "kfA", "kfB"


class _TableEmb:
    """Fixed text -> vector table. An unknown text raises: fixture drift must be loud."""

    def embed_documents(self, texts):
        return [list(_VECTORS[t]) for t in texts]

    def embed_query(self, text):
        return list(_VECTORS[text])


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _tok(entity_ids, act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "caller", "tid": "tA", "ent": list(entity_ids), "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
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


class _Env:
    def __init__(self, client, astore, kw_calls, dense_calls, rerank_calls):
        self.client = client
        self.astore = astore
        self.kw_calls = kw_calls
        self.dense_calls = dense_calls
        self.rerank_calls = rerank_calls


@pytest.fixture()
def env(monkeypatch):
    """Real pgvector store + document_tsv; both arms pointed at it; every arm counted."""
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
    rows = [(OWN_A, FILE_A, ENT_A), (OWN_A2, FILE_A, ENT_A), (OWN_B, FILE_B, ENT_B)]
    store.add_documents(
        [Document(page_content=t, metadata={"file_id": f, "user_id": u, "tenant_id": "tA"})
         for t, f, u in rows],
        ids=[f"r{i}" for i in range(len(rows))],
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

    astore = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)

    # Default for every test: hybrid on, rerank off. Modes flip these explicitly.
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(dr, "RERANK_ENABLED", False)

    kw_calls, dense_calls, rerank_calls = [], [], []
    real_kw = dr.keyword_search

    async def _counting_kw(*a, **k):
        # One asyncpg pool per call: TestClient runs each request on a fresh loop
        # (see test_hybrid_compose / F-HYBRID-COMPOSE for the measured deadlock).
        kw_calls.append(k.get("filters"))
        try:
            return await real_kw(*a, **k)
        finally:
            await PSQLDatabase.close_pool()
    monkeypatch.setattr(dr, "keyword_search", _counting_kw)

    real_dense = astore.asimilarity_search_with_score_by_vector

    async def _counting_dense(*a, **k):
        dense_calls.append(k.get("filter"))
        return await real_dense(*a, **k)
    monkeypatch.setattr(astore, "asimilarity_search_with_score_by_vector", _counting_dense)

    # The reranker's provider call is replaced by a deterministic REVERSAL (no Bedrock);
    # the real `rerank` wrapper still runs, so "reranked" is a real mode, not a skip.
    def _fake_rerank_sync(query, documents, top_n):
        rerank_calls.append(list(documents))
        order = list(range(len(documents)))[::-1][:top_n]
        return [{"index": i, "relevanceScore": 1.0 / (n + 1)} for n, i in enumerate(order)]
    monkeypatch.setattr(reranker, "_rerank_sync", _fake_rerank_sync)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield _Env(TestClient(main.app), astore, kw_calls, dense_calls, rerank_calls)

    if PSQLDatabase.pool is not None:
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


def _set_mode(monkeypatch, env, mode):
    """Select one retrieval entry point. Returns the arms that must have been called."""
    from app.routes import document_routes as dr
    from app.services import reranker

    if mode == "dense_only":
        monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", False)
        return {"dense"}
    if mode == "keyword_only":
        # Dense arm contributes nothing: the result is the keyword arm alone.
        async def _no_dense(*a, **k):
            env.dense_calls.append(k.get("filter"))
            return []
        monkeypatch.setattr(env.astore, "asimilarity_search_with_score_by_vector", _no_dense)
        return {"keyword"}
    if mode == "fused":
        return {"dense", "keyword"}
    if mode == "reranked":
        monkeypatch.setattr(dr, "RERANK_ENABLED", True)
        monkeypatch.setattr(reranker, "RERANK_ENABLED", True)
        return {"dense", "keyword"}
    raise AssertionError(mode)


_MODES = ["dense_only", "keyword_only", "fused", "reranked"]

# Every query route. The empty-entity caller names the foreign file ids where a route
# takes file ids: the file id is a filter, the entitlement must still win.
_ROUTES = {
    "query_by_entity": lambda c, q, ent: c.post(
        f"/query/{ENT_EMPTY}", json={"query": q, "k": 5}, headers=_tok(ent)),
    "query_by_file": lambda c, q, ent: c.post(
        "/query", json={"query": q, "file_id": FILE_B, "k": 5}, headers=_tok(ent)),
    "query_multiple": lambda c, q, ent: c.post(
        "/query_multiple", json={"query": q, "file_ids": [FILE_A, FILE_B], "k": 5},
        headers=_tok(ent)),
}


def _texts(resp):
    return [hit[0]["page_content"] for hit in resp.json()]


@needs_pg
def test_premise_every_mode_retrieves_foreign_rows_for_their_owner(env, monkeypatch):
    """Positive control: in each mode the foreign rows ARE retrievable by their owner, so
    their absence for the empty-entity caller below means the filter did it."""
    for mode in _MODES:
        with monkeypatch.context() as m:
            _set_mode(m, env, mode)
            r = env.client.post(f"/query/{ENT_B}", json={"query": TERM, "k": 5},
                                headers=_tok([ENT_B]))
            assert r.status_code == 200, (mode, r.text)
            assert _texts(r) == [OWN_B], (mode, _texts(r))
    assert env.rerank_calls, "the reranked mode never reached the reranker"


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("mode", _MODES)
def test_entitled_to_nothing_gets_empty_from_every_entry_point(env, monkeypatch, route, mode):
    """ACCEPTANCE: valid entitlement owning zero rows -> 200 [] on every route x mode, the
    required arms really ran, and the bytes equal a query that matches nothing at all."""
    arms = _set_mode(monkeypatch, env, mode)

    r = _ROUTES[route](env.client, TERM, [ENT_EMPTY])
    assert r.status_code == 200, r.text
    assert r.json() == [], r.json()

    if "dense" in arms:
        assert env.dense_calls, "dense arm never ran: not this entry point"
    if "keyword" in arms:
        assert env.kw_calls, "keyword arm never ran: not this entry point"
    for f in env.kw_calls:
        assert f["user_id"] in (ENT_EMPTY, [ENT_EMPTY]), f

    # No existence oracle: same status, same body, same content-length as a miss.
    miss = _ROUTES[route](env.client, NOTHING, [ENT_EMPTY])
    assert (miss.status_code, miss.content) == (r.status_code, r.content)
    assert miss.headers.get("content-length") == r.headers.get("content-length")


@needs_pg
@pytest.mark.parametrize("arm", ["dense", "keyword"])
def test_each_arm_applies_the_entitlement_filter(env, arm):
    """Layer 1, at the producer: `_hybrid_or_dense_search` with the empty entity's filter
    returns nothing, and the named arm returned nothing on its own. Dropping the user_id
    filter from EITHER arm reddens this -- the route post-filters cannot mask it here."""
    from app.routes import document_routes as dr
    from app.services.database import PSQLDatabase
    from app.services.hybrid_search import keyword_search

    class _Req:
        class app:
            class state:
                thread_pool = ThreadPoolExecutor(max_workers=1)

    async def _arm(entity):
        if arm == "dense":
            return await env.astore.asimilarity_search_with_score_by_vector(
                _VECTORS[TERM], k=5, filter=dr._to_langchain_filter({"user_id": [entity]}))
        return await keyword_search(TERM, k=5, filters={"user_id": [entity]})

    async def _go():
        try:
            fused = await dr._hybrid_or_dense_search(
                _Req, TERM, _VECTORS[TERM], 5, {"user_id": [ENT_EMPTY]})
            return fused, await _arm(ENT_EMPTY), await _arm(ENT_B)
        finally:
            await PSQLDatabase.close_pool()

    fused, alone, owner = _run(_go())
    assert fused == [], [d.page_content for d, _ in fused]
    assert alone == [], [d.page_content for d, _ in alone]
    # The same arm for the owner is non-empty: the filter, not the data, did it.
    assert [d.page_content for d, _ in owner] == [OWN_B], owner


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_route_post_filter_drops_rows_a_leaking_arm_returns(env, monkeypatch, route):
    """Layer 2, the defence in depth: an arm that IGNORES the entitlement (simulated by
    removing user_id from the filters it is handed) must not put a foreign row on the
    wire of ANY route. On main /query/{entity_id} had no post-filter and leaked here."""
    from app.routes import document_routes as dr

    real = dr._hybrid_or_dense_search

    async def _leaky(request, query, embedding, k, filters):
        unscoped = {key: v for key, v in filters.items() if key != "user_id"}
        return await real(request, query, embedding, k, unscoped)
    monkeypatch.setattr(dr, "_hybrid_or_dense_search", _leaky)

    r = _ROUTES[route](env.client, TERM, [ENT_EMPTY])
    assert r.status_code == 200, r.text
    assert r.json() == [], _texts(r)

    # The leak is real (otherwise [] above proves nothing): same arm, owner-less result.
    async def _raw():
        try:
            return await _leaky(type("R", (), {"app": env.client.app}), TERM,
                                _VECTORS[TERM], 5, {"user_id": [ENT_EMPTY]})
        finally:
            from app.services.database import PSQLDatabase
            await PSQLDatabase.close_pool()
    leaked = _run(_raw())
    assert {d.metadata["user_id"] for d, _ in leaked} & {ENT_A, ENT_B}, \
        "fixture cannot express the leak: the leaking arm returned no foreign row"


@needs_pg
def test_entity_route_post_filter_keeps_correct_data_byte_identical(env, monkeypatch):
    """On correct data the new /query/{entity_id} post-filter changes nothing: the owner's
    wire is byte-identical with the filter replaced by identity (and non-empty)."""
    from app.routes import document_routes as dr

    def call():
        return env.client.post(f"/query/{ENT_A}", json={"query": TERM, "k": 5},
                               headers=_tok([ENT_A]))

    with_filter = call()
    with monkeypatch.context() as m:
        # The TRUE identity: the same object back. `list(documents)` used to be one, but a result
        # now carries its declared score kind (ScoredHits, F-QUERY-SCORE-KIND-ON-THE-WIRE), and a
        # rebuilt plain list declares nothing -- so the byte comparison below now ALSO proves the
        # real filter carries the kind across unchanged.
        m.setattr(dr, "_authorized_only", lambda documents, entity_ids: documents)
        without_filter = call()

    assert with_filter.status_code == 200
    assert set(_texts(with_filter)) == {OWN_A, OWN_A2}
    assert with_filter.content == without_filter.content
