"""F-HYBRID-COMPOSE: hybrid retrieval proven END-TO-END on real pgvector, both arms live.

WHY THIS FILE EXISTS
--------------------
`HYBRID_SEARCH_ENABLED` defaults True, so every knowledge query runs a dense arm and a
keyword (FTS) arm and fuses them with RRF (`_hybrid_or_dense_search`, document_routes).
Each piece was covered on its own: #72 drives the keyword arm on a real DB but STUBS the
dense arm; #73 unit-tests `reciprocal_rank_fusion` on hand-built lists. Nothing proved the
COMPOSITION: that a real dense result and a real keyword result, fetched from the same
store through a real route, are fused so that a document matched by BOTH arms ranks first
while documents matched by only ONE arm are still returned.

THE FIXTURE IS BUILT SO THE ASSERTIONS CAN FAIL
-----------------------------------------------
Embeddings are a fixed text -> vector table (not hashes), so the dense ranking is chosen,
not incidental. Per arm, a SINGLE-arm document out-ranks the overlap document:

    dense   (cosine, top-3): DENSE_ONLY (0.0) > BOTH (~0.05) > NEAR (~0.2)    [KW_ONLY far]
    keyword (ts_rank_cd)   : KW_ONLY (term x3) > BOTH (term x1)               [others: no term]

So BOTH is first ONLY if the two lists are genuinely fused: dense-only puts DENSE_ONLY first,
keyword-only puts KW_ONLY first, and score-not-rank fusion would mix incompatible scales.
`test_premise_*` measures both arm orderings on the real DB before anything is concluded
from the fused order — if the premise drifts, that test fails rather than the fused test
passing by accident.

A FOREIGN document (owned by another entity) is the closest dense vector AND the strongest
keyword match, so an entitlement filter missing from EITHER arm would put it in the result.

Rerank is switched off (RERANK_ENABLED=False) for determinism, and `rerank` is replaced by a
tripwire so a rerank that ran anyway fails loudly instead of reordering silently.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so these provably ran.
Base: origin/main a4b47a6 — self-contained, does NOT depend on #72's fixtures.
"""
import asyncio
import datetime
import logging
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
    reason="RAG_TEST_PG_DSN not set: no pgvector for the composed hybrid path",
)

_SECRET = "test-secret-hybrid-compose"
_COLLECTION = "hybrid_compose"

TERM = "qorvexal9"  # the exact term; appears in BOTH, KW_ONLY and FOREIGN only

DENSE_ONLY = "treasury cash position narrative for the regional office"
BOTH = f"regional office treasury memo citing {TERM} in the appendix"
NEAR = "office supplies inventory for the regional branch"
KW_ONLY = f"{TERM} {TERM} {TERM} reconciliation checklist"
FAR = "holiday rota for the warehouse team"
FOREIGN = f"{TERM} {TERM} {TERM} {TERM} another entity's private note"

_VECTORS = {
    TERM: [1.0, 0.0, 0.0],  # the query
    DENSE_ONLY: [1.0, 0.0, 0.0],
    BOTH: [0.95, 0.31, 0.0],
    NEAR: [0.8, 0.6, 0.0],
    FAR: [0.3, 1.0, 0.0],
    KW_ONLY: [0.05, 0.0, 1.0],
    FOREIGN: [1.0, 0.0, 0.0],
}

OWNER, FOREIGN_ENTITY, EMPTY_ENTITY = "uA", "uB", "uC"
OWNER_FILE, FOREIGN_FILE = "kbA", "kbB"


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


@pytest.fixture()
def composed(monkeypatch):
    """Real pgvector store + document_tsv column; BOTH arms pointed at it; rerank off."""
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="sync")
    _real_post_init(store)
    owned = [DENSE_ONLY, BOTH, NEAR, KW_ONLY, FAR]
    store.add_documents(
        [Document(page_content=t, metadata={"file_id": OWNER_FILE, "user_id": OWNER,
                                            "tenant_id": "tA"}) for t in owned],
        ids=[f"a{i}" for i in range(len(owned))],
    )
    store.add_documents(
        [Document(page_content=FOREIGN, metadata={"file_id": FOREIGN_FILE,
                                                  "user_id": FOREIGN_ENTITY,
                                                  "tenant_id": "tA"})],
        ids=["b0"],
    )

    # Mirror external tempo migration 10081 (the column the keyword arm queries);
    # 'english' matches app.config.FTS_CONFIG's default.
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        c.commit()

    # Keyword arm: database.py bound `DSN` at import, so patch that name, drop the pool.
    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)

    astore = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)

    # Rerank OFF for determinism, and a tripwire in case it runs anyway.
    monkeypatch.setattr(dr, "RERANK_ENABLED", False)

    async def _rerank_tripwire(*a, **k):
        raise AssertionError("rerank ran although RERANK_ENABLED=False")
    monkeypatch.setattr(dr, "rerank", _rerank_tripwire)

    # Count real keyword-arm calls, so "the fused path ran" is measured, not assumed.
    calls = []
    real_kw = dr.keyword_search

    async def _counting_kw(*a, **k):
        # TestClient (used without `with`) runs every request on a FRESH event loop, and an
        # asyncpg pool is bound to the loop that created it. A pool reused across requests
        # leaves a query parked on a dead loop holding its table lock (measured: the next
        # fixture's DROP TABLE blocked forever). So each call gets a pool on its own loop
        # and closes it; production has one long-lived loop and one pool.
        calls.append(k.get("filters"))
        try:
            return await real_kw(*a, **k)
        finally:
            await PSQLDatabase.close_pool()
    monkeypatch.setattr(dr, "keyword_search", _counting_kw)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield TestClient(main.app), astore, calls

    if PSQLDatabase.pool is not None:  # only if a call escaped the per-call close above
        PSQLDatabase.pool.terminate()
        PSQLDatabase.pool = None


def _texts(resp):
    return [hit[0]["page_content"] for hit in resp.json()]


# Every query route reaches the same producer (_retrieve_documents). Each must return the
# same fused answer; a route that silently bypassed fusion would diverge here.
_ROUTES = {
    "query_by_entity": lambda c, k: c.post(
        f"/query/{OWNER}", json={"query": TERM, "k": k}, headers=_tok([OWNER])),
    "query_by_file": lambda c, k: c.post(
        "/query", json={"query": TERM, "file_id": OWNER_FILE, "k": k}, headers=_tok([OWNER])),
    "query_multiple": lambda c, k: c.post(
        "/query_multiple", json={"query": TERM, "file_ids": [OWNER_FILE], "k": k},
        headers=_tok([OWNER])),
}


@needs_pg
def test_hybrid_is_on_by_default():
    """The claim under test is the DEFAULT path, so it must not be switched on by the test."""
    from app.routes import document_routes as dr
    assert "HYBRID_SEARCH_ENABLED" not in os.environ, "test env overrides the default"
    assert dr.HYBRID_SEARCH_ENABLED is True


@needs_pg
def test_premise_each_arm_is_real_and_a_single_arm_doc_leads_each(composed):
    """Measure BOTH real arms on the real DB. In each arm a single-arm document out-ranks
    BOTH — which is what makes 'BOTH ranks first after fusion' a claim that can fail."""
    from app.routes import document_routes as dr
    from app.services.database import PSQLDatabase
    from app.services.hybrid_search import keyword_search

    _client, astore, _calls = composed
    lc_filter = dr._to_langchain_filter({"user_id": [OWNER]})

    async def _both():
        dense = await astore.asimilarity_search_with_score_by_vector(
            _VECTORS[TERM], k=3, filter=lc_filter)
        try:
            kw = await keyword_search(TERM, k=3, filters={"user_id": [OWNER]})
        finally:
            await PSQLDatabase.close_pool()
        return dense, kw

    dense, kw = _run(_both())
    dense_t = [d.page_content for d, _ in dense]
    kw_t = [d.page_content for d, _ in kw]

    assert dense_t == [DENSE_ONLY, BOTH, NEAR], dense_t
    assert kw_t == [KW_ONLY, BOTH], kw_t
    assert kw[0][1] > kw[1][1] > 0, "ts_rank_cd must rank KW_ONLY strictly above BOTH"
    assert KW_ONLY not in dense_t and DENSE_ONLY not in kw_t, "arms must not overlap beyond BOTH"


@needs_pg
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_fused_route_ranks_the_overlap_first_and_keeps_single_arm_docs(composed, caplog, route):
    """ACCEPTANCE: BOTH first; DENSE_ONLY and KW_ONLY both still returned; NEAR (dense
    rank 3, no keyword match) is the one squeezed out at k=3. No fallback happened."""
    client, _astore, calls = composed
    with caplog.at_level(logging.WARNING):
        r = _ROUTES[route](client, 3)
    assert r.status_code == 200, r.text
    got = _texts(r)

    assert calls, "the keyword arm was never called: this was not the fused path"
    assert not any("keyword search failed" in rec.getMessage() for rec in caplog.records), \
        "keyword arm fell back to dense-only; the result below is not a fusion"

    assert got[0] == BOTH, got
    assert set(got[1:]) == {DENSE_ONLY, KW_ONLY}, got
    assert FOREIGN not in got

    scores = [hit[1] for hit in r.json()]
    assert scores[0] > scores[1] >= scores[2], scores


@needs_pg
def test_fused_path_honours_the_entitlement_on_both_arms(composed):
    """FOREIGN is the nearest vector AND the strongest keyword match. It must reach its
    owner and nobody else; an unentitled request is refused; an entity with no documents
    gets [] from the fused path (the keyword arm was called, and still found nothing)."""
    client, _astore, calls = composed

    # Positive control: FOREIGN is retrievable, so its absence elsewhere means something.
    r_own = client.post(f"/query/{FOREIGN_ENTITY}", json={"query": TERM, "k": 3},
                        headers=_tok([FOREIGN_ENTITY]))
    assert r_own.status_code == 200, r_own.text
    assert _texts(r_own) == [FOREIGN]

    # Owner's fused result never carries the foreign doc (covered per-route above too).
    r_a = client.post(f"/query/{OWNER}", json={"query": TERM, "k": 6}, headers=_tok([OWNER]))
    assert r_a.status_code == 200
    assert FOREIGN not in _texts(r_a)
    assert {h[0]["metadata"]["user_id"] for h in r_a.json()} == {OWNER}

    # Cross-entity request: refused before retrieval (measured: 403, not a filtered 200).
    calls.clear()
    r_x = client.post(f"/query/{OWNER}", json={"query": TERM, "k": 3},
                      headers=_tok([FOREIGN_ENTITY]))
    assert r_x.status_code == 403, r_x.text
    assert calls == [], "a refused request must not reach retrieval"

    # DENIAL through the fused path: entitled to an entity that owns nothing -> [].
    calls.clear()
    r_empty = client.post(f"/query/{EMPTY_ENTITY}", json={"query": TERM, "k": 3},
                          headers=_tok([EMPTY_ENTITY]))
    assert r_empty.status_code == 200, r_empty.text
    assert r_empty.json() == []
    assert calls == [{"user_id": EMPTY_ENTITY}], calls

    # /query_multiple with a foreign file id: the entitlement filter, not the file id, wins.
    r_m = client.post("/query_multiple",
                      json={"query": TERM, "file_ids": [FOREIGN_FILE], "k": 3},
                      headers=_tok([OWNER]))
    assert r_m.status_code == 200, r_m.text
    assert r_m.json() == []


@needs_pg
def test_keyword_arm_failure_degrades_to_dense_only_and_is_logged(composed, caplog):
    """FAILURE: a REAL keyword-arm failure (the FTS column is missing, as on a store without
    tempo migration 10081) -> exactly the dense top-k, 200, and a logged warning. Never an
    error, never empty."""
    client, _astore, calls = composed
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("ALTER TABLE langchain_pg_embedding DROP COLUMN document_tsv")
        c.commit()

    with caplog.at_level(logging.WARNING):
        r = client.post(f"/query/{OWNER}", json={"query": TERM, "k": 3}, headers=_tok([OWNER]))

    assert r.status_code == 200, r.text
    assert calls, "precondition: the keyword arm was attempted"
    assert _texts(r) == [DENSE_ONLY, BOTH, NEAR], _texts(r)
    assert KW_ONLY not in _texts(r), "a keyword-only doc cannot survive a dead keyword arm"
    assert any("keyword search failed" in rec.getMessage() for rec in caplog.records), \
        "a failed keyword arm must be surfaced in the log"
