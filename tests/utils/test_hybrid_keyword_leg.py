"""F-HYBRID-DSN: the hybrid KEYWORD (FTS) leg, exercised against a real Postgres.

WHY THIS FILE EXISTS
--------------------
`HYBRID_SEARCH_ENABLED` defaults **True** (app/config.py), so every `/query` runs a
dense arm AND a keyword arm (`app/services/hybrid_search.keyword_search`) fused by RRF.
Yet no DB test exercised the keyword arm: it reads its connection from
`app.services.database.PSQLDatabase.get_pool()` -> `asyncpg.create_pool(dsn=DSN)` where
`DSN` is `app.config.DSN` (database.py:3,12), and `tests/conftest.py` hard-sets
`DB_HOST=localhost` / `DSN=dummy://` for the whole session. The existing real-pg tests
monkeypatch only the DENSE `vector_store` to the test DB; the keyword pool still dials
localhost:5432, fails, and `retrieve()` logs a warning and falls back to dense-only. So
the keyword arm silently never ran under test.

PRODUCTION IS NOT AFFECTED (measured): in a deployment both arms derive from the SAME
`connection_suffix` -> `DB_HOST`/`DB_PORT`/`POSTGRES_*` (config.py:242-247: `CONNECTION_STRING`
for the dense store, `DSN` for this pool). The localhost split is a TEST-INFRA artifact,
not a product defect. This file closes the coverage gap it exposed.

WHAT IT PROVES
--------------
* the keyword leg actually reaches the DB and matches an exact term dense embeddings miss;
* the keyword leg honours the SAME entitlement (user_id) filter as the dense leg — no
  cross-entity leak through the keyword arm;
* a keyword-only match surfaces in the fused `/query` result, and removing the keyword arm
  removes it (so the arm's contribution is real, not incidental);
* a failing keyword leg is SURFACED (logged) and degrades to dense-only, never a 500 and
  never silent;
* POSITIVE CONTROL: under conftest's localhost DSN the keyword leg cannot reach the DB —
  which is exactly why the arm was dark, and why the repoint below is load-bearing.

The `document_tsv` generated column is created here to mirror the external tempo migration
10081 (`to_tsvector('english', document)`, matching FTS_CONFIG='english'); rag_api depends
on that column existing but does not own its DDL.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so the leg provably ran.
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
    reason="RAG_TEST_PG_DSN not set: no pgvector for the real keyword leg",
)

_SECRET = "test-secret-hybrid"


class _DetEmb:
    """Deterministic hash embeddings (same shape as the empty-ent test's stub)."""

    def __init__(self, size=16):
        self.size = size

    def _v(self, t):
        import hashlib

        h = hashlib.sha256(t.encode()).digest()
        return [b / 255.0 for b in h[: self.size]]

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def embed_query(self, t):
        return self._v(t)


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _tok(entity_ids, tid="tA", uid="uA", act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid, "tid": tid, "ent": list(entity_ids), "act": list(act),
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


# The one exact term that appears ONLY in docB. Dense hash-embeddings do not
# privilege it; the keyword arm (to_tsquery) matches it precisely.
RARE_TERM = "zxqvff7"
DOC_A = "alpha quarterly summary overview of the general ledger"
DOC_B = f"beta remittance advice contains the marker {RARE_TERM} on the invoice"


@pytest.fixture()
def hybrid_db(monkeypatch):
    """A real pgvector store with a document_tsv column, and the KEYWORD pool
    repointed at the same DB (undoing conftest's localhost DSN)."""
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store

    raw = _raw_dsn()

    # Fresh tables.
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _DetEmb(), "hybrid_kw", mode="sync")
    _real_post_init(store)
    store.add_documents(
        [Document(page_content=DOC_A, metadata={"file_id": "fA", "user_id": "uA", "tenant_id": "tA"})],
        ids=["fA"],
    )
    store.add_documents(
        [Document(page_content=DOC_B, metadata={"file_id": "fB", "user_id": "uA", "tenant_id": "tA"})],
        ids=["fB"],
    )

    # Mirror external tempo migration 10081: the generated FTS column the keyword
    # arm queries. 'english' matches app.config.FTS_CONFIG's default.
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_tsv "
            "ON langchain_pg_embedding USING gin (document_tsv)"
        )
        c.commit()

    # Repoint the KEYWORD pool at the test DB. database.py did `from app.config import DSN`,
    # so the name to patch is `app.services.database.DSN`; reset the cached pool so it is
    # rebuilt against the new DSN.
    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)

    astore = get_vector_store(_sqlalchemy_dsn(), _DetEmb(), "hybrid_kw", mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)

    import main
    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    yield TestClient(main.app), astore

    # Teardown: drop the pool built against the test DSN so it can't leak into the
    # next test (which may run under the conftest localhost DSN again).
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(PSQLDatabase.close_pool())
        loop.close()
    except Exception:
        PSQLDatabase.pool = None


@needs_pg
def test_keyword_leg_runs_and_matches_the_exact_term(hybrid_db):
    """The keyword arm reaches the DB and matches the exact term, in docB only."""
    from app.services.hybrid_search import keyword_search

    loop = asyncio.new_event_loop()
    try:
        hits = loop.run_until_complete(
            keyword_search(RARE_TERM, k=5, filters={"user_id": ["uA"]})
        )
    finally:
        loop.close()

    contents = [d.page_content for d, _s in hits]
    assert any(RARE_TERM in c for c in contents), (RARE_TERM, contents)
    assert all(DOC_A != c for c in contents), "docA has no keyword match and must not appear"
    assert all(s > 0 for _d, s in hits), "ts_rank_cd score must be positive for a match"


@needs_pg
def test_keyword_leg_respects_the_entitlement_filter(hybrid_db):
    """CROSS-ENTITY on the keyword arm: a foreign entity gets nothing even though the
    exact term matches docB — the same user_id filter constrains both arms."""
    from app.services.hybrid_search import keyword_search

    loop = asyncio.new_event_loop()
    try:
        own = loop.run_until_complete(
            keyword_search(RARE_TERM, k=5, filters={"user_id": ["uA"]})
        )
        foreign = loop.run_until_complete(
            keyword_search(RARE_TERM, k=5, filters={"user_id": ["someone-else"]})
        )
    finally:
        loop.close()

    assert own, "precondition: the owner sees the keyword match"
    assert foreign == [], "the keyword arm leaked docB to a foreign entity"


@needs_pg
def test_hybrid_route_surfaces_a_keyword_only_match_and_logs_any_failure(hybrid_db, monkeypatch, caplog):
    """Through /query: with the dense arm stubbed to return ONLY docA, docB still
    appears — contributed solely by the keyword arm. Remove the keyword arm and docB
    is gone, the response is still 200 (dense-only), and the fallback is LOGGED."""
    import logging
    client, astore = hybrid_db

    # Force the dense arm to miss docB by construction: it returns only docA.
    async def _dense_only_docA(embedding, k, filter=None, executor=None):
        return [(Document(page_content=DOC_A,
                          metadata={"file_id": "fA", "user_id": "uA", "tenant_id": "tA"}), 0.1)]
    monkeypatch.setattr(astore, "asimilarity_search_with_score_by_vector", _dense_only_docA)

    # Keyword arm live -> docB surfaces from the keyword arm alone. `/query/{entity_id}`
    # is the entity-wide route (no file_id); it filters user_id=entity_id on both arms.
    r = client.post("/query/uA", json={"query": RARE_TERM, "k": 5}, headers=_tok(["uA"]))
    assert r.status_code == 200, r.text
    bodies = [hit[0]["page_content"] for hit in r.json()]
    assert any(RARE_TERM in b for b in bodies), ("keyword-only match missing", bodies)

    # Control: break the keyword arm. docB disappears (dense-only had only docA), the
    # request still succeeds, and the failure is surfaced in the log — never silent.
    from app.routes import document_routes as dr

    async def _boom(*a, **k):
        raise RuntimeError("keyword arm forced down")
    monkeypatch.setattr(dr, "keyword_search", _boom)

    with caplog.at_level(logging.WARNING):
        r2 = client.post("/query/uA", json={"query": RARE_TERM, "k": 5}, headers=_tok(["uA"]))
    assert r2.status_code == 200, r2.text
    bodies2 = [hit[0]["page_content"] for hit in r2.json()]
    assert all(RARE_TERM not in b for b in bodies2), "docB survived without the keyword arm"
    assert any("keyword search failed" in rec.getMessage() for rec in caplog.records), \
        "a failed keyword leg must be surfaced in the log, not silently dropped"


@needs_pg
def test_control_localhost_dsn_cannot_reach_the_db(hybrid_db, monkeypatch):
    """POSITIVE CONTROL / red-first, made permanent: under conftest's localhost DSN the
    keyword pool cannot reach the DB. This is exactly why the arm was dark before the
    fixture's repoint — proving the repoint is load-bearing, not decorative."""
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.hybrid_search import keyword_search

    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", "postgresql://myuser:mypassword@localhost:5432/mydatabase")

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(Exception):
            loop.run_until_complete(keyword_search(RARE_TERM, k=5, filters={"user_id": ["uA"]}))
    finally:
        PSQLDatabase.pool = None
        loop.close()
