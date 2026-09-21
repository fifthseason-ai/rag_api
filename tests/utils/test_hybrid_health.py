"""F-HYBRID-HEALTH: a missing keyword arm must be visible on /health, on a REAL pgvector store.

MEASURED before this change: /health ran only `SELECT 1`, and `retrieve()` swallows a keyword-arm
failure into a dense-only answer plus one log line per query. With `document_tsv` absent the service
answered UP and nothing an operator watches said keyword search was gone.

The store here is built by langchain itself, which creates `langchain_pg_embedding` WITHOUT
`document_tsv` -- the real degraded shape, not a simulated one. Adding the column (and then the GIN
index) the way tempo migration 10081 does moves the same database to healthy. The probe is never
patched: a green result comes from real catalog SQL.

Needs a Postgres with pgvector (RAG_TEST_PG_DSN); RAG_TEST_PG_REQUIRED=1 makes absence an error.
"""

import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
_SECRET = "hybrid_health_secret"

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the keyword-arm health probe",
)


class _DetEmb(Embeddings):
    def _v(self, t):
        import hashlib

        h = hashlib.sha256(t.encode()).digest()
        return [((h[i % len(h)] / 255.0) * 2 - 1) for i in range(16)]

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def embed_query(self, t):
        return self._v(t)


def _raw_dsn():
    return PG_DSN.replace("postgresql+psycopg2://", "postgresql://")


def _sql(statement):
    import psycopg2

    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute(statement)
        c.commit()


def _add_tsv_column():
    _sql(
        "ALTER TABLE langchain_pg_embedding ADD COLUMN document_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(document, ''))) STORED"
    )


def _add_tsv_gin():
    _sql("CREATE INDEX idx_hh_document_tsv ON langchain_pg_embedding USING gin (document_tsv)")


def _tok(uid="uA"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid, "tid": "tA", "ent": [uid], "act": ["read"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


@pytest.fixture()
def store(monkeypatch):
    """A real langchain-built store (no document_tsv) holding one row owned by uA."""
    import asyncpg
    from app.routes import document_routes as dr
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store

    _sql("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")

    def _real_post_init(self):
        # tests/conftest.py no-ops PGVector.__post_init__; run the real one so the tables exist.
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

    os.environ["JWT_SECRET"] = _SECRET
    dsn = _raw_dsn().replace("postgresql://", "postgresql+psycopg2://", 1)
    sync = get_vector_store(dsn, _DetEmb(), "hybrid_health", mode="sync")
    _real_post_init(sync)
    sync.add_documents(
        [Document(page_content="quarterly revenue grew", metadata={"file_id": "fA", "user_id": "uA", "tenant_id": "tA"})],
        ids=["fA"],
    )
    astore = get_vector_store(dsn, _DetEmb(), "hybrid_health", mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)

    # One fresh pool per call: TestClient runs each request on its own event loop, and a pool
    # reused across loops parks connections on a dead loop (measured in F-HYBRID-COMPOSE).
    pools = []

    async def _fresh_pool():
        pool = await asyncpg.create_pool(dsn=_raw_dsn(), min_size=1, max_size=2)
        pools.append(pool)
        return pool

    monkeypatch.setattr(PSQLDatabase, "get_pool", _fresh_pool)

    async def _up():
        return True

    monkeypatch.setattr(dr, "is_health_ok", _up)  # overall status is not under test here
    # Rerank is on by default and would try Bedrock; keep this test provider-free and deterministic.
    monkeypatch.setattr(dr, "_cohere_rerank_enabled", lambda args: False)
    import main

    if getattr(main.app.state, "thread_pool", None) is None:
        main.app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    yield TestClient(main.app)
    for pool in pools:
        try:
            pool.terminate()
        except RuntimeError:  # its request's loop is already closed; the sockets go with it
            pass
    _sql("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")


@needs_pg
def test_hybrid_is_on_by_default():
    """The probe matters because the keyword arm runs by default; pin that premise."""
    from app import config

    assert "HYBRID_SEARCH_ENABLED" not in os.environ
    assert config.HYBRID_SEARCH_ENABLED is True


@needs_pg
def test_missing_column_is_reported_degraded(store):
    r = store.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "UP"  # additive: dense still serves, so status does not flip
    assert body["keyword_search"] == {"state": "degraded", "reason": "document_tsv column absent"}


@needs_pg
def test_column_without_gin_is_reported_degraded(store):
    _add_tsv_column()
    body = store.get("/health").json()
    assert body["keyword_search"] == {"state": "degraded", "reason": "document_tsv GIN index absent"}


@needs_pg
def test_column_and_gin_present_is_available(store):
    _add_tsv_column()
    _add_tsv_gin()
    r = store.get("/health")
    assert r.status_code == 200
    assert r.json()["keyword_search"] == {"state": "available"}
    assert r.json()["status"] == "UP"


@needs_pg
def test_query_still_answers_dense_only_when_degraded(store, caplog):
    """No query-path change: with the column absent, /query is 200 with the real dense hit and the
    keyword failure is logged -- the same answer it gave before this change."""
    import logging

    assert store.get("/health").json()["keyword_search"]["state"] == "degraded"
    with caplog.at_level(logging.WARNING):
        r = store.post("/query", json={"query": "quarterly revenue", "file_id": "fA", "k": 3}, headers=_tok())
    assert r.status_code == 200, r.text
    hits = r.json()
    assert len(hits) == 1
    assert "quarterly revenue grew" in str(hits[0])
    assert any("keyword" in rec.getMessage().lower() for rec in caplog.records)


@needs_pg
def test_probe_failure_is_unknown_not_an_error(monkeypatch):
    """A probe that cannot reach the database answers `unknown` with a fixed reason -- never raises,
    never echoes the exception (/health is unauthenticated)."""
    from app.services import hybrid_search
    from app.services.database import PSQLDatabase

    async def _boom():
        raise OSError("connect to secret-host:5432 failed")

    monkeypatch.setattr(PSQLDatabase, "get_pool", _boom)
    state = asyncio.run(hybrid_search.keyword_search_state())
    assert state == {"state": "unknown", "reason": "keyword search probe failed"}
    assert "secret-host" not in str(state)


def test_disabled_flag_reports_disabled_without_probing(monkeypatch):
    from app.utils import health

    async def _must_not_run():
        raise AssertionError("probe ran although hybrid search is disabled")

    monkeypatch.setattr(health, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(health, "keyword_search_state", _must_not_run)
    assert asyncio.run(health.keyword_search_health()) == {"state": "disabled"}
