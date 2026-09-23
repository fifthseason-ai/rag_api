"""P06-4 — `ingest_id` on EVERY retrieved chunk is the RECEIPT JOIN KEY (load-bearing).

WHY THIS IS LOAD-BEARING (KD-INDEX-BLOCK-CONSUMER, Integration decision 2026-09-22):
Core resolves completeness at ANSWER time by JOINING a retrieved chunk's `ingest_id` to
the `/embed` `index` receipt it persists per `ingest_id`. So `ingest_id` on every stored
chunk is the join key the whole incomplete-state story depends on: if a chunk ever
lacked it, Core's join returns nothing and an `unverified`/`partial` source would
silently read as COMPLETE -- a false success one hop downstream of rag_api. This test
exists to make that impossible, named for the property it protects so it is not deleted
as redundant to test_ingest_id.py (which pins uuid4-ness and the additive/replace
bookkeeping, NOT "present == receipt on every hit from every route on real pg").

WHAT IT PROVES
--------------
On a REAL pgvector store, one source ingested via the real `POST /embed`:
  * the `/embed` receipt carries `index.ingest_id`;
  * EVERY hit from EVERY query route (/query, /query/{entity_id}, /query_multiple)
    carries `metadata.ingest_id`, and it EQUALS the receipt's `ingest_id`.

RED-FIRST / non-vacuity
-----------------------
  * PERMANENT in-file control (test_control_without_the_stamp_the_join_key_is_absent):
    with `_prepare_documents_sync` patched to drop the stamp, the join key is absent on
    the retrieved chunks -- so the positive assertion above genuinely can fail.
  * LIVE SOURCE-MUTATION control (executed in the slot phase, recorded in the receipt):
    neutralise the `ingest_id` stamp in `_prepare_documents_sync` at source (md5 the file
    before/after), run this file -> the named test reddens; `git checkout --` restore ->
    green. That proves the guard is real against the shipped code, not only a monkeypatch.

Built on the P06-2 collection shapes (ent-knowledge-1 / tenant-vivaldi / kn-* / filename).
SYNTHETIC only (A3/A4 blocker). Needs a Postgres with pgvector; RAG_TEST_PG_DSN selects
it, RAG_TEST_PG_REQUIRED=1 turns "no DSN" into an error so this provably ran.
"""
import asyncio
import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the ingest_id join-key proof",
)

_SECRET = "test-secret-joinkey"
# P06-2 collection shapes.
ENT = "ent-knowledge-1"
TENANT = "tenant-vivaldi"
FID = "kn-src-brief"
FILENAME = "astra-brief.txt"

# Long enough to split into several chunks so "every chunk" is a real quantifier.
QUERY = "astra knowledge launch readiness"
TEXT = "\n\n".join(
    f"Section {i}. {QUERY} synthetic passage for the receipt join-key proof. "
    + ("synthetic words " * 60)
    for i in range(6)
)


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _hdr(act=("read", "write")):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "uploader-user", "tid": TENANT, "ent": [ENT], "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _real_store(monkeypatch, collection):
    """Real async pgvector store + document_tsv FTS column + keyword pool repointed, so
    both legs of every route run against real pg."""
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store
    from tests.utils.test_empty_entitlement_query_path import _DetEmb
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()
    dsn = PG_DSN.replace("postgresql://", "postgresql+psycopg2://", 1)
    store = get_vector_store(dsn, _DetEmb(), collection, mode="async")
    if store.create_extension:
        store.create_vector_extension()
    store.EmbeddingStore, store.CollectionStore = _get_embedding_collection_store(
        store._embedding_length, use_jsonb=store.use_jsonb
    )
    store.create_tables_if_not_exists()
    store.create_collection()
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

    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)
    return store


def _client(monkeypatch, store):
    from app.services.database import PSQLDatabase
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr("app.config.vector_store", store, raising=False)
    monkeypatch.setattr(document_routes, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(document_routes, "RERANK_ENABLED", False)

    # TestClient uses a fresh loop per request: close the asyncpg keyword pool after each
    # keyword call so it rebuilds on the request's own loop (mirror F-ENTITLEMENT-FUSED),
    # else the stale pool errors and leaks connections that deadlock a later DROP.
    real_kw = document_routes.keyword_search

    async def _kw(*a, **k):
        try:
            return await real_kw(*a, **k)
        finally:
            await PSQLDatabase.close_pool()
    monkeypatch.setattr(document_routes, "keyword_search", _kw)
    return TestClient(app)


def _embed(client):
    return client.post("/embed", data={"file_id": FID, "entity_id": ENT}, headers=_hdr(),
                       files={"file": (FILENAME, io.BytesIO(TEXT.encode()), "text/plain")})


def _teardown():
    from app.services.database import PSQLDatabase
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(PSQLDatabase.close_pool())
        loop.close()
    except Exception:
        PSQLDatabase.pool = None


# Every query route, each naming the same source. All must carry the join key on every hit.
_ROUTES = {
    "query_by_file": lambda c: c.post(
        "/query", json={"query": QUERY, "file_id": FID, "k": 20, "entity_id": ENT}, headers=_hdr()),
    "query_by_entity": lambda c: c.post(
        f"/query/{ENT}", json={"query": QUERY, "k": 20}, headers=_hdr()),
    "query_multiple": lambda c: c.post(
        "/query_multiple", json={"query": QUERY, "file_ids": [FID], "k": 20}, headers=_hdr()),
}


@needs_pg
def test_every_retrieved_chunk_carries_ingest_id_the_receipt_join_key(monkeypatch):
    store = _real_store(monkeypatch, "joinkey")
    try:
        client = _client(monkeypatch, store)
        emb = _embed(client)
        assert emb.status_code == 200, emb.text
        receipt_ingest = emb.json()["index"]["ingest_id"]
        assert receipt_ingest, "precondition: the /embed receipt carries index.ingest_id"

        for name, call in _ROUTES.items():
            r = call(client)
            assert r.status_code == 200, (name, r.text)
            hits = r.json()
            assert hits, (name, "precondition: the source is retrievable on this route")
            for hit in hits:
                got = hit[0]["metadata"].get("ingest_id")
                assert got == receipt_ingest, (
                    "route %s returned a chunk whose ingest_id (%r) is missing or does not "
                    "equal the /embed receipt's ingest_id (%r) -- Core's completeness JOIN "
                    "would fail and an unverified/partial source could read as complete"
                    % (name, got, receipt_ingest)
                )
    finally:
        _teardown()


@needs_pg
def test_control_without_the_stamp_the_join_key_is_absent(monkeypatch):
    """PERMANENT non-vacuity: patch _prepare_documents_sync to drop the ingest_id stamp;
    the retrieved chunks then lack the join key, so the named test above genuinely can
    fail. (The stronger live SOURCE mutation with md5 before/after is run in the slot
    phase and recorded in the receipt.)"""
    real_prepare = document_routes._prepare_documents_sync

    def _prepare_without_stamp(*args, **kwargs):
        # Call the real preparer unchanged (avoids any positional/keyword collision on
        # ingest_id), then strip the stamp from the output -- the join key is neutralised
        # exactly as a missing stamp would leave it.
        docs = real_prepare(*args, **kwargs)
        for d in docs:
            d.metadata.pop("ingest_id", None)
        return docs

    store = _real_store(monkeypatch, "joinkey_ctrl")
    try:
        monkeypatch.setattr(document_routes, "_prepare_documents_sync", _prepare_without_stamp)
        client = _client(monkeypatch, store)
        assert _embed(client).status_code == 200
        r = _ROUTES["query_by_file"](client)
        assert r.status_code == 200 and r.json(), r.text
        assert any(hit[0]["metadata"].get("ingest_id") is None for hit in r.json()), (
            "control expected the join key to be ABSENT with the stamp neutralised, but "
            "every chunk still carried an ingest_id -- the positive test would be vacuous"
        )
    finally:
        _teardown()
