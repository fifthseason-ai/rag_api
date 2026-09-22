"""P06-4 item 3 — EXPECTED SOURCE VERSION at retrieval time, on a REAL pgvector store.

Two versions of the SAME synthetic source are ingested (v2 via replace=true). A query
must return the CURRENT version's passage with the CURRENT version's provenance
(`ingest_id`), and the SUPERSEDED version must not leak in as a second hit -- on the
dense leg AND on the keyword (FTS) leg. This is the retrieval-side face of the P06
outcome "open the correct source version".

WHAT IS GENUINELY NEW (not covered by test_ingest_id / test_replace_not_accumulate)
--------------------------------------------------------------------------------
Those suites prove the STORE bookkeeping on a FakeStore: a successful replace leaves
only the new ingest_id among the rows, and /query echoes the stored id (with a
monkeypatched `_retrieve_documents` that returns every row). They do NOT prove, on a
REAL pgvector store with the real retrieval path, that:
  * a query for the current content returns the current passage + current ingest_id;
  * the superseded passage / ingest_id is physically gone and cannot come back as a
    second hit through EITHER leg;
  * a term that existed ONLY in v1 no longer matches on the keyword arm after replace.

RED-FIRST. The negative is load-bearing only if the superseded rows would otherwise be
retrievable. test_control_without_deletion_v1_leaks disables `delete_rows_by_uuid` and
shows BOTH versions become retrievable -- so the deletion the replace path performs
(document_routes.py ~2734-2742) is what makes the supersession real, and the main test
would redden if it were skipped.

SYNTHETIC documents only (built here); no real source (A3/A4 blocker).

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it; RAG_TEST_PG_REQUIRED=1
turns "no DSN" from a skip into an error so this provably ran.
"""
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
    reason="RAG_TEST_PG_DSN not set: no pgvector for version-supersession retrieval",
)

_SECRET = "test-secret-supersession"
# Built on the P06-2 collection shapes: same source identity across the two versions
# (stable file_id + filename), version distinguished by ingest_id, owned by the P06-2
# entity/tenant (P06-2-INGESTION-RECEIPT-2026-09-22T221458Z.md).
FID = "kn-vers-src"
FILENAME = "astra-brief.txt"   # SAME filename across versions: one source, two versions
ENT = "ent-knowledge-1"
TENANT = "tenant-vivaldi"

# v1 and v2 share topic wording (so the query text is close to both) but differ in the
# value AND each carries a version-unique rare term the keyword arm can pin.
V1_TERM = "onlyinvone11"
V2_TERM = "onlyinvtwo22"
V1_TEXT = (f"fiscal 2023 revenue narrative {V1_TERM}. "
           + "the reported revenue figure was one hundred. " * 8)
V2_TEXT = (f"fiscal 2023 revenue narrative {V2_TERM}. "
           + "the reported revenue figure was two hundred fifty. " * 8)
QUERY = "fiscal 2023 revenue narrative"


def _hdr(act=("read", "write")):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "uploader-user", "tid": TENANT, "ent": [ENT], "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _real_store(monkeypatch, collection):
    """Real async pgvector store + the document_tsv FTS column + the keyword pool
    repointed at the test DB, so BOTH retrieval legs run against real pg."""
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store
    from tests.utils.test_empty_entitlement_query_path import _DetEmb
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store

    raw = PG_DSN.replace("postgresql+psycopg2://", "postgresql://")
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
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr("app.config.vector_store", store, raising=False)
    monkeypatch.setattr(document_routes, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(document_routes, "RERANK_ENABLED", False)
    return TestClient(app)


def _embed(client, text, name, replace=None):
    data = {"file_id": FID, "entity_id": ENT}
    if replace is not None:
        data["replace"] = str(replace).lower()
    return client.post("/embed", data=data, headers=_hdr(),
                       files={"file": (name, io.BytesIO(text.encode()), "text/plain")})


def _real_rows(store):
    from app.services.vector_store.extended_pg_vector import ExtendedPgVector
    return ExtendedPgVector.get_row_uuids(store, FID)


def _teardown(store):
    from app.services.database import PSQLDatabase
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(PSQLDatabase.close_pool())
        loop.close()
    except Exception:
        PSQLDatabase.pool = None


def _hits(resp):
    return resp.json()


def _contents(resp):
    return [h[0]["page_content"] for h in resp.json()]


def _ingest_ids(resp):
    return {h[0]["metadata"].get("ingest_id") for h in resp.json()}


@needs_pg
def test_query_returns_current_version_and_superseded_does_not_leak(monkeypatch):
    store = _real_store(monkeypatch, "vers_source")
    try:
        client = _client(monkeypatch, store)
        assert _embed(client, V1_TEXT, FILENAME).status_code == 200
        # capture v1's ingest id from a query BEFORE replace (precondition/positive control)
        pre = client.post("/query", json={"query": QUERY, "file_id": FID, "k": 10,
                                          "entity_id": ENT}, headers=_hdr())
        assert pre.status_code == 200 and any(V1_TERM in c for c in _contents(pre)), pre.text
        v1_ids = _ingest_ids(pre)

        # replace with v2
        assert _embed(client, V2_TEXT, FILENAME, replace=True).status_code == 200

        # dense/fused query for the shared topic -> current version only
        post = client.post("/query", json={"query": QUERY, "file_id": FID, "k": 10,
                                           "entity_id": ENT}, headers=_hdr())
        assert post.status_code == 200, post.text
        contents = _contents(post)
        assert any(V2_TERM in c for c in contents), ("current version missing", contents)
        assert all(V1_TERM not in c for c in contents), ("superseded version leaked", contents)
        assert all("one hundred" not in c for c in contents), contents

        post_ids = _ingest_ids(post)
        assert len(post_ids) == 1, ("more than one version's ingest_id present", post_ids)
        assert post_ids.isdisjoint(v1_ids), ("still carrying v1's ingest_id", post_ids, v1_ids)

        # the table physically holds only v2's rows
        assert len(_real_rows(store)) >= 1
    finally:
        _teardown(store)


@needs_pg
def test_keyword_leg_no_longer_matches_a_v1_only_term_after_replace(monkeypatch):
    """The version-unique v1 term matched on the keyword arm before replace; after replace
    it matches nothing, proving the superseded rows are gone from the FTS leg too."""
    store = _real_store(monkeypatch, "vers_source_kw")
    try:
        client = _client(monkeypatch, store)
        assert _embed(client, V1_TEXT, FILENAME).status_code == 200
        before = client.post("/query", json={"query": V1_TERM, "file_id": FID, "k": 10,
                                             "entity_id": ENT}, headers=_hdr())
        assert before.status_code == 200 and any(V1_TERM in c for c in _contents(before)), \
            ("precondition: v1 term retrievable before replace", before.text)

        assert _embed(client, V2_TEXT, FILENAME, replace=True).status_code == 200

        after = client.post("/query", json={"query": V1_TERM, "file_id": FID, "k": 10,
                                            "entity_id": ENT}, headers=_hdr())
        assert after.status_code == 200, after.text
        assert all(V1_TERM not in c for c in _contents(after)), \
            ("v1-only term still matches after replace", _contents(after))
    finally:
        _teardown(store)


@needs_pg
def test_control_without_deletion_v1_leaks(monkeypatch):
    """POSITIVE CONTROL / red-first made permanent: with the superseded-row deletion
    disabled, BOTH versions become retrievable and two ingest_ids appear -- so the main
    test's single-version assertion is load-bearing, not vacuous."""
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    store = _real_store(monkeypatch, "vers_source_ctrl")
    try:
        async def _no_delete(self, row_uuids, executor=None):
            return 0  # pretend nothing was removed
        monkeypatch.setattr(AsyncPgVector, "delete_rows_by_uuid", _no_delete)

        client = _client(monkeypatch, store)
        assert _embed(client, V1_TEXT, FILENAME).status_code == 200
        assert _embed(client, V2_TEXT, FILENAME, replace=True).status_code == 200

        r = client.post("/query", json={"query": QUERY, "file_id": FID, "k": 10,
                                        "entity_id": ENT}, headers=_hdr())
        assert r.status_code == 200, r.text
        contents = _contents(r)
        assert any(V1_TERM in c for c in contents) and any(V2_TERM in c for c in contents), \
            ("with deletion disabled BOTH versions must be retrievable", contents)
        assert len(_ingest_ids(r)) == 2, _ingest_ids(r)
    finally:
        _teardown(store)
