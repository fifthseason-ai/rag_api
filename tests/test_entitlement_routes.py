"""Route-level entitlement enforcement (KSPT-01, D-KSPT-1).

Negative space: a caller-supplied id outside the token entitlement must never be
read, embedded or deleted; an action not granted by the token must be refused; a
document owned by an unauthorized entity must never be returned.
"""
import io
import os
import datetime

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector

client = TestClient(app)

SECRET = "testsecret"

# Mutable owner used by the retrieval/get dummies so a test can simulate a
# document owned by another entity.
OWNER = {"user_id": "testuser"}


def hdr(ent, act, tid="tenantA", uid="testuser"):
    os.environ["JWT_SECRET"] = SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ent,
        "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, SECRET, algorithm='HS256')}"}


@pytest.fixture(autouse=True)
def _patch_store(monkeypatch):
    os.environ["JWT_SECRET"] = SECRET
    OWNER["user_id"] = "testuser"

    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3]
    )

    async def dummy_get_all_ids(self, executor=None):
        return ["testid1", "testid2"]

    async def dummy_get_filtered_ids(
        self, ids, user_id=None, document_origin_type=None, subscription_id=None, executor=None
    ):
        return [i for i in ["testid1", "testid2"] if i in ids]

    async def dummy_get_documents_by_ids(self, ids, executor=None):
        return [
            Document(
                page_content="Test content",
                metadata={"file_id": i, "user_id": OWNER["user_id"]},
            )
            for i in ids
        ]

    async def dummy_asim(self, embedding, k, filter=None, executor=None):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": (filter or {}).get("file_id", "testid1"),
                "user_id": OWNER["user_id"],
            },
        )
        return [(doc, 0.9)]

    def dummy_sim(self, embedding, k, filter):
        return []

    async def dummy_aadd(self, docs, ids=None, executor=None):
        return ids

    async def dummy_delete(
        self, ids=None, collection_only=False, user_id=None,
        document_origin_type=None, subscription_id=None, executor=None
    ):
        return None

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids)
    monkeypatch.setattr(AsyncPgVector, "asimilarity_search_with_score_by_vector", dummy_asim)
    monkeypatch.setattr(AsyncPgVector, "similarity_search_with_score_by_vector", dummy_sim)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)

    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    from app.config import vector_store
    vector_store.embedding_function = DummyEmbedding()
    yield


# --- /query/{entity_id} -----------------------------------------------------


def test_query_entity_cross_entity_denied():
    h = hdr(ent=["kbA"], act=["read"])
    r = client.post("/query/kbB", json={"query": "q", "k": 2}, headers=h)
    assert r.status_code == 403


def test_query_entity_authorized_ok():
    h = hdr(ent=["kbA"], act=["read"])
    r = client.post("/query/kbA", json={"query": "q", "k": 2}, headers=h)
    assert r.status_code == 200


def test_query_entity_read_action_required():
    # A write-only token cannot query.
    h = hdr(ent=["kbA"], act=["write"])
    r = client.post("/query/kbA", json={"query": "q", "k": 2}, headers=h)
    assert r.status_code == 403


# --- /embed (write) ---------------------------------------------------------


def _file():
    return {"file": ("t.txt", io.BytesIO(b"hello world"), "text/plain")}


def test_embed_cross_entity_denied():
    h = hdr(ent=["userA"], act=["write"])
    r = client.post(
        "/embed", data={"file_id": "f1", "entity_id": "userB"}, files=_file(), headers=h
    )
    assert r.status_code == 403


def test_embed_read_token_denied():
    h = hdr(ent=["userA"], act=["read"])
    r = client.post(
        "/embed", data={"file_id": "f1", "entity_id": "userA"}, files=_file(), headers=h
    )
    assert r.status_code == 403


# --- /text is a READ op (extracts text, writes no vectors) — WP2BR F1 --------


def test_text_accepts_read_token():
    h = hdr(ent=["userA"], act=["read"])
    r = client.post(
        "/text", data={"file_id": "f1", "entity_id": "userA"}, files=_file(), headers=h
    )
    assert r.status_code == 200, r.text
    assert r.json()["text"].strip() == "hello world"


def test_text_rejects_write_only_token():
    h = hdr(ent=["userA"], act=["write"])
    r = client.post(
        "/text", data={"file_id": "f1", "entity_id": "userA"}, files=_file(), headers=h
    )
    assert r.status_code == 403


def test_text_cross_entity_denied():
    h = hdr(ent=["userA"], act=["read"])
    r = client.post(
        "/text", data={"file_id": "f1", "entity_id": "userB"}, files=_file(), headers=h
    )
    assert r.status_code == 403


# --- DELETE /documents (delete) --------------------------------------------


def test_delete_cross_entity_denied():
    h = hdr(ent=["userA"], act=["delete"])
    r = client.request(
        "DELETE", "/documents",
        json={"entity_id": "userB", "file_ids": ["testid1"]}, headers=h,
    )
    assert r.status_code == 403


def test_delete_requires_delete_action():
    h = hdr(ent=["userA"], act=["read", "write"])
    r = client.request(
        "DELETE", "/documents",
        json={"entity_id": "userA", "file_ids": ["testid1"]}, headers=h,
    )
    assert r.status_code == 403


def test_delete_without_entity_denied():
    h = hdr(ent=["userA"], act=["delete"])
    r = client.request(
        "DELETE", "/documents", json={"file_ids": ["testid1"]}, headers=h
    )
    assert r.status_code == 403


def test_delete_authorized_ok():
    h = hdr(ent=["userA"], act=["delete"])
    r = client.request(
        "DELETE", "/documents",
        json={"entity_id": "userA", "file_ids": ["testid1"]}, headers=h,
    )
    assert r.status_code == 200


# --- GET /documents ---------------------------------------------------------


def test_get_documents_unauthorized_owner_not_disclosed():
    # Document exists but is owned by another entity -> 404 (not disclosed).
    OWNER["user_id"] = "otheruser"
    h = hdr(ent=["testuser"], act=["read"])
    r = client.get("/documents", params={"ids": ["testid1"]}, headers=h)
    assert r.status_code == 404


def test_get_documents_authorized_ok():
    OWNER["user_id"] = "testuser"
    h = hdr(ent=["testuser"], act=["read"])
    r = client.get("/documents", params={"ids": ["testid1"]}, headers=h)
    assert r.status_code == 200
    assert r.json()[0]["metadata"]["file_id"] == "testid1"


# --- /documents/{id}/context -----------------------------------------------


def test_context_unauthorized_owner_not_found():
    OWNER["user_id"] = "otheruser"
    h = hdr(ent=["testuser"], act=["read"])
    r = client.get("/documents/testid1/context", headers=h)
    assert r.status_code == 404


# --- /query (by file id) ----------------------------------------------------


def test_query_file_id_filters_unauthorized_docs():
    # Retrieval surfaces a doc owned by another entity -> filtered out, not leaked.
    OWNER["user_id"] = "otheruser"
    h = hdr(ent=["testuser"], act=["read"])
    r = client.post(
        "/query", json={"query": "q", "file_id": "testid1", "k": 2}, headers=h
    )
    assert r.status_code == 200
    assert r.json() == []


def test_query_file_id_entity_filter_must_be_subset():
    h = hdr(ent=["testuser"], act=["read"])
    r = client.post(
        "/query",
        json={"query": "q", "file_id": "testid1", "k": 2, "entity_id": "otheruser"},
        headers=h,
    )
    assert r.status_code == 403


# --- /query_multiple --------------------------------------------------------


def test_query_multiple_filters_unauthorized():
    OWNER["user_id"] = "otheruser"
    h = hdr(ent=["testuser"], act=["read"])
    r = client.post(
        "/query_multiple",
        json={"query": "q", "file_ids": ["testid1", "testid2"], "k": 2},
        headers=h,
    )
    # All candidate docs belong to another entity -> nothing authorized -> 404.
    assert r.status_code == 404


# --- tenant tag on embed (D-KSPT-1) ----------------------------------------


def test_embed_stores_tenant_id_in_metadata(monkeypatch):
    captured = {}

    def fake_prepare(
        data, file_id, user_id, clean_content, document_origin_type=None,
        filename=None, link=None, subscription_id=None, tenant_id=None,
    ):
        captured["tenant_id"] = tenant_id
        captured["user_id"] = user_id
        return [Document(page_content="x", metadata={"file_id": file_id, "user_id": user_id})]

    monkeypatch.setattr(document_routes, "_prepare_documents_sync", fake_prepare)

    h = hdr(ent=["userA"], act=["write"], tid="tenantXYZ")
    r = client.post(
        "/embed", data={"file_id": "f1", "entity_id": "userA"}, files=_file(), headers=h
    )
    assert r.status_code == 200, r.text
    assert captured["tenant_id"] == "tenantXYZ"
    assert captured["user_id"] == "userA"


# --- pgvector debug record routes (reviewer F1) -----------------------------


def test_pgvector_records_gating_and_filter():
    from fastapi import HTTPException
    from app.routes import pgvector_routes as pv

    ent = {"entity_ids": {"userA"}, "tenant_id": "tenantA", "actions": {"read"}}

    class Req:
        def __init__(self, entitlement):
            self.state = type("S", (), {})()
            if entitlement is not None:
                self.state.entitlement = entitlement

    # No entitlement -> 403; missing read action -> 403; read present -> ok.
    with pytest.raises(HTTPException) as e1:
        pv.require_read_entitlement(Req(None))
    assert e1.value.status_code == 403
    with pytest.raises(HTTPException) as e2:
        pv.require_read_entitlement(
            Req({"entity_ids": {"userA"}, "tenant_id": "tenantA", "actions": {"write"}})
        )
    assert e2.value.status_code == 403
    assert pv.require_read_entitlement(Req(ent)) is ent

    # Keep own entity+tenant; drop cross-entity, cross-tenant, and no-user rows;
    # tolerate cmetadata as dict or JSON string.
    rows = [
        {"custom_id": "1", "cmetadata": {"user_id": "userA", "tenant_id": "tenantA"}},
        {"custom_id": "2", "cmetadata": {"user_id": "userB", "tenant_id": "tenantA"}},
        {"custom_id": "3", "cmetadata": {"user_id": "userA", "tenant_id": "tenantB"}},
        {"custom_id": "4", "cmetadata": '{"user_id": "userA"}'},  # JSON str, no tenant
        {"custom_id": "5", "cmetadata": {}},  # no user_id
    ]
    kept = {r["custom_id"] for r in pv.filter_rows_by_entitlement(rows, ent)}
    assert kept == {"1", "4"}
