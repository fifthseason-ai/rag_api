"""RATB-01 Increment B gap tests — tenant isolation (NOT yet enforceable).

These encode the *intended* cross-tenant boundary described in the Demian
reconciliation delta. rag_api rows carry no tenant column and Core mints no
tenant claim today, so the boundary CANNOT be enforced yet. Each cross-tenant
case is therefore marked ``xfail(strict=True)``: it asserts the secure outcome
(refuse + leak nothing), which currently fails — proving the gap is real. The
strict marker turns any silent pass (XPASS) into a red run, so the day a real
tenant claim + row ownership lands, these flip to XPASS and force the markers
to be removed.

A same-tenant positive (NOT xfail) proves the synthetic fixture actually wires
documents through the endpoints, so the xfails are real gaps and not fixture
breakage.

Proposed JWT claim contract (section 5 of the delta): the RAG token gains a
``tenant_id`` claim minted by Core; rag_api persists ``cmetadata.tenant_id`` at
embed and filters query/read/delete by the verified claim.
"""

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


# Synthetic two-tenant store. `user_id` is what rag_api actually stores today
# (a knowledge-base id); `tenant_id` is the PROPOSED ownership dimension that is
# NOT persisted or enforced yet.
_STORE = {
    "kb-tenant-a": {
        "file_id": "file-a",
        "user_id": "kb-tenant-a",
        "tenant_id": "tenant-a",
        "content": "tenant A secret",
    },
    "kb-tenant-b": {
        "file_id": "file-b",
        "user_id": "kb-tenant-b",
        "tenant_id": "tenant-b",
        "content": "tenant B secret",
    },
}
_BY_FILE_ID = {entry["file_id"]: entry for entry in _STORE.values()}


def _doc(entry):
    return Document(
        page_content=entry["content"],
        metadata={
            "file_id": entry["file_id"],
            "user_id": entry["user_id"],
            "tenant_id": entry["tenant_id"],
        },
    )


def _token(tenant_id, user_id="user-a"):
    """A validly-signed RAG token carrying the PROPOSED tenant_id claim."""
    secret = os.environ["JWT_SECRET"]
    payload = {
        "id": user_id,
        "tenant_id": tenant_id,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, secret, algorithm='HS256')}"}


def _requested_user_id(filter_):
    if not filter_:
        return None
    value = filter_.get("user_id")
    if isinstance(value, dict):
        return value.get("$eq") or value.get("$in")
    return value


@pytest.fixture(autouse=True)
def tenant_store(monkeypatch):
    """Wire a tenant-aware dummy store and disable hybrid/rerank (which need a DB
    and Bedrock). The store deliberately enforces NO tenant boundary — it mirrors
    rag_api's real behavior: query filters by the caller-named user_id/entity_id,
    while read (GET /documents) has no owner filter at all."""
    deleted = []

    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="tenant-test"
        )

    monkeypatch.setattr(document_routes, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(document_routes, "RERANK_ENABLED", False)
    monkeypatch.setattr(
        document_routes,
        "get_cached_query_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    async def dummy_asimilarity(self, embedding, k, filter=None, executor=None):
        requested = _requested_user_id(filter)
        entry = _STORE.get(requested)
        return [(_doc(entry), 0.9)] if entry else []

    async def dummy_get_filtered_ids(
        self, ids, user_id=None, document_origin_type=None,
        subscription_id=None, executor=None,
    ):
        present = [fid for fid in (ids or []) if fid in _BY_FILE_ID]
        if user_id is not None:
            present = [fid for fid in present if _BY_FILE_ID[fid]["user_id"] == user_id]
        return present

    async def dummy_get_documents_by_ids(self, ids, executor=None):
        # No owner filter — this is exactly why cross-tenant read leaks today.
        return [_doc(_BY_FILE_ID[fid]) for fid in (ids or []) if fid in _BY_FILE_ID]

    async def dummy_delete(
        self, ids=None, collection_only=False, user_id=None,
        document_origin_type=None, subscription_id=None, executor=None,
    ):
        for fid in ids or []:
            entry = _BY_FILE_ID.get(fid)
            if entry is None:
                continue
            if user_id is not None and entry["user_id"] != user_id:
                continue
            deleted.append(fid)
        return None

    monkeypatch.setattr(AsyncPgVector, "asimilarity_search_with_score_by_vector", dummy_asimilarity)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)

    yield {"deleted": deleted}


# --- Same-tenant positive: proves the fixture wires documents through. ---


def test_same_tenant_query_returns_rows():
    resp = client.post(
        "/query/kb-tenant-a",
        json={"query": "what is the secret", "k": 4},
        headers=_token("tenant-a"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body, "same-tenant query must return its own rows"
    assert body[0][0]["page_content"] == "tenant A secret"


# --- Cross-tenant cases: intended-secure assertions that currently fail. ---


@pytest.mark.xfail(
    strict=True,
    reason="RATB-01: rag_api rows carry no tenant and Core mints no tenant claim; "
    "enforcement lands with the reconciliation delta",
)
def test_cross_tenant_query_denied():
    """A tenant-a principal querying tenant-b's knowledge base must get nothing."""
    resp = client.post(
        "/query/kb-tenant-b",
        json={"query": "what is the secret", "k": 4},
        headers=_token("tenant-a"),
    )
    leaked = resp.status_code == 200 and resp.json()
    assert resp.status_code in (403, 404) or not leaked, (
        "cross-tenant query leaked tenant-b rows to a tenant-a principal"
    )


@pytest.mark.xfail(
    strict=True,
    reason="RATB-01: rag_api rows carry no tenant and Core mints no tenant claim; "
    "enforcement lands with the reconciliation delta",
)
def test_cross_tenant_read_denied():
    """A tenant-a principal reading tenant-b's document by id must be refused."""
    resp = client.get(
        "/documents",
        params={"ids": ["file-b"]},
        headers=_token("tenant-a"),
    )
    leaked = resp.status_code == 200 and resp.json()
    assert resp.status_code in (403, 404) or not leaked, (
        "cross-tenant read returned tenant-b document to a tenant-a principal"
    )


@pytest.mark.xfail(
    strict=True,
    reason="RATB-01: rag_api rows carry no tenant and Core mints no tenant claim; "
    "enforcement lands with the reconciliation delta",
)
def test_cross_tenant_delete_denied(tenant_store):
    """A tenant-a principal deleting tenant-b's document must be refused and
    delete zero rows."""
    resp = client.request(
        "DELETE",
        "/documents",
        json={"file_ids": ["file-b"], "entity_id": "kb-tenant-b"},
        headers=_token("tenant-a"),
    )
    refused = resp.status_code in (403, 404)
    assert refused and tenant_store["deleted"] == [], (
        "cross-tenant delete removed tenant-b rows on behalf of a tenant-a principal"
    )
