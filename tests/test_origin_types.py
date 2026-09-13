"""DocumentOriginType parity with Core (KI-02 WP-C2).

Core `packages/data-provider/src/tempo.ts` declares
`DocumentOriginType { ORGANIC, SHAREPOINT, ONEDRIVE, GDRIVE, GMAIL }` and tags
embeds/deletes per provider (`fileSyncListener.js` origin tag; OneDrive/GDrive/
Gmail sync paths). rag_api previously declared only `{ORGANIC, SHAREPOINT}`, so
Pydantic rejected ONEDRIVE/GDRIVE/GMAIL on `/embed` and `DELETE /documents` and
the per-origin vector cleanup those providers rely on could not run. WP-C2 adds
ONEDRIVE, GDRIVE, GMAIL and BOX.

BOX is the forward-add for the KI-02 inc-B Box adapter (which is currently forced
to emit ORGANIC because rag_api lacks a BOX bucket); Core will switch Box to a
BOX origin only after this rag_api change is DEPLOYED.

The vector store is SIMULATED here: AsyncPgVector's DB methods and
`store_data_in_vector_db` are monkeypatched, so no real pgvector connection is
made. These tests prove the request models/routes ACCEPT each new origin and
thread its value through, that an unknown origin is still rejected with today's
status, that origin-scoped deletion removes only that origin's rows, and pin the
enum against future drift. They do NOT exercise the real SQL metadata filter
(that needs a live DB); that filter is pre-existing and unchanged by WP-C2.
"""
import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes import document_routes
from app.models import DocumentOriginType
from app.services.vector_store.async_pg_vector import AsyncPgVector

client = TestClient(app)

SECRET = "testsecret"

# Core's canonical DocumentOriginType set (tempo.ts) plus the BOX forward-add
# (KI-02 inc-B). name == value for every member (string-value parity with Core).
CORE_ORIGINS = {"ORGANIC", "SHAREPOINT", "ONEDRIVE", "GDRIVE", "GMAIL"}
NEW_ORIGINS = ["ONEDRIVE", "GDRIVE", "GMAIL", "BOX"]


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


def _file():
    return {"file": ("t.txt", io.BytesIO(b"hello world"), "text/plain")}


@pytest.fixture(autouse=True)
def _simulated_store(monkeypatch):
    """SIMULATED store: no real pgvector. Sensible defaults; individual tests
    override get_filtered_ids/delete when they need origin-scoped behavior."""
    os.environ["JWT_SECRET"] = SECRET

    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    async def dummy_get_filtered_ids(
        self, ids, user_id=None, document_origin_type=None,
        subscription_id=None, executor=None
    ):
        # Echo requested ids as "existing" so the route's 404 guard passes.
        return list(ids)

    async def dummy_delete(
        self, ids=None, collection_only=False, user_id=None,
        document_origin_type=None, subscription_id=None, executor=None
    ):
        return None

    async def dummy_aadd(self, docs, ids=None, executor=None):
        return ids

    async def dummy_delete_summaries(document_ids, user_id=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd)
    monkeypatch.setattr(
        document_routes, "delete_summaries_by_file_ids", dummy_delete_summaries
    )

    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    from app.config import vector_store
    vector_store.embedding_function = DummyEmbedding()
    yield


# --- /embed accepts each new origin (store SIMULATED) -----------------------


@pytest.mark.parametrize("origin", NEW_ORIGINS)
def test_embed_accepts_new_origin(monkeypatch, origin):
    """Each new origin is accepted by the /embed Form model and its value is
    threaded into storage. Pre-WP-C2 (enum = {ORGANIC, SHAREPOINT}) this 422s at
    Form validation -> RED."""
    captured = {}

    async def fake_store(*args, **kwargs):
        captured["document_origin_type"] = kwargs.get("document_origin_type")
        return {"message": "ok", "ids": ["f1"], "docs": [Document(page_content="x", metadata={})]}

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", fake_store)

    h = hdr(ent=["userA"], act=["write"])
    r = client.post(
        "/embed",
        data={"file_id": "f1", "entity_id": "userA", "document_origin_type": origin},
        files=_file(),
        headers=h,
    )
    assert r.status_code == 200, r.text
    # Route passed the caller's origin string (not the ORGANIC default) downstream.
    assert captured["document_origin_type"] == origin


# --- DELETE /documents accepts each new origin (store SIMULATED) ------------


@pytest.mark.parametrize("origin", NEW_ORIGINS)
def test_delete_accepts_new_origin(origin):
    """DeleteDocumentsBody.document_origin_type accepts each new origin.
    Pre-WP-C2 this 422s at body validation -> RED."""
    h = hdr(ent=["testuser"], act=["delete"])
    r = client.request(
        "DELETE",
        "/documents",
        json={
            "entity_id": "testuser",
            "file_ids": ["testid1"],
            "document_origin_type": origin,
        },
        headers=h,
    )
    assert r.status_code == 200, r.text


# --- unknown origin still rejected with today's status (pin) ----------------


def test_embed_rejects_unknown_origin():
    """An origin not in the enum (DROPBOX) is still rejected at /embed with the
    same status as today (422). WP-C2 must not widen the enum to arbitrary
    strings."""
    h = hdr(ent=["userA"], act=["write"])
    r = client.post(
        "/embed",
        data={"file_id": "f1", "entity_id": "userA", "document_origin_type": "DROPBOX"},
        files=_file(),
        headers=h,
    )
    assert r.status_code == 422, r.text


def test_delete_rejects_unknown_origin():
    """An origin not in the enum (DROPBOX) is still rejected on DELETE with the
    same status as today (422)."""
    h = hdr(ent=["testuser"], act=["delete"])
    r = client.request(
        "DELETE",
        "/documents",
        json={
            "entity_id": "testuser",
            "file_ids": ["testid1"],
            "document_origin_type": "DROPBOX",
        },
        headers=h,
    )
    assert r.status_code == 422, r.text


# --- origin-scoped deletion removes ONLY that origin's rows (SIMULATED) ------


def test_delete_is_origin_scoped_only_that_origin_removed(monkeypatch):
    """A single file_id embedded under two origins: deleting one origin must
    remove only that origin's row and leave the other intact. SIMULATED store
    honors document_origin_type exactly as the real cmetadata filter would, so a
    route that dropped the origin filter (deleting both rows) is caught."""
    rows = [
        {"custom_id": "shared", "document_origin_type": "ONEDRIVE"},
        {"custom_id": "shared", "document_origin_type": "GDRIVE"},
    ]
    seen = {}

    def _matches(row, ids, origin):
        if ids and row["custom_id"] not in ids:
            return False
        if origin is not None and row["document_origin_type"] != origin:
            return False
        return True

    async def fake_get_filtered_ids(
        self, ids, user_id=None, document_origin_type=None,
        subscription_id=None, executor=None
    ):
        return sorted(
            {r["custom_id"] for r in rows if _matches(r, ids, document_origin_type)}
        )

    async def fake_delete(
        self, ids=None, collection_only=False, user_id=None,
        document_origin_type=None, subscription_id=None, executor=None
    ):
        seen["origin"] = document_origin_type
        rows[:] = [r for r in rows if not _matches(r, ids, document_origin_type)]
        return None

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", fake_get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "delete", fake_delete)

    h = hdr(ent=["testuser"], act=["delete"])
    r = client.request(
        "DELETE",
        "/documents",
        json={
            "entity_id": "testuser",
            "file_ids": ["shared"],
            "document_origin_type": "ONEDRIVE",
        },
        headers=h,
    )
    assert r.status_code == 200, r.text
    # Route threaded the requested origin into the store delete call...
    assert seen["origin"] == "ONEDRIVE"
    # ...and only the ONEDRIVE row was removed; the GDRIVE row survives.
    assert [r["document_origin_type"] for r in rows] == ["GDRIVE"]


# --- regression pin: enum == Core set + BOX ---------------------------------


def test_enum_is_core_set_plus_box():
    """Pin the membership so future drift from Core (tempo.ts) is caught. Core =
    {ORGANIC, SHAREPOINT, ONEDRIVE, GDRIVE, GMAIL}; rag_api adds BOX. name ==
    value for every member (string-value parity with Core)."""
    names = {m.name for m in DocumentOriginType}
    values = {m.value for m in DocumentOriginType}
    expected = CORE_ORIGINS | {"BOX"}
    assert names == expected
    assert values == expected
    assert all(m.name == m.value for m in DocumentOriginType)
    # Pre-existing members are unchanged.
    assert DocumentOriginType.ORGANIC.value == "ORGANIC"
    assert DocumentOriginType.SHAREPOINT.value == "SHAREPOINT"
