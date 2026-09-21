"""Tenant/entity isolation guard — F-TENANT-GUARD (Candidate B).

Ported from the superseded RATB-01 commit (preserve/ratb-01-tenant-boundary-5d9fe48,
tests/test_tenant_boundary.py) and rewritten against main's CURRENT authority model.

The old model PROPOSED a `tenant_id` claim and marked every cross-tenant case
xfail(strict) because rag_api could not enforce it yet. main was rebuilt by
KSPT-01 on a different model: authority is the signed token's entitlement —
`tid` (tenant), `ent` (entity id set), `act` (action set) — verified in
app.middleware, and caller-supplied ids are only FILTERS re-checked against the
entitlement (app.routes.document_routes `_require_action` / `_require_entity`).

The old commit's three cross-tenant cases (query / read / delete) are ALREADY
covered under the new model by tests/test_entitlement_routes.py:
    - cross-entity query   -> test_query_entity_cross_entity_denied
    - cross-entity read     -> test_get_documents_unauthorized_owner_not_disclosed
    - cross-entity delete   -> test_delete_cross_entity_denied
so they are NOT re-ported here (that would only duplicate existing coverage).

What was GENUINELY missing is the modern equivalent of the old commit's
middleware guard "a validly-signed token whose `id` is empty/whitespace must not
be treated as an authenticated principal" (old app/middleware.py returned 401).
main dropped that middleware check because `id` is no longer an authority claim;
it survives only as the fallback OWNER in `document_routes.get_user_id` when a
route is called WITHOUT an explicit entity_id:

    return entity_id if entity_id else request.state.user.get("id")

NO existing test reaches that `else` branch — every entitlement test passes an
explicit entity_id. These tests exercise exactly that fallback with an empty /
whitespace `id` and prove main FAILS CLOSED: the resolved empty owner is not in
the token's `ent` set, so `_require_entity` refuses with 403. The empty `id`
never becomes an owner or a "public" identity for a read or a write.

MEASURED (2026-09-21, files01-ocr-test:wip, HEAD 36d4fb6):
    /embed  , id="" , ent=["userA"], no entity_id -> 403
    /embed  , id="  ", ent=["userA"], no entity_id -> 403
    /text   , id="" , ent=["userA"], no entity_id -> 403
    /text   , id="userA", ent=["userA"], no entity_id -> 200  (positive control)

PROOF that these are real controls, not vacuous: dropping the `_require_entity`
entity-membership check (the entitlement guard) turns every 403 below into a 200
(store proceeds under the empty owner). Verified red under that mutation.
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


def hdr(ent, act, tid="tenantA", uid="testuser"):
    """A validly-signed token. `uid` is the `id` claim; the isolation cases set
    it empty / whitespace to prove it never becomes an owner."""
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

    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3]
    )

    async def dummy_aadd(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd)

    # Keep the write path off the real embedding/DB machinery. This dummy stores a
    # row under whatever user_id the route resolved — so if the guard were removed
    # an empty-owner write would visibly SUCCEED (200) rather than error for an
    # unrelated reason. That is what makes the 403 assertions real controls.
    def fake_prepare(
        data, file_id, user_id, clean_content, document_origin_type=None,
        filename=None, link=None, subscription_id=None, tenant_id=None,
    ):
        return [
            Document(page_content="x", metadata={"file_id": file_id, "user_id": user_id})
        ]

    monkeypatch.setattr(document_routes, "_prepare_documents_sync", fake_prepare)
    yield


def _file():
    return {"file": ("t.txt", io.BytesIO(b"hello world"), "text/plain")}


# --- Positive control: the get_user_id `id` fallback path is real -----------
# Proves the empty-id 403s below are about the empty identity, not a broken
# route or fixture: an AUTHORIZED id resolved through the same fallback reaches
# success.


def test_authorized_id_fallback_reaches_success():
    """No entity_id supplied, so the owner is resolved from `id`. When that id is
    within the entitlement, the read succeeds (200) through the fallback path."""
    h = hdr(ent=["userA"], act=["read"], uid="userA")
    r = client.post("/text", data={"file_id": "f1"}, files=_file(), headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["text"].strip() == "hello world"


# --- Negative space: an empty / whitespace `id` never gains owner authority ---
# These reach document_routes.get_user_id's `request.state.user.get("id")`
# branch (no explicit entity_id), which no other test exercises.


def test_embed_empty_id_no_entity_denied():
    """Write with an empty `id` and no entity_id: the empty owner is not in the
    token entitlement, so the write is refused (403). An empty id must never
    become an authorized write owner."""
    h = hdr(ent=["userA"], act=["write"], uid="")
    r = client.post("/embed", data={"file_id": "f1"}, files=_file(), headers=h)
    assert r.status_code == 403, r.text


def test_embed_whitespace_id_no_entity_denied():
    """A whitespace-only `id` is likewise not a usable owner: refused (403)."""
    h = hdr(ent=["userA"], act=["write"], uid="   ")
    r = client.post("/embed", data={"file_id": "f1"}, files=_file(), headers=h)
    assert r.status_code == 403, r.text


def test_text_empty_id_no_entity_denied():
    """Read (/text) with an empty `id` and no entity_id: refused (403). The empty
    id never becomes a readable owner or a 'public' identity."""
    h = hdr(ent=["userA"], act=["read"], uid="")
    r = client.post("/text", data={"file_id": "f1"}, files=_file(), headers=h)
    assert r.status_code == 403, r.text
