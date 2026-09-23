"""Read/delete routes never hand a caller the raw exception text (FILES-01 disclosure).

The intake surfaces and the auth middleware already replaced `detail=str(e)` with a generic
message plus a logged reference (#26/#34/#35). Six read/delete routes still leaked -- GET and
DELETE /documents, /query, /query/{entity_id}, /documents/{id}/context, /query_multiple. This
pins that an unexpected error on each returns a caller-safe message with a reference, while the
real text (a secret marker standing in for store internals or another tenant's data) stays in
the log only.
"""

import datetime
import os

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.routes import document_routes as dr
from app.services.vector_store.async_pg_vector import AsyncPgVector

SECRET = "SECRET-INTERNAL-9f3c2a-do-not-disclose"
_JWT = "testsecret"


def _hdr(ent=("kbA",), act=("read", "write", "delete"), tid="tA", uid="u"):
    os.environ["JWT_SECRET"] = _JWT
    return {"Authorization": "Bearer " + jwt.encode(
        {"id": uid, "tid": tid, "ent": list(ent), "act": list(act),
         "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
        _JWT, algorithm="HS256")}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    os.environ["JWT_SECRET"] = _JWT
    from concurrent.futures import ThreadPoolExecutor
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(dr, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    yield


client = TestClient(app)


def _boom(*a, **k):
    raise RuntimeError(SECRET)


async def _aboom(*a, **k):
    raise RuntimeError(SECRET)


def _assert_safe(resp, expected_status):
    assert resp.status_code == expected_status, resp.text
    body = resp.json()
    detail = body.get("detail", body)
    text = detail if isinstance(detail, str) else str(detail)
    assert SECRET not in text, f"raw exception text leaked to caller: {text!r}"
    assert "reference" in text.lower(), f"no operator reference offered: {text!r}"


def test_get_documents_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", _aboom)
    r = client.get("/documents", params={"ids": ["x"]}, headers=_hdr())
    _assert_safe(r, 500)


def test_delete_documents_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", _aboom)
    r = client.request("DELETE", "/documents",
                       json={"entity_id": "kbA", "file_ids": ["x"]}, headers=_hdr())
    _assert_safe(r, 500)


def test_query_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(dr, "get_cached_query_embedding", _boom)
    r = client.post("/query", json={"query": "q", "file_id": "x", "k": 1, "entity_id": "kbA"},
                    headers=_hdr())
    _assert_safe(r, 500)


def test_query_entity_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(dr, "get_cached_query_embedding", _boom)
    r = client.post("/query/kbA", json={"query": "q", "k": 1}, headers=_hdr())
    _assert_safe(r, 500)


def test_query_multiple_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(dr, "get_cached_query_embedding", _boom)
    r = client.post("/query_multiple", json={"query": "q", "file_ids": ["x"], "k": 1},
                    headers=_hdr())
    _assert_safe(r, 500)


def test_document_context_internal_error_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", _aboom)
    r = client.get("/documents/x/context", headers=_hdr())
    # This route already returned 400 for an unexpected error; the disclosure fix keeps
    # that status and removes only the raw text.
    _assert_safe(r, 400)


def test_the_reference_reaches_the_log_not_the_caller(monkeypatch, caplog):
    """The reference is only useful if the operator can find the real error by it."""
    import logging
    monkeypatch.setattr(dr, "get_cached_query_embedding", _boom)
    with caplog.at_level(logging.ERROR):
        r = client.post("/query", json={"query": "q", "file_id": "x", "k": 1, "entity_id": "kbA"},
                        headers=_hdr())
    import re
    detail = r.json()["detail"]
    m = re.search(r"reference ([0-9a-f]{12})", detail)
    assert m, f"no 12-hex reference in the caller message: {detail!r}"
    ref = m.group(1)
    logged = "\n".join(rec.getMessage() for rec in caplog.records)
    assert ref in logged, "the reference on the wire is not findable in the log"
    assert SECRET in logged, "the real exception text must be in the log for the operator"
