"""RV-197 F1 SECURITY: R2 redaction must cover EVERY exit that serves metadata, and the
redactor must fail CLOSED.

RV-183 (post-merge review of #114) found that `_redacted_metadata` is applied on /query,
/query/{entity_id} and /query_multiple (the `_on_the_wire` seam) but NOT on GET /documents
(`get_documents_by_ids`), which returns the stored `Document.metadata` verbatim. A LEGACY row
whose `quarantined_link` is still the raw string is served in full over GET /documents,
including credentials, path and token -- the exact leak the marker exists to prevent. The
docstrings say the raw "never leaves the process"; that was true only for the query routes.

Sub-defect (fail-OPEN redactor): `_redacted_metadata` used to redact only when
`quarantined_link` was a `str`. A non-string, non-canonical marker (e.g. `{"url": raw}` or
`[raw]`) passed through UNCHANGED, on every route. The fix redacts anything that is not
EXACTLY the canonical redacted object `{scheme, host, refusal_reason, sha256}` (fail-closed).

RED-FIRST at aa66a3ab9 (the merged tree, before the F1 fix):
  * GET /documents serves the legacy raw string verbatim -> the 'no raw substring' assertions RED.
  * the fail-open shapes ride the wire verbatim on BOTH GET /documents and /query -> RED.
"""
import datetime
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "test-secret-rv197-getdocs"

ENT_A, ENT_B = "ent-alpha", "ent-beta"
TENANT_A, TENANT_B = "tenant-alpha", "tenant-beta"
FILE_A = "file-alpha-legacy-raw"

#: A raw refused URL carrying exactly the things a quarantine marker must NEVER leak.
RAW_URL = "http://user:s3cr3tPW@raw-bucket.example/secret-path/file.pdf?token=tokXYZ123"
RAW_SHA256 = hashlib.sha256(RAW_URL.encode("utf-8")).hexdigest()
#: Fragments that must appear NOWHERE in a served response (the host is kept, so it is excluded).
RAW_LEAK_MARKERS = ("s3cr3tPW", "secret-path", "file.pdf", "token=tokXYZ123", RAW_URL)

CANONICAL_KEYS = {"scheme", "host", "refusal_reason", "sha256"}

#: LEGACY at rest: an earlier backfill moved the raw string verbatim.
LEGACY_RAW = {
    "file_id": FILE_A,
    "user_id": ENT_A,
    "tenant_id": TENANT_A,
    "filename": "brief.pdf",
    "quarantined_link": RAW_URL,
    "quarantined_reason": "link_scheme_not_allowed",
    "quarantined_at": "2026-09-24T20:00:00Z",
}

#: Fail-OPEN shapes: non-string, non-canonical `quarantined_link` values that must NOT pass through.
FAILOPEN_DICT = {
    "file_id": FILE_A,
    "user_id": ENT_A,
    "tenant_id": TENANT_A,
    "quarantined_link": {"url": RAW_URL},
}
FAILOPEN_LIST = {
    "file_id": FILE_A,
    "user_id": ENT_A,
    "tenant_id": TENANT_A,
    "quarantined_link": [RAW_URL],
}


def _tok(entity_ids, tenant_id=TENANT_A, act=("read",)):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": "caller",
        "tid": tenant_id,
        "ent": list(entity_ids),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


class RowStore(AsyncPgVector):
    """Holds one row and hands it over unfiltered, so the ROUTE's own filter is exercised."""

    def __init__(self, metadata):
        self._bind = None
        self.rows = {metadata["file_id"]: Document(page_content="the passage", metadata=dict(metadata))}

    async def get_documents_by_ids(self, ids, executor=None):
        return [self.rows[i] for i in ids if i in self.rows]


@pytest.fixture()
def env(monkeypatch):
    from app.routes import document_routes as dr
    from main import app

    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    def _make(metadata):
        store = RowStore(metadata)
        monkeypatch.setattr(dr, "vector_store", store)
        monkeypatch.setattr("app.config.vector_store", store, raising=False)
        client = TestClient(app)
        return client

    return _make


# ---------------------------------------------------------------------------
# F1(a): GET /documents redacts a legacy raw quarantined_link.
# ---------------------------------------------------------------------------


def test_get_documents_never_serves_a_legacy_raw_quarantined_link(env):
    """RED-FIRST at aa66a3ab9. A same-tenant owner reads its own legacy row over GET /documents;
    the raw refused URL must NOT appear anywhere in the response, and quarantined_link must be
    the redacted object. On the merged tree GET /documents applied no redaction, so the raw rode
    the wire verbatim (status 200 with s3cr3tPW/secret-path/token in the body)."""
    client = env(LEGACY_RAW)
    r = client.get("/documents", params={"ids": [FILE_A]}, headers=_tok([ENT_A]))
    assert r.status_code == 200, r.text

    for marker in RAW_LEAK_MARKERS:
        assert marker not in r.text, (
            "GET /documents leaked the raw quarantined URL fragment %r: %s" % (marker, r.text))

    ql = r.json()[0]["metadata"]["quarantined_link"]
    assert isinstance(ql, dict), "quarantined_link must be the redacted object, not a string: %r" % ql
    assert set(ql) == CANONICAL_KEYS, ql
    assert ql["host"] == "raw-bucket.example" and ql["scheme"] == "http", ql
    assert ql["sha256"] == RAW_SHA256, ql


def test_get_documents_serves_a_same_tenant_row_redacted_not_refused(env):
    """The redaction is applied AFTER the entitlement filter: the owning entity still gets a 200
    with its row, only with the raw stripped. Redaction must not turn a legitimate read into a 404."""
    client = env(LEGACY_RAW)
    r = client.get("/documents", params={"ids": [FILE_A]}, headers=_tok([ENT_A]))
    assert r.status_code == 200, r.text
    assert r.json()[0]["metadata"]["file_id"] == FILE_A


def test_get_documents_cross_tenant_stays_404_with_no_leak(env):
    """A caller whose entitlement does not include the row's owner gets 404 and no raw leak --
    the entitlement filter still fires, and redaction does not weaken it."""
    client = env(LEGACY_RAW)
    r = client.get("/documents", params={"ids": [FILE_A]}, headers=_tok([ENT_B]))
    assert r.status_code == 404, r.text
    for marker in RAW_LEAK_MARKERS:
        assert marker not in r.text, r.text


# ---------------------------------------------------------------------------
# F1(b): the redactor fails CLOSED on non-string, non-canonical marker shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [FAILOPEN_DICT, FAILOPEN_LIST], ids=["dict_with_url", "list"])
def test_get_documents_fails_closed_on_a_non_canonical_marker(env, shape):
    """RED-FIRST. A `{"url": raw}` or `[raw]` quarantined_link is neither a canonical marker nor a
    string; the merged redactor let it through unchanged. It must be replaced by the redacted
    object, so the embedded raw URL never reaches the wire."""
    client = env(shape)
    r = client.get("/documents", params={"ids": [FILE_A]}, headers=_tok([ENT_A]))
    assert r.status_code == 200, r.text
    for marker in RAW_LEAK_MARKERS:
        assert marker not in r.text, (
            "GET /documents leaked a raw fragment %r through a non-canonical marker: %s" % (marker, r.text))
    ql = r.json()[0]["metadata"]["quarantined_link"]
    assert isinstance(ql, dict) and set(ql) == CANONICAL_KEYS, ql


def test_redacted_metadata_unit_fails_closed_but_passes_the_canonical_object():
    """Unit-level fail-closed contract. Only the EXACT canonical object passes through untouched
    (no churn -> the retrieved==stored equality tests hold); every other shape is redacted."""
    from app.routes import document_routes as dr

    canonical = {"scheme": "http", "host": "h.example", "refusal_reason": "link_scheme_not_allowed",
                 "sha256": "0" * 64}
    passed = dr._redacted_metadata({"quarantined_link": dict(canonical), "k": 1})
    assert passed["quarantined_link"] == canonical, "the canonical object at rest must pass unchanged"

    for bad in (RAW_URL, {"url": RAW_URL}, [RAW_URL], {"scheme": "x"}, 123):
        out = dr._redacted_metadata({"quarantined_link": bad})
        marker = out["quarantined_link"]
        assert isinstance(marker, dict) and set(marker) == CANONICAL_KEYS, (bad, marker)
        assert RAW_URL not in str(marker), (bad, marker)

    # nothing to redact -> the SAME dict is returned (no churn).
    same = {"file_id": "x"}
    assert dr._redacted_metadata(same) is same


# ---------------------------------------------------------------------------
# F1(b) also hardens /query*: a non-canonical marker must be redacted there too.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [FAILOPEN_DICT, FAILOPEN_LIST], ids=["dict_with_url", "list"])
def test_query_fails_closed_on_a_non_canonical_marker(monkeypatch, shape):
    """RED-FIRST. The same fail-open shapes leaked on /query too (RV-183 PASSTHRU probe). After the
    fix the query seam redacts them, so no raw fragment reaches the wire."""
    from app.routes import document_routes as dr
    from main import app

    async def fake_retrieve(*_a, **_k):
        return [(Document(page_content="p", metadata=dict(shape)), 0.25)]

    monkeypatch.setattr(dr, "_retrieve_documents", fake_retrieve)
    monkeypatch.setattr(dr, "get_cached_query_embedding", lambda q: [0.1, 0.2, 0.3])
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    client = TestClient(app)
    r = client.post("/query", json={"query": "q", "file_id": FILE_A, "k": 1, "entity_id": ENT_A},
                    headers=_tok([ENT_A]))
    assert r.status_code == 200, r.text
    for marker in RAW_LEAK_MARKERS:
        assert marker not in r.text, r.text
    ql = r.json()[0][0]["metadata"]["quarantined_link"]
    assert isinstance(ql, dict) and set(ql) == CANONICAL_KEYS, ql
