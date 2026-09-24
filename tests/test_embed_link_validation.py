"""A caller-supplied `link` must be an https URL, or /embed refuses it in a way the caller sees.

F-EMBED-LINK-VALIDATE (producer half of CORE-CITATION-GOVERNED-LINK).

WHY. `link` arrives on POST /embed as a form field and was stamped VERBATIM into chunk
cmetadata (document_routes.py:1828), where Core opens it from a citation -- historically with
`window.open` and no governed-URL check. The producer promised nothing about that value, so an
entitled caller could put `javascript:`, `data:`, `http:` or any host in front of a reader's
browser. Core keeps its own consumer gate permanently, because a write-time rule cannot clean
rows already stored; this closes the producer side for everything written from now on.

SAFE BY MEASUREMENT (STEP 1, 2026-09-23, recorded in
FILES-DEV/F-EMBED-LINK-VALIDATE-STEP1-20260923.md): Core's ONLY sender is the connector sync
loop (`fileSyncListener.js:517`), which always sends the source system's own https item URL
(SharePoint/OneDrive Graph `webUrl`, Box `shared_link.url`); organic uploads send no link at
all. So https-only refuses nothing legitimate today. That measurement is why this ships as a
refusal rather than a warning.

SCOPE. Scheme only. The HOST allowlist is deferred behind a deployment config
(`RAG_GOVERNED_LINK_HOSTS`) whose value is the operator's -- the connector hosts it must admit
are per-tenant for Box, and a half-guessed host rule would be worse than none.

Controls:
  * delete the `_reject_ungoverned_link(link, file_id)` call in the route -> every refusal
    test reds (the check is what refuses, not something incidental).
  * widen `_ALLOWED_LINK_SCHEMES` to include "http" -> the http refusal case reds, plus the
    one test that pins the advertised list. MEASURED: an earlier draft asserted the exact
    allowed-scheme list inside the parametrised test, which made ALL five cases red on any
    widening -- so the control did not isolate what this line claimed. The list is now pinned
    once, and the other cases judge their own scheme.
  * make the guard fire on an absent link -> the organic-upload test reds, which is the
    regression that would break every user upload.
"""
import datetime
import io
import os

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app

SECRET = "test-secret-embed-link"

client = TestClient(app)


@pytest.fixture(autouse=True)
def _store_double(monkeypatch):
    """The minimum that lets /embed reach 200 without a database: a thread pool and an
    in-memory write. Mirrors tests/test_entitlement_routes.py's fixture -- the ACCEPT cases
    below are only meaningful if the route can actually complete."""
    from concurrent.futures import ThreadPoolExecutor

    from app.services.vector_store.async_pg_vector import AsyncPgVector

    os.environ["JWT_SECRET"] = SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    async def _aadd(self, docs, ids=None, executor=None):
        return ids

    async def _filtered(self, ids, user_id=None, document_origin_type=None,
                        subscription_id=None, executor=None):
        return []

    async def _delete(self, ids=None, collection_only=False, user_id=None,
                      document_origin_type=None, subscription_id=None, executor=None, **_):
        return None

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", _aadd)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", _filtered)
    monkeypatch.setattr(AsyncPgVector, "delete", _delete)
    yield


def hdr(ent=("userA",), act=("write",), tid="tenantA", uid="userA"):
    os.environ["JWT_SECRET"] = SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": list(ent),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, SECRET, algorithm='HS256')}"}


def _file():
    return {"file": ("t.txt", io.BytesIO(b"hello world"), "text/plain")}


def _embed(link=None, file_id="f-link"):
    data = {"file_id": file_id, "entity_id": "userA"}
    if link is not None:
        data["link"] = link
    return client.post("/embed", data=data, files=_file(), headers=hdr())


@pytest.mark.parametrize(
    "bad_link, expected_scheme",
    [
        ("javascript:alert(1)", "javascript"),
        ("data:text/html;base64,PHNjcmlwdD4=", "data"),
        ("http://contoso.sharepoint.com/x/Deck.pptx", "http"),
        ("//contoso.sharepoint.com/x/Deck.pptx", None),   # scheme-relative: no scheme at all
        ("/local/path/Deck.pptx", None),                  # not a URL
    ],
)
def test_a_link_that_is_not_https_is_refused_with_a_typed_reason(bad_link, expected_scheme):
    """THE REFUSAL. 422 with a machine-readable reason, so Core's callers can surface it
    under the D1/D7 STATUS-field rule instead of losing the link silently."""
    r = _embed(link=bad_link)

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["link"]["reason"] == "link_scheme_not_allowed", detail
    assert detail["link"]["scheme"] == expected_scheme, detail
    assert "message" in detail and detail["message"], detail
    # NOTE: the exact allowed-scheme LIST is pinned once, below, not here. Asserting it in
    # every case made the whole parametrisation red whenever the constant was widened, so
    # the control "widen the list -> only the http case reds" was not true of the suite as
    # written (measured). Keeping the list assertion in one place restores that isolation.


def test_the_allowed_scheme_list_on_the_wire_is_https_only():
    """The advertised contract, pinned ONCE. A caller reads `allowed_schemes` to know what
    to send, so widening it is a wire change and should redden exactly here."""
    detail = _embed(link="http://contoso.sharepoint.com/x.pptx").json()["detail"]
    assert detail["link"]["allowed_schemes"] == ["https"], detail


def test_the_refused_value_is_never_echoed_back():
    """A `javascript:` payload reflected into an error body is the same problem wearing a
    different hat. Only the SCHEME is reported."""
    payload = "javascript:alert('xss-canary-7f3a')"

    r = _embed(link=payload)

    assert r.status_code == 422, r.text
    assert "xss-canary-7f3a" not in r.text, r.text
    assert "alert(" not in r.text, r.text


def test_an_https_link_is_accepted():
    """The shape Core's connector sync actually sends must pass untouched."""
    r = _embed(link="https://contoso.sharepoint.com/sites/x/Shared%20Documents/Deck.pptx")
    assert r.status_code == 200, r.text


def test_an_upload_without_a_link_is_unaffected():
    """ABSENT IS NOT A REFUSAL. Organic uploads send no `link` (Core's
    VectorDB/crud.js), and that path must behave exactly as before."""
    r = _embed(link=None)
    assert r.status_code == 200, r.text


def test_an_empty_link_is_treated_as_absent_not_as_a_bad_scheme():
    """An empty form field is 'no link', not 'a link with no scheme'. Refusing it would
    reject callers who always send the field and sometimes have nothing to put in it."""
    r = _embed(link="")
    assert r.status_code == 200, r.text
