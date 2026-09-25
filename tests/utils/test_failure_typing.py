"""FAILED / UNSUPPORTED typing: can a consumer branch without parsing prose? (CARD-P2-01 S1, part D).

MEASURED on main `afd44b9` before this change:

  422 verdict   detail.extraction.status == "unsupported" + detail.extraction.verdict
                (encrypted | corrupt | unsupported)                          TYPED  -> pinned below
  422 empty     detail.extraction.status == "empty"                          TYPED  -> pinned below
  422 link      detail.link.reason == "link_scheme_not_allowed"              TYPED  -> pinned below
  422 request   detail is a LIST (FastAPI validation), message "Request validation failed"
                                                                             TYPED by shape -> pinned
  503           status alone = ours, transient, retry                        TYPED by status
  400 x5        pandoc missing / LibreOffice missing / file name too long / cause undetermined /
                upload path refused -- ONE status, five different things a consumer must do
                (operator / operator-or-resave / rename / do-not-loop / fix the request).
                Distinguishable ONLY by reading the sentence.                PROSE  -> FIXED
  reference     on every 503/400 and the read-route 500s, the reference exists ONLY inside the
                sentence ("Reference: abc123def456." / "Quote reference ...").  PROSE  -> FIXED

THE FIX is additive and changes no status code and no body: `detail` stays a plain STRING on these
paths on purpose (Core interpolates it into a toast; an object renders `[object Object]`), so the
typed half rides two RESPONSE HEADERS --
  X-Failure-Attribution  the SAME token the log line already carries as `[attribution=...]`
  X-Error-Reference      the reference, as a field
-- one vocabulary shared by the operator's log and the consumer's branch.

RED-FIRST: on `afd44b9` none of these responses carries either header, so every header assertion
fails; the verdict/empty/link/validation pins pass there (they document what was already typed).
Faults are INJECTED at the loader or the store; no dependency is removed, no provider is called.
"""

import asyncio
import datetime
import errno
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document

os.environ.setdefault("JWT_SECRET", "testsecret")

from app.routes import document_routes  # noqa: E402
from app.routes.document_routes import (  # noqa: E402
    _assert_extractable_content,
    _reject_ungoverned_link,
    load_file_content,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector  # noqa: E402
from app.utils.document_loader import (  # noqa: E402
    CorruptDocumentError,
    EncryptedDocumentError,
    UnsupportedDocumentError,
)
from main import app  # noqa: E402

ATTRIBUTION = "X-Failure-Attribution"
REFERENCE = "X-Error-Reference"
_REF_RE = re.compile(r"^[0-9a-f]{12}$")

_SOFFICE = "soffice command was not found. Please install libreoffice\non your system and try again."
_PANDOC = "No pandoc was found: either install pandoc and add it to your PATH or ..."

#: (injected exception, status UNCHANGED from main, the typed token a consumer branches on)
SEAM_CASES = {
    "service": (MemoryError("cannot allocate"), 503, "service"),
    "undetermined": (ValueError("corrupt sector table at 0x1f40"), 400, "undetermined"),
    "pandoc": (OSError(_PANDOC), 400, "service:pandoc_not_installed"),
    "libreoffice": (FileNotFoundError(_SOFFICE), 400, "service:libreoffice_not_installed"),
    "name_too_long": (OSError(errno.ENAMETOOLONG, "File name too long"), 400, "content:name_too_long"),
}


class _Raising:
    def __init__(self, exc):
        self._exc = exc

    def lazy_load(self):
        raise self._exc


def _drive(exc, filename="q4.xlsx"):
    """The REAL `load_file_content` seam with the loader replaced. Returns the HTTPException."""
    with patch("app.routes.document_routes.get_loader", return_value=(_Raising(exc), True, "xlsx")):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(load_file_content(filename, "application/vnd.ms-excel", "/tmp/x.xlsx",
                                          ThreadPoolExecutor(max_workers=1)))
    return caught.value


def _hdr(act):
    payload = {
        "id": "userA", "tid": "tenantA", "ent": ["userA"], "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")}


def _client():
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


# ---------------------------------------------------------------------------------------------
# The five 400/503 intake failures: typed by header, status and sentence unchanged
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(SEAM_CASES))
def test_each_intake_failure_carries_its_typed_attribution_and_reference(case, caplog):
    exc, status_code, token = SEAM_CASES[case]
    with caplog.at_level(logging.ERROR):
        err = _drive(exc)
    assert err.status_code == status_code, "the status code changed: %s" % err.status_code
    assert isinstance(err.detail, str), "detail must stay a plain string (Core's toast)"
    headers = err.headers or {}
    assert headers.get(ATTRIBUTION) == token, (
        "%s: a consumer cannot tell this 400/503 apart from the others without reading the "
        "sentence -- headers=%r" % (case, headers))
    ref = headers.get(REFERENCE)
    assert ref and _REF_RE.match(ref), headers
    assert ("Reference: %s." % ref) in err.detail, "the header and the sentence name different references"
    # ONE vocabulary: the log line the operator reads carries the same token and reference.
    assert ("[reference=%s]" % ref) in caplog.text
    assert ("[attribution=%s]" % token) in caplog.text


def test_the_tokens_are_distinct_and_all_declared():
    """Five different answers must not collapse into one, and every token the paths can send is
    in the published vocabulary (`FAILURE_ATTRIBUTIONS`), so a consumer can enumerate them."""
    tokens = [t for _e, _s, t in SEAM_CASES.values()] + ["request:invalid_path"]
    assert len(set(tokens)) == len(tokens)
    assert set(tokens) == set(document_routes.FAILURE_ATTRIBUTIONS)


def test_the_typing_survives_the_http_layer_on_embed_and_text():
    """A header set on an HTTPException is only evidence once it is on the RESPONSE. Driven through
    the real routes: /embed and /text, loader failing with an out-of-memory (503, ours)."""
    client = _client()
    with patch("app.routes.document_routes.get_loader",
               return_value=(_Raising(MemoryError("oom")), True, "txt")):
        for path, act in (("/embed", ["write"]), ("/text", ["read"])):
            r = client.post(path, data={"file_id": "f1", "entity_id": "userA"},
                            files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")},
                            headers=_hdr(act))
            assert r.status_code == 503, (path, r.status_code, r.text)
            assert isinstance(r.json()["detail"], str), r.text
            assert r.headers.get(ATTRIBUTION) == "service", (path, dict(r.headers))
            ref = r.headers.get(REFERENCE)
            assert ref and ref in r.json()["detail"], (path, dict(r.headers), r.text)


def test_a_store_failure_after_loading_is_typed_on_the_outer_handler(monkeypatch):
    """The OTHER call site: a vector-store outage raised by the insert reaches /embed's outer
    `except Exception` (not the loader seam). It must be typed exactly the same way."""
    async def outage(self, docs, ids=None, executor=None):
        raise ConnectionError("vector store unreachable")

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", outage)
    r = _client().post("/embed", data={"file_id": "f-store", "entity_id": "userA"},
                       files={"file": ("a.txt", io.BytesIO(b"real text to embed"), "text/plain")},
                       headers=_hdr(["write"]))
    assert r.status_code == 503, r.text
    assert r.headers.get(ATTRIBUTION) == "service", dict(r.headers)
    assert r.headers.get(REFERENCE) in r.json()["detail"]


@pytest.mark.parametrize("path,act", [("/embed", ["write"]), ("/embed-upload", ["write"]), ("/text", ["read"])])
def test_a_refused_upload_path_is_typed(path, act):
    """The traversal refusal answers 400 BEFORE any work -- the same status as "cause
    undetermined", for an entirely different fix. Typed now; no reference (nothing was logged
    under one, and none is invented)."""
    field = "uploaded_file" if path == "/embed-upload" else "file"
    r = _client().post(path, data={"file_id": "f1", "entity_id": "../../etc"},
                       files={field: ("safe.txt", io.BytesIO(b"x"), "text/plain")},
                       headers=_hdr(act))
    assert r.status_code == 400, r.text
    assert r.headers.get(ATTRIBUTION) == "request:invalid_path", dict(r.headers)
    assert REFERENCE not in r.headers


def test_read_route_500_carries_its_reference_as_a_field(monkeypatch):
    """`client_safe_error` (the read routes) quotes a reference only inside the sentence."""
    async def broken(*_a, **_k):
        raise RuntimeError("store exploded with internals")

    monkeypatch.setattr(document_routes, "_retrieve_documents", broken)
    monkeypatch.setattr(document_routes, "get_cached_query_embedding", lambda q: [0.1])
    r = _client().post("/query", json={"query": "q", "file_id": "f1", "k": 1, "entity_id": "userA"},
                       headers=_hdr(["read"]))
    assert r.status_code == 500, r.text
    ref = r.headers.get(REFERENCE)
    assert ref and _REF_RE.match(ref) and ref in r.json()["detail"], (dict(r.headers), r.text)


# ---------------------------------------------------------------------------------------------
# PINS -- what was already typed on main, named by field so a regression reddens here
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [EncryptedDocumentError, CorruptDocumentError, UnsupportedDocumentError])
def test_pin_a_terminal_verdict_is_typed_by_status_and_verdict(cls):
    err = _drive(cls("synthetic verdict"))
    assert err.status_code == 422
    assert isinstance(err.detail, dict)
    assert err.detail["extraction"]["status"] == "unsupported"
    assert err.detail["extraction"]["verdict"] == cls.verdict
    assert cls.verdict in ("encrypted", "corrupt", "unsupported")


def test_pin_an_empty_extraction_is_typed_empty():
    with pytest.raises(HTTPException) as caught:
        _assert_extractable_content([Document(page_content="  \n ", metadata={})], "blank.txt")
    assert caught.value.status_code == 422
    assert caught.value.detail["extraction"]["status"] == "empty"


def test_pin_a_refused_link_is_typed_by_reason():
    with pytest.raises(HTTPException) as caught:
        _reject_ungoverned_link("javascript:alert(1)", "f1")
    assert caught.value.status_code == 422
    assert caught.value.detail["link"]["reason"] == "link_scheme_not_allowed"


def test_pin_request_validation_422_is_distinguishable_by_shape():
    """Four different 422s share a status. A consumer tells them apart by SHAPE: a validation
    error's `detail` is a LIST; the verdict/empty bodies carry `detail.extraction`; the link
    refusal carries `detail.link`."""
    r = _client().post("/embed", data={"entity_id": "userA"}, headers=_hdr(["write"]))
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], list) and body["message"] == "Request validation failed", body
