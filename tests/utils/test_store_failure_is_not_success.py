"""A failed vector-store write must never be reported to the caller as success
(FILES-01 F3).

The defect, found by tracing what `/local/embed` does with its result
=====================================================================
`store_data_in_vector_db` signals failure by RETURNING A DICT:

    success:  {"message": "Documents added successfully", "ids": [...], "docs": [...]}
    failure:  {"message": "An error occurred while adding documents.", "error": str(e)}

Both are truthy. Its four callers then disagree about how to read that, and only one
of them is right:

    /summarize     `if not result or "error" in result:`   <- correct
    /local/embed   `if result:`                            <- ALWAYS true
    /embed-upload  `if not result:`                        <- NEVER true
    /embed         does have an `if "error" in result` check, but it sets
                   response_message to the RAW EXCEPTION and then falls through to
                   the 200 success return

Measured against the real routes with a simulated pgvector outage, not inferred:

    /local/embed   200  {"status": true,  ...}                    fake success
    /embed-upload  200  {"status": true,  "message": "File processed successfully."}
    /embed         200  {"status": false, "message": "<raw exception text>"}

So two routes tell the uploader and Core that a file with zero rows was ingested,
and nothing retries because nothing failed as far as any consumer can tell. The
third at least says `status: false` -- but over HTTP **200**, which a listener
keying retry on 5xx will never act on, and with `str(e)` handed to the caller: the
exact disclosure this lane closed everywhere else on this surface.

This is the exact inverse of the failure this programme started from. There, a file
that COULD be ingested was reported as bad. Here, a file that was NOT ingested is
reported as fine -- and it is worse, because a false failure is visible and a false
success is not.

The fix is at the shared source, not in three callers
=====================================================
Three consumers reading one producer's return value three different ways is the
defect; patching each reader would leave the next caller free to get it wrong again.
`store_data_in_vector_db` now returns `None` on failure, which makes every existing
check correct at once -- `if result:`, `if not result:` and `"error" in result` all
agree on a falsy value. The exception and traceback are still logged at the point of
failure, so no diagnostic detail is lost.

Everything here is synthetic: the store is simulated and made to fail; no client
content, no network, no database.
"""

import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient

from main import app
from app.routes import document_routes
from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "test_key"
_OUTAGE = "pgvector connection refused FILES01-STORE-OUTAGE"


def _hdr(ent, act, tid="tenantA", uid="testuser"):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": uid,
        "tid": tid,
        "ent": ent,
        "act": act,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _write_hdr(entity="userA"):
    return _hdr(ent=[entity], act=["write"])


@pytest.fixture()
def client():
    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return TestClient(app)


@pytest.fixture()
def failing_store(monkeypatch):
    """Make the REAL store path fail the way an outage does.

    Deliberately NOT a patch of `store_data_in_vector_db` itself: patching the
    producer would prove nothing about how it reports failure. The exception is
    raised where a genuine pgvector outage raises it, and the producer's own
    `except` turns it into whatever it turns it into.
    """

    async def boom(self, docs, ids=None, executor=None):
        raise ConnectionError(_OUTAGE)

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", boom)


@pytest.fixture()
def working_store(monkeypatch):
    recorded = []

    async def ok(self, docs, ids=None, executor=None):
        recorded.append(list(docs))
        return ids or ["id1"]

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", ok)
    return recorded


def _embed(client, filename="notes.txt", content=b"real extractable content"):
    return client.post(
        "/embed",
        data={"file_id": "f-store-1", "entity_id": "userA"},
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        headers=_write_hdr(),
    )


def _embed_upload(client, filename="notes.txt", content=b"real extractable content"):
    return client.post(
        "/embed-upload",
        data={"file_id": "f-store-2", "entity_id": "userA"},
        files={"uploaded_file": (filename, io.BytesIO(content), "text/plain")},
        headers=_write_hdr(),
    )


def _local_embed(client, tmp_name):
    return client.post(
        "/local/embed",
        json={
            "file_id": "f-store-3",
            "filename": os.path.basename(tmp_name),
            "filepath": os.path.basename(tmp_name),
            "file_content_type": "text/plain",
        },
        params={"entity_id": "userA"},
        headers=_write_hdr(),
    )


@pytest.fixture()
def local_file():
    """A real file inside RAG_UPLOAD_DIR, since /local/embed validates the path."""
    from app.config import RAG_UPLOAD_DIR

    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)
    path = os.path.join(RAG_UPLOAD_DIR, "files01-local-store.txt")
    with open(path, "wb") as fh:
        fh.write(b"real extractable content")
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


# ===========================================================================
# The defect: three intake routes reporting a failed write as success
# ===========================================================================


def test_embed_does_not_report_success_when_the_store_failed(client, failing_store):
    r = _embed(client)

    assert r.status_code != 200, (
        "a vector-store outage was reported to the caller as a successful ingest; "
        f"body={r.text}"
    )
    assert '"status":true' not in r.text.replace(" ", "").lower()
    # Our outage, not the file's fault -- so it must be retryable once the store is back.
    assert r.status_code >= 500, f"an outage must not be permanent: {r.status_code} {r.text}"
    assert _OUTAGE not in r.text, "the raw exception reached the caller"


def test_embed_upload_does_not_report_success_when_the_store_failed(
    client, failing_store
):
    r = _embed_upload(client)

    assert r.status_code != 200, (
        "a vector-store outage was reported to the caller as a successful ingest; "
        f"body={r.text}"
    )
    assert r.status_code >= 500, f"an outage must not be permanent: {r.status_code} {r.text}"
    assert _OUTAGE not in r.text


def test_local_embed_does_not_report_success_when_the_store_failed(
    client, failing_store, local_file
):
    """The worst of the three: `if result:` is true for the FAILURE dict, so this
    route returned `200 {"status": true}` with zero rows written."""
    r = _local_embed(client, local_file)

    assert r.status_code != 200, (
        "a vector-store outage was reported to the caller as a successful ingest; "
        f"body={r.text}"
    )
    body = r.text
    assert '"status":true' not in body.replace(" ", "").lower()
    assert _OUTAGE not in body


# ===========================================================================
# Negative control -- the fix must not make every ingest fail
# ===========================================================================


def test_embed_still_succeeds_when_the_store_works(client, working_store):
    r = _embed(client)

    assert r.status_code == 200, r.text
    assert r.json()["status"] is True
    assert working_store, "nothing was actually written"


def test_embed_upload_still_succeeds_when_the_store_works(client, working_store):
    r = _embed_upload(client)

    assert r.status_code == 200, r.text
    assert working_store, "nothing was actually written"


def test_local_embed_still_succeeds_when_the_store_works(
    client, working_store, local_file
):
    r = _local_embed(client, local_file)

    assert r.status_code == 200, r.text
    assert r.json()["status"] is True
    assert working_store, "nothing was actually written"


# ===========================================================================
# The producer's own contract, pinned so the next caller cannot misread it
# ===========================================================================


@pytest.mark.asyncio
async def test_store_returns_falsy_on_failure_so_no_caller_can_misread_it(
    monkeypatch,
):
    """The root cause, stated as an invariant.

    Failure used to be a TRUTHY dict carrying an "error" key, which is why three
    callers checking `if result:` / `if not result:` all got it wrong. Any falsy
    value makes every one of those checks correct simultaneously.
    """

    async def boom(self, docs, ids=None, executor=None):
        raise ConnectionError(_OUTAGE)

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", boom)

    result = await document_routes.store_data_in_vector_db(
        data=[document_routes.Document(page_content="x", metadata={})],
        file_id="f",
        user_id="u",
        executor=None,
    )

    assert not result, (
        "failure must be falsy: a truthy failure value is what let `if result:` and "
        f"`if not result:` both report success. got {result!r}"
    )


@pytest.mark.asyncio
async def test_store_still_reports_success_truthily(monkeypatch):
    """The other half: success must stay truthy, or every route starts failing."""

    async def ok(self, docs, ids=None, executor=None):
        return ["id1"]

    monkeypatch.setattr(AsyncPgVector, "aadd_documents", ok)

    result = await document_routes.store_data_in_vector_db(
        data=[document_routes.Document(page_content="x", metadata={})],
        file_id="f",
        user_id="u",
        executor=None,
    )

    assert result, "success must remain truthy"


# ===========================================================================
# /local/embed's outer handler -- the last raw str(e) on the intake surface
# (re-review NOTE-2)
# ===========================================================================

_LOCAL_MARKER = "FILES01-LOCAL-MARKER-4b7e"


def test_local_embed_does_not_echo_the_exception_to_the_caller(
    client, monkeypatch, local_file
):
    """`ERROR_MESSAGES.DEFAULT(e)` interpolates the exception verbatim
    (`f"Something went wrong :/\n{err}"`), so anything reaching this handler handed
    the caller our internal text -- and at 400, telling Core not to retry a fault that
    may well be ours."""

    def boom(data, filename):
        raise ValueError(f"internal detail at /tmp/uploads/tenantA/x {_LOCAL_MARKER}")

    monkeypatch.setattr(document_routes, "_assert_extractable_content", boom)

    r = _local_embed(client, local_file)

    assert _LOCAL_MARKER not in r.text, f"exception text reached the caller: {r.text}"
    assert "/tmp/uploads" not in r.text
    assert "Something went wrong" not in r.text


def test_local_embed_attributes_our_outage_as_retryable(
    client, monkeypatch, local_file
):
    """Same attribution contract as /text: our fault is a retryable 503 that
    exonerates the file, not a permanent 400 blaming it."""

    def boom(data, filename):
        raise MemoryError(f"cannot allocate 4.2 GiB {_LOCAL_MARKER}")

    monkeypatch.setattr(document_routes, "_assert_extractable_content", boom)

    r = _local_embed(client, local_file)

    assert r.status_code == 503, f"got {r.status_code}: {r.text}"
    assert "not with your file" in r.text
    assert _LOCAL_MARKER not in r.text


def test_local_embed_keeps_the_actionable_pandoc_message(
    client, monkeypatch, local_file
):
    """The dead substring branch is gone; the answer it used to give must survive via
    the shared classifier, matched by type and position instead."""
    from app.constants import ERROR_MESSAGES

    def boom(data, filename):
        raise OSError("No pandoc was found: either install pandoc and add it")

    monkeypatch.setattr(document_routes, "_assert_extractable_content", boom)

    r = _local_embed(client, local_file)

    assert ERROR_MESSAGES.PANDOC_NOT_INSTALLED in r.text, r.text
    assert r.status_code == 400, r.text


def test_local_embed_filename_cannot_disguise_our_outage_as_pandoc(
    client, monkeypatch, local_file
):
    """The caller-influenceable-substring bug must not be re-introduced here either."""
    from app.constants import ERROR_MESSAGES

    def boom(data, filename):
        raise PermissionError(
            13, "Permission denied: '/tmp/uploads/tenantA/No pandoc was found.txt'"
        )

    monkeypatch.setattr(document_routes, "_assert_extractable_content", boom)

    r = _local_embed(client, local_file)

    assert r.status_code == 503, f"got {r.status_code}: {r.text}"
    assert ERROR_MESSAGES.PANDOC_NOT_INSTALLED not in r.text
