"""A caller that stopped waiting must never find its upload stored afterwards (FILES-01 F04).

WHY THIS FILE EXISTS. Measured 2026-09-21 against a REAL uvicorn server, the real middleware
stack and the real /embed route, with a store whose insert takes 2 s:

    client gives up (ReadTimeout)        0.81 s
    insert commits                       2.05 s
    route returns its success receipt    2.05 s   -- to nobody
    rows for the file afterwards         7

On the batched path (EMBEDDING_BATCH_SIZE > 0, the production default is 500) every later
batch kept inserting after the caller had gone. Starlette does not cancel a handler when
its client disconnects. Core's /embed client times out at 120 s and tells the uploader the
upload FAILED -- so a document reported as failed became retrievable later, and a retry of
that "failed" upload wrote the same file twice.

WHY A REAL SERVER. TestClient cannot express a caller who leaves mid-request: it waits for
the response by construction. The first fix used `request.is_disconnected()` and would have
passed any TestClient test; against the real server it answered "connected" at every check
for 8 s after the client had gone, because each BaseHTTPMiddleware layer turns its
zero-timeout poll into a cancellation that always wins. Only a real socket closing showed
that. So the outcome tests below start uvicorn in-process on a free port and use a real
HTTP client with a real timeout.

DETERMINISM. The store's insert BLOCKS on an event that the test sets only after the client
has given up, so "the caller left during the insert" is arranged, not hoped for.
"""

import asyncio
import datetime
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import jwt
import pytest
import uvicorn
from fastapi import HTTPException
from langchain_core.documents import Document

from main import app
from app.routes import document_routes
from tests.utils.test_replace_not_accumulate import FakeRow, FakeStore, V1, V2

_SECRET = "testsecret"
FID = "f04-abandoned"


def _token():
    os.environ["JWT_SECRET"] = _SECRET
    return jwt.encode(
        {
            "id": "testuser",
            "tid": "tenantA",
            "ent": ["userA"],
            "act": ["write"],
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
        },
        _SECRET,
        algorithm="HS256",
    )


class GatedStore(FakeStore):
    """The table model from test_replace_not_accumulate, with an insert that runs in an
    executor thread (as AsyncPgVector's does) and cannot commit until the test says so."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.insert_started = threading.Event()
        self.inserts = 0

    async def aadd_documents(self, docs, ids=None, executor=None):
        loop = asyncio.get_running_loop()

        def commit():
            self.inserts += 1
            self.insert_started.set()
            assert self.release.wait(20), "test never released the insert"
            for d in docs:
                self.rows.append(FakeRow(ids[0], d.page_content, d.metadata))
            return ids

        return await loop.run_in_executor(executor, commit)


@pytest.fixture()
def server(monkeypatch):
    """uvicorn on a free port, lifespan off (it would open a real Postgres pool)."""
    store = GatedStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    os.environ["JWT_SECRET"] = _SECRET
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=4)

    finished = threading.Event()
    real_store = document_routes.store_data_in_vector_db

    async def observed(*args, **kwargs):
        # Observes the real function; changes nothing. Lets a test wait for the handler
        # to finish instead of sleeping and hoping.
        try:
            return await real_store(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", observed)

    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not srv.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    port = srv.servers[0].sockets[0].getsockname()[1]
    try:
        yield store, f"http://127.0.0.1:{port}", finished
    finally:
        store.release.set()
        srv.should_exit = True
        thread.join(10)


def _post(base, text, *, timeout, replace=None, filename="policy.txt"):
    data = {"file_id": FID, "entity_id": "userA"}
    if replace is not None:
        data["replace"] = str(replace).lower()
    return httpx.post(
        f"{base}/embed",
        data=data,
        files={"file": (filename, io.BytesIO(text.encode("utf-8")), "text/plain")},
        headers={"Authorization": f"Bearer {_token()}"},
        timeout=timeout,
    )


def _abandon_during_insert(store, base, finished, text, **kw):
    """Client gives up while the insert is blocked; then the insert is allowed to commit."""
    with pytest.raises(httpx.ReadTimeout):
        _post(base, text, timeout=1.0, **kw)
    assert store.insert_started.is_set(), "the request never reached the insert"
    time.sleep(0.3)  # let the server's connection_lost land before the commit
    store.release.set()
    assert finished.wait(15), "the handler never finished"


# ---------------------------------------------------------------------------
# The measured regression
# ---------------------------------------------------------------------------


def test_a_caller_who_left_during_the_insert_does_not_get_the_file_stored(server):
    store, base, finished = server
    _abandon_during_insert(store, base, finished, V1)

    assert store.inserts == 1, "precondition: the insert really ran and committed"
    assert store.documents_for(FID) == [], (
        "the caller timed out and was told the upload failed, yet the file is retrievable"
    )


def test_an_abandoned_replacement_keeps_the_version_it_was_superseding(server):
    """The rollback must remove only what the abandoned call wrote. Deleting by file_id
    would also destroy V1 -- still the only good copy, since V2 never completed."""
    store, base, finished = server
    for d in (V1,):
        store.rows.append(FakeRow(FID, d, {"user_id": "userA", "tenant_id": "tenantA",
                                           "filename": "policy-v1.txt"}))
    v1_uuids = store.uuids_for(FID)
    assert v1_uuids, "precondition: a prior version exists"

    _abandon_during_insert(store, base, finished, V2, replace=True, filename="policy-v2.txt")

    assert store.uuids_for(FID) == v1_uuids, "the prior version was not kept intact"
    assert all("900 EUR" not in d for d in store.documents_for(FID))


def test_batched_path_stops_inserting_once_the_caller_has_gone(server, monkeypatch):
    store, base, finished = server
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    text = V1 * 3
    _abandon_during_insert(store, base, finished, text)

    assert store.inserts == 1, (
        f"{store.inserts} batches were inserted; every batch after the caller left is a late write"
    )
    assert store.documents_for(FID) == []


def _seed_prior_version(store):
    store.rows.append(FakeRow(FID, V1, {"user_id": "userA", "tenant_id": "tenantA",
                                        "filename": "policy-v1.txt"}))
    prior = store.uuids_for(FID)
    assert prior, "precondition: a prior version exists"
    return prior


def test_batched_abandoned_additive_upload_keeps_the_earlier_rows(server, monkeypatch):
    """Review MINOR-2. The batched pipeline has its OWN file-wide rollback
    (`delete(ids=[file_id])`); if it ever ran on an abandoned write it would destroy the
    rows an earlier upload left. Only the capture-based undo may run here."""
    store, base, finished = server
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    prior = _seed_prior_version(store)

    _abandon_during_insert(store, base, finished, V2 * 3, filename="policy-v2.txt")

    assert store.inserts == 1, "precondition: the batched path ran and was stopped"
    assert store.uuids_for(FID) == prior
    assert "delete-by-file_id" not in store.calls, store.calls


def test_batched_abandoned_replacement_keeps_the_superseded_version(server, monkeypatch):
    store, base, finished = server
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    prior = _seed_prior_version(store)

    _abandon_during_insert(store, base, finished, V2 * 3, replace=True,
                           filename="policy-v2.txt")

    assert store.inserts == 1, "precondition: the batched path ran and was stopped"
    assert store.uuids_for(FID) == prior, "the version being superseded was not kept intact"
    assert "delete-by-file_id" not in store.calls, store.calls


def test_a_caller_who_waits_still_gets_the_file_stored(server):
    """The gate must not cost the ordinary case anything."""
    store, base, finished = server
    store.release.set()
    r = _post(base, V1, timeout=15)
    assert r.status_code == 200, r.text
    assert store.documents_for(FID), "a waiting caller's upload was not stored"


def test_a_waiting_caller_on_the_batched_path_gets_every_batch(server, monkeypatch):
    store, base, finished = server
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    store.release.set()
    r = _post(base, V1 * 3, timeout=15)
    assert r.status_code == 200, r.text
    assert store.inserts > 1, "precondition: the fixture really spans several batches"
    assert len(store.documents_for(FID)) == store.inserts


# ---------------------------------------------------------------------------
# The store seam directly: additive default, and a store that cannot list rows
# ---------------------------------------------------------------------------


def _gone_after(n_true):
    """A still_wanted probe that answers True n times, then False."""
    answers = iter([True] * n_true)

    async def probe():
        return next(answers, False)

    return probe


def test_without_replace_an_abandoned_write_keeps_the_rows_an_earlier_upload_left(monkeypatch):
    """The additive default keeps earlier rows on purpose (the OCR->native swap depends on
    it). Rolling back by file_id would delete them; the rollback must not."""
    store = FakeStore()
    store.rows.append(FakeRow(FID, "earlier upload", {"user_id": "userA"}))
    earlier = store.uuids_for(FID)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V2, metadata={})], FID, "userA",
            executor=pool, still_wanted=_gone_after(1),
        ))

    assert err.value.status_code == document_routes.CALLER_GONE_STATUS
    assert err.value.detail["stage"] == "after insert"
    assert "insert" in store.calls, "precondition: the insert happened before the caller left"
    assert store.uuids_for(FID) == earlier


def test_a_caller_gone_before_the_insert_writes_nothing(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA",
            executor=pool, still_wanted=_gone_after(0),
        ))

    assert err.value.detail["stage"] == "before insert"
    assert "insert" not in store.calls
    assert err.value.detail["rows_removed"] == 0


def test_a_store_that_cannot_list_rows_reports_unknown_not_zero(monkeypatch):
    """Without row listing nothing can be removed safely. The answer is 'unknown', never a
    0 that reads as 'nothing was left behind'."""

    class NoListing(FakeStore):
        @property
        def get_row_uuids(self):  # hasattr() is False, as for a store without the method
            raise AttributeError("get_row_uuids")

    store = NoListing()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    assert not hasattr(store, "get_row_uuids"), "precondition: the store cannot list rows"

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA",
            executor=pool, still_wanted=_gone_after(1),
        ))

    assert err.value.detail["rows_removed"] is None


class _CaptureFails(FakeStore):
    async def get_row_uuids(self, file_id, user_id=None, tenant_id=None, executor=None):
        raise RuntimeError("listing unavailable")


def test_a_failed_capture_does_not_cost_a_waiting_caller_the_upload(monkeypatch):
    """The capture exists only for the abandoned case; it must never fail the ordinary one."""
    store = _CaptureFails()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)

    with ThreadPoolExecutor(2) as pool:
        result = asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA",
            executor=pool, still_wanted=_gone_after(99),
        ))

    assert result["ids"], "a caller who was still waiting lost the upload"
    assert store.documents_for(FID)


def test_a_failed_capture_reports_the_abandoned_rows_as_unknown(monkeypatch):
    store = _CaptureFails()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA",
            executor=pool, still_wanted=_gone_after(1),
        ))

    assert err.value.detail["rows_removed"] is None
