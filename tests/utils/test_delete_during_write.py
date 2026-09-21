"""A delete that lands during an upload must not leave a partial document (FILES-01 F02B).

WHY THIS FILE EXISTS. Measured 2026-09-21 on a real uvicorn server through the real routes,
EMBEDDING_BATCH_SIZE=1, a 7-chunk upload: DELETE /documents was sent after batch 1 had
committed. The delete removed batch 1 and answered "deleted successfully"; the upload then
wrote batches 2..7 and answered 200. The file ended with 6 of its 7 chunks -- a document
neither caller knew existed, missing its opening.

DELETE now takes the same per-file_id write lock the upload holds, so it waits for the
upload to finish and then removes the whole file. Here the lock is the in-process model
from test_simultaneous_write (same `hold` contract); the Postgres advisory lock itself is
proven there against a real database.
"""

import asyncio
import datetime
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import jwt
import pytest
import uvicorn

from main import app
from app.routes import document_routes
from app.services.file_write_lock import NoFileWriteLock
from tests.utils.test_replace_not_accumulate import FakeRow, FakeStore, V1
from tests.utils.test_simultaneous_write import InProcessFileLock

_SECRET = "testsecret"
FID = "f02b"


class PausingStore(FakeStore):
    """Commits batch 1, then holds the upload until the test lets it continue."""

    def __init__(self):
        super().__init__()
        self.first_batch_done = threading.Event()
        self.go_on = threading.Event()
        self.batches = 0

    async def aadd_documents(self, docs, ids=None, executor=None):
        loop = asyncio.get_running_loop()

        def commit():
            for d in docs:
                self.rows.append(FakeRow(ids[0], d.page_content, d.metadata))
            self.batches += 1
            if self.batches == 1:
                self.first_batch_done.set()
                assert self.go_on.wait(20), "test never let the upload continue"
            return ids

        return await loop.run_in_executor(executor, commit)

    async def get_filtered_ids(self, ids, user_id=None, document_origin_type=None,
                               subscription_id=None, executor=None):
        return sorted({r.custom_id for r in self.rows if r.custom_id in ids})


def _headers():
    os.environ["JWT_SECRET"] = _SECRET
    tok = jwt.encode(
        {"id": "u", "tid": "tenantA", "ent": ["userA"], "act": ["write", "delete"],
         "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
        _SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {tok}"}


class _WatchedLock:
    """Wraps a lock and signals when a SECOND holder (the delete) has reached it, so the test
    can release the paused upload at a known point instead of after a fixed sleep."""

    def __init__(self, inner):
        self.inner = inner
        self.second_holder_arrived = threading.Event()
        self._calls = 0

    @asynccontextmanager
    async def hold(self, file_id, stop_waiting=None, max_wait=0):
        self._calls += 1
        if self._calls >= 2:
            self.second_holder_arrived.set()
        async with self.inner.hold(file_id, stop_waiting=stop_waiting, max_wait=max_wait):
            yield


def _run(monkeypatch, lock):
    lock = _WatchedLock(lock)
    store = PausingStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 1, raising=False)
    monkeypatch.setattr(document_routes, "file_write_lock", lock)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=8)

    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not srv.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    base = f"http://127.0.0.1:{srv.servers[0].sockets[0].getsockname()[1]}"
    out = {}

    def upload():
        r = httpx.post(f"{base}/embed", timeout=30, headers=_headers(),
                       data={"file_id": FID, "entity_id": "userA"},
                       files={"file": ("doc.txt", io.BytesIO((V1 * 3).encode()), "text/plain")})
        out["upload"] = r.status_code

    def delete():
        r = httpx.request("DELETE", f"{base}/documents", timeout=30, headers=_headers(),
                          json={"file_ids": [FID], "entity_id": "userA"})
        out["delete"] = r.status_code

    try:
        up = threading.Thread(target=upload)
        up.start()
        assert store.first_batch_done.wait(20), "the upload never reached its first batch"
        dl = threading.Thread(target=delete)
        dl.start()
        assert lock.second_holder_arrived.wait(20), "the delete never reached the file lock"
        # Without a real lock the delete now runs to completion; with one it stays blocked,
        # and this join simply times out. Either way the state below is settled.
        dl.join(2)
        out["rows_while_upload_paused"] = len(store.documents_for(FID))
        store.go_on.set()
        up.join(30)
        dl.join(30)
    finally:
        store.go_on.set()
        srv.should_exit = True
        thread.join(10)
    assert store.batches > 1, "precondition: the upload really spanned several batches"
    return store, out


def test_without_the_lock_a_mid_upload_delete_leaves_a_partial_document(monkeypatch):
    """The measured defect, so the fixture is proven able to show it."""
    store, out = _run(monkeypatch, NoFileWriteLock())
    assert out["upload"] == 200 and out["delete"] == 200
    remaining = len(store.documents_for(FID))
    assert 0 < remaining < store.batches, (remaining, store.batches)


def test_a_delete_waits_for_the_upload_and_removes_the_whole_file(monkeypatch):
    store, out = _run(monkeypatch, InProcessFileLock())
    assert out["rows_while_upload_paused"] == 1, (
        "the delete ran while the upload was still writing"
    )
    assert out["upload"] == 200 and out["delete"] == 200, out
    assert store.documents_for(FID) == [], "a partial document survived the delete"
