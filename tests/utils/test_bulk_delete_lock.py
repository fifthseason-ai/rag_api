"""A BULK delete that lands during an upload must not leave a partial document (F-BULK-DELETE-LOCK).

WHY THIS FILE EXISTS. DELETE /documents with NO file_ids is a deliberate bulk purge: it
removes every row of the entity (Core uses it for knowledge-base delete and folder unlink;
see FILES-DEV/F-DELETE-SCOPE-MEASURE-2026-09-22.md). F02B made a per-file delete wait for
the upload's write lock, but the lock set was built from the REQUEST's file_ids -- an empty
list -- so a bulk purge locked nothing. Reproduced here red-first on origin/main a4b47a6:
the purge landed after batch 1 of a multi-batch upload, removed that batch, answered 200;
the upload then wrote batches 2..N and answered 200. The file survived the purge with its
opening missing, and neither caller knew.

The fix resolves the purge to the entity's file_ids first, takes each of THOSE files' write
locks (the same sorted _hold_file_locks the per-file path uses), and then deletes exactly
the locked set, never the open filter. A file whose first batch had not landed when the
purge resolved is therefore left WHOLE (it is ordered after the purge), never halved.

The store here is a model of the table that honours the empty-file_ids semantics of
get_filtered_ids/_delete_multiple (empty = every row of the entity, still entity-scoped);
the real advisory lock variant runs when RAG_TEST_PG_DSN points at an isolated Postgres.
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

from main import app
from app.routes import document_routes
from app.services.file_write_lock import NoFileWriteLock
from tests.utils.test_delete_during_write import PausingStore
from tests.utils.test_replace_not_accumulate import FakeRow, V1
from tests.utils.test_simultaneous_write import InProcessFileLock, PG_DSN, needs_pg, _pg_lock

_SECRET = "testsecret"
FID = "bulk-upload"
OTHER = "bulk-other"  # an already-stored file of the same entity
FOREIGN = "bulk-foreign"  # another entity's file; a bulk purge must never reach it


class EntityStore(PausingStore):
    """PausingStore whose id lookup and delete honour an EMPTY id list the way
    _delete_multiple and get_filtered_ids do: no id clause, every row of the entity.
    Without this the double cannot express a bulk purge at all -- the parent's lookup
    answers [] for [] and would make every assertion below vacuous."""

    def __init__(self):
        super().__init__()
        self.delete_calls = []

    @staticmethod
    def _match(r, ids, user_id):
        return (not ids or r.custom_id in ids) and (
            user_id is None or r.metadata.get("user_id") == user_id
        )

    async def get_filtered_ids(self, ids, user_id=None, document_origin_type=None,
                               subscription_id=None, executor=None):
        return sorted({r.custom_id for r in self.rows if self._match(r, ids, user_id)})

    async def delete(self, ids=None, collection_only=False, user_id=None,
                     document_origin_type=None, subscription_id=None, text_source=None,
                     executor=None, **_):
        self.delete_calls.append(list(ids or []))
        self.rows = [r for r in self.rows if not self._match(r, ids, user_id)]


def _headers():
    os.environ["JWT_SECRET"] = _SECRET
    tok = jwt.encode(
        {"id": "u", "tid": "tenantA", "ent": ["userA"], "act": ["write", "delete"],
         "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)},
        _SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {tok}"}


def _run(monkeypatch, lock, on_server_loop_exit=None):
    store = EntityStore()
    store.rows.append(FakeRow(OTHER, "already stored", {"user_id": "userA"}))
    store.rows.append(FakeRow(FOREIGN, "someone else's", {"user_id": "userB"}))
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

    def purge():
        # The bulk purge: entity only, NO file_ids.
        r = httpx.request("DELETE", f"{base}/documents", timeout=30, headers=_headers(),
                          json={"entity_id": "userA"})
        out["delete"] = r.status_code
        out["delete_body"] = r.json()

    try:
        up = threading.Thread(target=upload)
        up.start()
        assert store.first_batch_done.wait(20), "the upload never reached its first batch"
        dl = threading.Thread(target=purge)
        dl.start()
        # Unlocked, the purge finishes well inside this; locked, it is still waiting.
        dl.join(3)
        out["purge_finished_while_upload_paused"] = not dl.is_alive()
        out["rows_while_upload_paused"] = len(store.documents_for(FID))
        store.go_on.set()
        up.join(30)
        dl.join(30)
        if on_server_loop_exit is not None:
            on_server_loop_exit()
    finally:
        store.go_on.set()
        srv.should_exit = True
        thread.join(10)
    assert store.batches > 1, "precondition: the upload really spanned several batches"
    return store, out


def _assert_whole_purge(store, out):
    assert out["upload"] == 200 and out["delete"] == 200, out
    assert not out["purge_finished_while_upload_paused"], (
        "the bulk purge ran while the upload was still writing (it took no file lock)"
    )
    assert out["rows_while_upload_paused"] == 1, out
    assert store.documents_for(FID) == [], (
        f"a partial document survived the bulk purge: "
        f"{len(store.documents_for(FID))} of {store.batches} chunks"
    )
    assert store.documents_for(OTHER) == [], "the purge missed a stored file of the entity"
    assert store.documents_for(FOREIGN) == ["someone else's"], "the purge crossed entities"
    # The delete names the locked files; it never falls back to the open entity filter,
    # which could halve a file that started writing after the locks were taken.
    assert store.delete_calls and all(store.delete_calls), store.delete_calls
    assert out["delete_body"]["message"].startswith("Documents for 2 files"), out


def test_without_a_lock_a_bulk_purge_mid_upload_leaves_a_partial_document(monkeypatch):
    """The defect, so the fixture is proven able to show it (fails if it cannot)."""
    store, out = _run(monkeypatch, NoFileWriteLock())
    assert out["upload"] == 200 and out["delete"] == 200, out
    remaining = len(store.documents_for(FID))
    assert 0 < remaining < store.batches, (remaining, store.batches)
    assert store.documents_for(FOREIGN) == ["someone else's"]


def test_a_bulk_purge_waits_for_the_upload_and_removes_the_whole_file(monkeypatch):
    store, out = _run(monkeypatch, InProcessFileLock())
    _assert_whole_purge(store, out)


def test_a_bulk_purge_of_an_empty_entity_is_still_a_clean_200(monkeypatch):
    """No rows -> nothing to lock, nothing deleted, no 404 (bulk is not an id lookup)."""
    store = EntityStore()
    store.rows.append(FakeRow(FOREIGN, "someone else's", {"user_id": "userB"}))
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "file_write_lock", InProcessFileLock())
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=8)
    from fastapi.testclient import TestClient

    r = TestClient(app).request("DELETE", "/documents", headers=_headers(),
                                json={"entity_id": "userA"})
    assert r.status_code == 200, r.text
    assert store.documents_for(FOREIGN) == ["someone else's"]
    assert store.delete_calls == [], "an empty purge must not issue an open-filter delete"


@needs_pg
def test_real_advisory_lock_serializes_a_bulk_purge_with_an_upload(monkeypatch):
    lock, state = _pg_lock(first_poll=0.01, max_poll=0.02)
    loop_box = {}
    get_pool = lock._get_pool if hasattr(lock, "_get_pool") else None

    def close_pool():
        pool, loop = state.get("pool"), loop_box.get("loop")
        if pool is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(pool.close(), loop).result(10)

    if get_pool is not None:
        async def remembering_get_pool():
            loop_box["loop"] = asyncio.get_running_loop()
            return await get_pool()

        lock._get_pool = remembering_get_pool
    store, out = _run(monkeypatch, lock, on_server_loop_exit=close_pool)
    _assert_whole_purge(store, out)
