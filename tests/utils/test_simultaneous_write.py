"""Two writes of one file_id at the same moment must not leave two versions (FILES-01 F02).

WHY THIS FILE EXISTS. Measured 2026-09-21 through the real /embed route on a real uvicorn
server: two `replace=true` uploads of file_id `f02`, sent together, both answered 200 and
the file ended holding BOTH new versions. Each call captured the same original row,
inserted its own version and deleted only its capture:

    A: superseded_rows 1, removed 1, status complete     <- false
    B: superseded_rows 1, removed 0, status incomplete
    end state: filenames ['A.txt', 'B.txt']

The fix holds a per-file_id write lock from before the captures to after the last
delete (app/services/file_write_lock.py), so the second writer supersedes the first.

TWO LAYERS OF EVIDENCE, and which is which:
- The ROUTE-LEVEL race below runs everywhere. Its lock is an in-process model with the
  same `hold` contract; it proves the store path takes the lock around the whole
  capture -> insert -> delete sequence, and its control shows the race returns without it.
- The ADVISORY-LOCK tests need a real Postgres and run only when RAG_TEST_PG_DSN is set
  (CI has no Postgres service, so there they SKIP, visibly). Run locally against an
  isolated container, e.g.
      docker run -d --name f02-pg -e POSTGRES_PASSWORD=f02 postgres:16
  and RAG_TEST_PG_DSN=postgresql://postgres:f02@<host>:5432/postgres.
  Rows are still the in-memory table model; the LOCK is real.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.routes import document_routes
from app.services.file_write_lock import FileWriteBusy, NoFileWriteLock, PgAdvisoryFileLock
from tests.utils.test_replace_not_accumulate import FakeRow, FakeStore, V1, V2

FID = "f02"
PG_DSN = os.environ.get("RAG_TEST_PG_DSN")


class SlowStore(FakeStore):
    """Insert yields to the event loop part-way, so two writers really interleave."""

    async def aadd_documents(self, docs, ids=None, executor=None):
        self.calls.append("insert")
        await asyncio.sleep(0.05)
        for d in docs:
            self.rows.append(FakeRow(ids[0], d.page_content, d.metadata))
        return ids


class InProcessFileLock:
    """Same `hold` contract as PgAdvisoryFileLock, one asyncio.Lock per file_id."""

    def __init__(self):
        self._locks = {}

    @asynccontextmanager
    async def hold(self, file_id, stop_waiting=None, max_wait=0):
        lock = self._locks.setdefault(file_id, asyncio.Lock())
        async with lock:
            yield


def _seed_v0(store):
    store.rows.append(FakeRow(FID, "V0 original. Limit 100 EUR.",
                              {"user_id": "userA", "tenant_id": "tenantA", "filename": "v0.txt"}))


async def _race(pool):
    async def write(text, name):
        return await document_routes.store_data_in_vector_db(
            [Document(page_content=text, metadata={})], FID, "userA",
            executor=pool, filename=name, tenant_id="tenantA", replace=True,
        )

    return await asyncio.gather(write(V1, "A.txt"), write(V2, "B.txt"))


def _run_race(monkeypatch, lock, finally_=None):
    store = SlowStore()
    _seed_v0(store)
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0, raising=False)
    monkeypatch.setattr(document_routes, "file_write_lock", lock)

    async def main(pool):
        try:
            return await _race(pool)
        finally:
            if finally_ is not None:
                await finally_()

    with ThreadPoolExecutor(4) as pool:
        results = asyncio.run(main(pool))
    return store, results


def test_without_a_lock_the_race_leaves_both_versions(monkeypatch):
    """The measured defect, reproduced so the fixture is proven able to show it."""
    store, results = _run_race(monkeypatch, NoFileWriteLock())
    assert store.filenames_for(FID) == ["A.txt", "B.txt"]
    statuses = sorted(r["replacement"]["status"] for r in results)
    assert statuses == ["complete", "incomplete"]


def test_with_the_lock_the_last_writer_wins_and_says_so_truthfully(monkeypatch):
    store, results = _run_race(monkeypatch, InProcessFileLock())

    names = store.filenames_for(FID)
    assert len(names) == 1, f"two versions retrievable after a serialized replace: {names}"
    assert "v0.txt" not in names
    # Both replacements are complete: the second captured and removed the FIRST's rows.
    assert [r["replacement"]["status"] for r in results] == ["complete", "complete"]
    first, second = results
    assert second["replacement"]["superseded_rows"] == len(first["ids"])
    # The calls did not interleave: each writer's capture/insert/delete ran as a block.
    calls = store.calls
    half = len(calls) // 2
    assert calls[:half] == calls[half:] == ["capture", "capture", "insert", "delete"], calls


def test_a_caller_who_leaves_while_waiting_writes_nothing(monkeypatch):
    """F04 and F02 together: waiting for the lock is still waiting."""

    class NeverFree:
        @asynccontextmanager
        async def hold(self, file_id, stop_waiting=None, max_wait=0):
            await stop_waiting()
            raise AssertionError("stop_waiting should have raised for a departed caller")
            yield  # pragma: no cover

    store = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "file_write_lock", NeverFree())

    async def gone():
        return False

    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA",
            executor=pool, still_wanted=gone,
        ))
    assert err.value.status_code == document_routes.CALLER_GONE_STATUS
    assert store.calls == [], "something was captured or written for a departed caller"


def test_a_write_that_cannot_get_the_lock_in_time_is_a_409_with_nothing_stored(monkeypatch):
    class Busy:
        @asynccontextmanager
        async def hold(self, file_id, stop_waiting=None, max_wait=0):
            raise FileWriteBusy(file_id, 120)
            yield  # pragma: no cover

    store = FakeStore()
    monkeypatch.setattr(document_routes, "vector_store", store)
    monkeypatch.setattr(document_routes, "file_write_lock", Busy())
    with pytest.raises(HTTPException) as err, ThreadPoolExecutor(2) as pool:
        asyncio.run(document_routes.store_data_in_vector_db(
            [Document(page_content=V1, metadata={})], FID, "userA", executor=pool,
        ))
    assert err.value.status_code == 409
    assert "nothing from this request was stored" in err.value.detail
    assert store.calls == []


# ---------------------------------------------------------------------------
# The real advisory lock -- needs Postgres
# ---------------------------------------------------------------------------

# RAG_TEST_PG_REQUIRED (set in CI, which provides a Postgres service) makes a missing
# database a FAILURE rather than a skip: a skip there would read as "covered" while the
# real lock never ran.
needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no isolated Postgres for the real advisory lock",
)


def _pg_lock(**kw):
    import asyncpg

    state = {}

    async def get_pool():
        if "pool" not in state:
            state["pool"] = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=4)
        return state["pool"]

    return PgAdvisoryFileLock(get_pool, **kw), state


async def _close(state):
    if "pool" in state:
        await state["pool"].close()


@needs_pg
def test_real_lock_serializes_one_file_and_not_two():
    lock, state = _pg_lock(first_poll=0.01, max_poll=0.02)
    order = []

    async def writer(file_id, tag):
        async with lock.hold(file_id, max_wait=10):
            order.append(f"{tag}-in")
            await asyncio.sleep(0.2)
            order.append(f"{tag}-out")

    async def main():
        try:
            await asyncio.gather(writer("same", "A"), writer("same", "B"))
            same = list(order)
            order.clear()
            await asyncio.gather(writer("one", "C"), writer("two", "D"))
            return same, list(order)
        finally:
            await _close(state)

    same, different = asyncio.run(main())
    assert same in (["A-in", "A-out", "B-in", "B-out"], ["B-in", "B-out", "A-in", "A-out"]), same
    assert different[:2] in (["C-in", "D-in"], ["D-in", "C-in"]), (
        f"two different files were serialized: {different}"
    )


@needs_pg
def test_real_lock_is_released_when_the_write_fails_or_is_cancelled():
    lock, state = _pg_lock(first_poll=0.01, max_poll=0.02)

    async def main():
        try:
            with pytest.raises(RuntimeError):
                async with lock.hold("f", max_wait=5):
                    raise RuntimeError("insert failed")

            async def held_forever():
                async with lock.hold("f", max_wait=5):
                    await asyncio.sleep(60)

            task = asyncio.create_task(held_forever())
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            async with lock.hold("f", max_wait=1):  # must not wait: nobody holds it
                return True
        finally:
            await _close(state)

    assert asyncio.run(main())


@needs_pg
def test_real_lock_gives_up_after_max_wait_and_when_the_caller_leaves():
    lock, state = _pg_lock(first_poll=0.01, max_poll=0.05)

    class Gone(Exception):
        pass

    async def main():
        try:
            async with lock.hold("f", max_wait=5):
                with pytest.raises(FileWriteBusy):
                    async with lock.hold("f", max_wait=0.3):
                        pass

                async def caller_left():
                    raise Gone()

                with pytest.raises(Gone):
                    async with lock.hold("f", stop_waiting=caller_left, max_wait=5):
                        pass
            return True
        finally:
            await _close(state)

    assert asyncio.run(main())


@needs_pg
def test_real_lock_route_race_last_writer_wins(monkeypatch):
    lock, state = _pg_lock(first_poll=0.01, max_poll=0.02)
    store, results = _run_race(monkeypatch, lock, finally_=lambda: _close(state))
    names = store.filenames_for(FID)
    assert len(names) == 1, names
    assert [r["replacement"]["status"] for r in results] == ["complete", "complete"]
