"""One writer at a time per file_id (FILES-01 F02).

THE DEFECT. Measured 2026-09-21 through the real /embed route: two `replace=true` uploads
of the same file_id, sent at the same moment, both answered 200 -- and the file ended
with BOTH new versions retrievable. Each call captured the rows present before it
started (the same original rows), inserted its own version, and deleted only what it had
captured. Neither knew about the other's rows. The first to finish reported
`status: complete`, which was false; the second reported `incomplete` because the rows it
meant to delete were already gone. Replacement's whole promise -- "after this, only the
current version is retrievable" -- did not hold under concurrency.

THE RULE. A write to a file_id holds a lock for that file_id from before its captures
until after its last delete. The second writer waits, then captures the FIRST writer's
rows and supersedes them: the last writer wins, the file never holds two versions, and
never passes through zero rows. The same lock is what makes F04's undo of an abandoned
write safe: with no other writer inside the section, "rows present now that were not
present before" can only be this call's rows.

WHY A POSTGRES ADVISORY LOCK. The service runs as several processes and tasks; an
in-process lock would serialize nothing across them. The lock is taken on the asyncpg
pool (`PSQLDatabase`), NOT on the SQLAlchemy engine the inserts use: a waiting or holding
writer must never occupy the connections the holder needs in order to finish.

WHY POLL, NOT BLOCK. `pg_advisory_lock` would park a connection -- and, through the
executor, a worker thread -- for as long as the other writer takes. The lock holder needs
those threads to embed and insert, so enough waiters could starve the very writer they
are waiting for. A waiter here holds NOTHING between attempts: it borrows a connection
for one `pg_try_advisory_lock`, returns it, and sleeps.

A waiter stops waiting when its caller has gone (F04 -- a departed caller must not be
stored later), or after `max_wait` seconds, when it gives up with nothing written.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

from app.config import logger

#: First key of the two-key advisory lock: a namespace, so this lock can never collide
#: with an advisory lock taken for another purpose on the same database. ASCII "FILE".
LOCK_NAMESPACE = 0x46494C45

_TRY_LOCK = "SELECT pg_try_advisory_lock($1, hashtext($2))"
_UNLOCK = "SELECT pg_advisory_unlock($1, hashtext($2))"


class FileWriteBusy(Exception):
    """Another writer held this file_id for longer than this request would wait."""

    def __init__(self, file_id: str, waited: float):
        super().__init__(f"file {file_id!r} still being written after {waited:.0f}s")
        self.file_id = file_id
        self.waited = waited


class NoFileWriteLock:
    """For stores with no shared database to lock in (and for tests that model the
    table in memory). Serializes nothing, and says so in its name."""

    @asynccontextmanager
    async def hold(self, file_id: str, stop_waiting=None, max_wait: float = 0):
        yield


class PgAdvisoryFileLock:
    def __init__(
        self,
        get_pool: Callable[[], Awaitable],
        first_poll: float = 0.05,
        max_poll: float = 0.5,
    ):
        self._get_pool = get_pool
        self._first_poll = first_poll
        self._max_poll = max_poll

    @asynccontextmanager
    async def hold(
        self,
        file_id: str,
        stop_waiting: Optional[Callable[[], Awaitable[None]]] = None,
        max_wait: float = 120.0,
    ):
        """Hold the file_id's lock for the body of the `async with`.

        `stop_waiting` is awaited between attempts and may raise to abandon the wait
        (the route passes its caller-gone check). Raises FileWriteBusy after `max_wait`.
        """
        pool = await self._get_pool()
        loop = asyncio.get_running_loop()
        started = loop.time()
        delay = self._first_poll
        while True:
            conn = await pool.acquire()
            got = False
            try:
                got = await conn.fetchval(_TRY_LOCK, LOCK_NAMESPACE, file_id)
            finally:
                if not got:
                    await pool.release(conn)
            if got:
                break
            if stop_waiting is not None:
                await stop_waiting()
            waited = loop.time() - started
            if waited >= max_wait:
                raise FileWriteBusy(file_id, waited)
            await asyncio.sleep(min(delay, max(0.0, max_wait - waited)))
            delay = min(delay * 2, self._max_poll)

        try:
            yield
        finally:
            # The lock belongs to this connection's SESSION. If the unlock fails, the
            # connection must not go back to the pool still holding it -- every later
            # writer of this file would wait forever. Terminating the connection ends
            # the session, and Postgres releases the lock with it.
            #
            # Defense in depth, and measured as such: asyncpg 0.29's Pool.release resets
            # the session with `SELECT pg_advisory_unlock_all()`, so with this explicit
            # unlock deleted the real-Postgres tests still pass. The explicit unlock
            # releases the lock at the moment the write ends rather than relying on a
            # driver's reset query staying what it is today.
            try:
                await conn.fetchval(_UNLOCK, LOCK_NAMESPACE, file_id)
            except BaseException:
                logger.error(
                    "Could not release the write lock for %s; closing its connection so "
                    "the lock goes with it", file_id,
                )
                conn.terminate()
                raise
            finally:
                await pool.release(conn)
