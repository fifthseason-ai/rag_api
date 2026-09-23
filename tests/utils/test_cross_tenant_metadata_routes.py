"""P06-4 residual (a), HERMETIC HALF: the metadata routes disclose nothing about another
tenant's files — not their content, not their existence.

WHY THIS HALF IS HERMETIC AND THE OTHER HALF IS NOT
---------------------------------------------------
`GET /documents` and `GET /documents/{id}/context` do not retrieve. They fetch by id and
then filter IN PYTHON on `d.metadata["user_id"] in ent["entity_ids"]`
(document_routes.py:815-818 and :3143-3146). That filter is the whole guard, so a store
double exercises it exactly as production does and no database is involved.

The retrieval half — /query, /query/{entity_id}, /query_multiple — needs real pgvector,
because there the guard is split between the arm filters and `_authorized_only` and a
stubbed retrieval cannot observe a leak inside the path it stubs. That half lives in
`test_cross_tenant_query_isolation.py`.

Splitting them this way is deliberate: this file runs in the no-DB suite and in CI, which
is where most people will actually see it go red.

WHAT MAKES THE DOUBLE HONEST
----------------------------
`ForeignRowStore.get_documents_by_ids` returns the FOREIGN row when asked for it. If the
double filtered by owner itself, the route's filter would never run and every test here
would pass with the guard deleted — the defect this repo has already hit once, recorded in
`test_ids_entitlement_scope.StoreWithoutScoping`: "a double that accidentally satisfies the
capability check cannot test the capability check".
`test_the_double_really_hands_over_the_foreign_row` pins that directly.

SCOPE, stated so no green here is read as more than it is: this proves the ROUTE filter
given distinct entity ids. It does not prove tenant isolation, which rests on an invariant
rag_api cannot see — see FILES-DEV/XTENANT-INVARIANT-FOR-CORE-20260923T072400Z.md.
"""
import datetime
import os

import jwt
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.services.vector_store.async_pg_vector import AsyncPgVector

_SECRET = "test-secret-xtenant-metadata"

TENANT_A, TENANT_B = "tenant-alpha", "tenant-beta"
ENT_A, ENT_B = "ent-alpha", "ent-beta"
FILE_A, FILE_B = "file-alpha-confidential", "file-beta-routine"
ABSENT_FILE = "file-that-was-never-stored"

A_TEXT = "acme merger 2026 confidential board memo"
B_TEXT = "routine tenant B expense note"


def _tok(tenant_id, entity_ids, act=("read", "write")):
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": f"caller-of-{tenant_id}",
        "tid": tenant_id,
        "ent": list(entity_ids),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _b():
    """Tenant B: a legitimate caller with a valid NON-EMPTY entitlement."""
    return _tok(TENANT_B, [ENT_B])


class ForeignRowStore(AsyncPgVector):
    """Holds both tenants' rows and hands back WHATEVER IS ASKED FOR, unfiltered.

    Subclasses the real class so `isinstance(vector_store, AsyncPgVector)` takes the same
    branch production takes. `super().__init__` is deliberately not called — no engine, no
    connection.
    """

    def __init__(self):
        self._bind = None
        self.rows = {
            FILE_A: Document(
                page_content=A_TEXT,
                metadata={"file_id": FILE_A, "user_id": ENT_A, "tenant_id": TENANT_A},
            ),
            FILE_B: Document(
                page_content=B_TEXT,
                metadata={"file_id": FILE_B, "user_id": ENT_B, "tenant_id": TENANT_B},
            ),
        }
        self.handed_over = []

    async def get_documents_by_ids(self, ids, executor=None):
        found = [self.rows[i] for i in ids if i in self.rows]
        self.handed_over = [d.metadata["file_id"] for d in found]
        return found


@pytest.fixture()
def env(monkeypatch):
    """TestClient WITHOUT the context manager, matching test_ids_entitlement_scope.

    Entering it would run the app lifespan, which opens a real asyncpg pool — this suite is
    hermetic by design and a Postgres dependency here would defeat the point of splitting
    it from the pgvector half.
    """
    from app.routes import document_routes as dr
    from main import app

    os.environ["JWT_SECRET"] = _SECRET
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    store = ForeignRowStore()
    monkeypatch.setattr(dr, "vector_store", store)
    monkeypatch.setattr("app.config.vector_store", store, raising=False)
    client = TestClient(app)
    client.store = store
    return client


# ---------------------------------------------------------------------------
# NON-VACUITY — both directions. Read these before trusting anything below.
# ---------------------------------------------------------------------------


def test_the_foreign_tenant_can_still_read_its_own_document(env):
    """POSITIVE CONTROL. Every other assertion here is a negative, and negatives are all
    satisfied at once by a store that returns nothing or a token that is rejected."""
    r = env.get("/documents", params={"ids": [FILE_B]}, headers=_b())
    assert r.status_code == 200, r.text
    assert B_TEXT in r.text


def test_the_double_really_hands_over_the_foreign_row(env):
    """The guard under test is the ROUTE's filter. If the store double filtered by owner
    itself, every negative in this file would pass with the route's filter deleted."""
    env.get("/documents", params={"ids": [FILE_A]}, headers=_b())
    assert env.store.handed_over == [FILE_A], (
        "the double did not hand the foreign row to the route (handed_over=%r), so the "
        "route's entitlement filter was never exercised and the greens below are vacuous"
        % (env.store.handed_over,))


# ---------------------------------------------------------------------------
# NO CONTENT, AND NO EXISTENCE ORACLE
# ---------------------------------------------------------------------------


def test_documents_refuses_a_foreign_id_exactly_as_an_absent_one(env):
    foreign = env.get("/documents", params={"ids": [FILE_A]}, headers=_b())
    absent = env.get("/documents", params={"ids": [ABSENT_FILE]}, headers=_b())

    assert foreign.status_code == 404, foreign.text
    assert A_TEXT not in foreign.text
    assert (foreign.status_code, foreign.json()) == (absent.status_code, absent.json()), (
        "a FOREIGN file id is answered differently from an ABSENT one: %r vs %r. The "
        "difference is itself the disclosure — it confirms the file exists."
        % (foreign.json(), absent.json()))


def test_context_refuses_a_foreign_id_exactly_as_an_absent_one(env):
    foreign = env.get(f"/documents/{FILE_A}/context", headers=_b())
    absent = env.get(f"/documents/{ABSENT_FILE}/context", headers=_b())

    assert foreign.status_code == 404, foreign.text
    assert A_TEXT not in foreign.text
    assert (foreign.status_code, foreign.json()) == (absent.status_code, absent.json()), (
        "a FOREIGN file id is answered differently from an ABSENT one on /context: %r vs %r"
        % (foreign.json(), absent.json()))


def test_a_mixed_request_discloses_nothing_about_the_foreign_half(env):
    """Own id + foreign id must be answered exactly as own id + absent id. Otherwise a
    caller enumerates the other tenant one id at a time while holding a valid request."""
    mixed = env.get("/documents", params={"ids": [FILE_B, FILE_A]}, headers=_b())
    control = env.get("/documents", params={"ids": [FILE_B, ABSENT_FILE]}, headers=_b())

    assert (mixed.status_code, mixed.json()) == (control.status_code, control.json()), (
        "own+foreign is answered differently from own+absent: %r vs %r"
        % (mixed.json(), control.json()))
    assert A_TEXT not in mixed.text


def test_the_refusal_body_never_echoes_the_requested_foreign_id(env):
    """A 404 that quotes the id back is still a 404, but it confirms the caller's guess
    reached a real lookup. Cheap to get wrong in an error-message change."""
    r = env.get(f"/documents/{FILE_A}/context", headers=_b())
    assert r.status_code == 404
    assert FILE_A not in r.text, (
        "the refusal echoed the foreign file id: %r" % r.text)
