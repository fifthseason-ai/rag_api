"""P06-4 residual (a): a caller from ANOTHER TENANT, holding a valid NON-EMPTY entitlement,
gets neither content nor restricted metadata from any route that can return either.

WHY THIS IS NOT ALREADY COVERED
-------------------------------
`test_entitlement_fused.py` proves the EMPTY-entitlement case: a caller entitled to nothing
gets `[]` from every retrieval arm on the three /query routes. That is a different input
class. A caller entitled to nothing is filtered out by almost any filter that runs at all;
a caller entitled to SOMETHING is only filtered out by a filter that compares the right
field to the right value. A bug that filtered on the wrong key, or that trusted the path
`entity_id` over the token, would pass the fused suite and fail here.

It is also a different route set. Fused covers /query, /query/{entity_id} and
/query_multiple. The routes that leak METADATA rather than content -- GET /ids,
GET /documents, GET /documents/{id}/context -- are covered here, plus
POST /summarize/{entity_id}, which reads an entity's documents.

WHAT THIS SUITE CAN PROVE, AND WHAT IT CANNOT
---------------------------------------------
PROVEN HERE: given entity ids that are distinct between the two tenants, every listed route
refuses or empties, and does so WITHOUT an existence oracle -- a foreign id is answered
byte-identically to an id that does not exist at all, so the response cannot be used to
enumerate another tenant's files or entities.

**NOT PROVABLE FROM INSIDE rag_api, AND DELIBERATELY NOT IMPLIED BY ANY GREEN BELOW:**
retrieval is scoped by `user_id` (the entity), never by tenant. Measured at 5816e13:

  * every query route filters on `ent["entity_ids"]` from the caller's token, and
    `_authorized_only` (document_routes.py:1217) re-filters the result to the same set;
  * `tenant_id` IS stored on every embed path, in cmetadata (document_routes.py:1825);
  * the ONLY tenant predicate in the codebase is in `get_row_uuids`
    (extended_pg_vector.py:164-166), which serves the delete/row-lookup path.
    **No retrieval path consults it.**

So tenant isolation holds if and only if an entity id is never shared between tenants, and
that invariant is established by the PRODUCER of entity ids, not by rag_api. This suite
cannot test it: it can only choose distinct ids and demonstrate the filter works when they
are distinct. Choosing the ids is the thing being assumed.

Routed as its own card rather than absorbed into this green -- see
FILES-DEV/XTENANT-INVARIANT-FOR-CORE-20260923T072400Z.md. An isolation claim that silently depends on someone
else's untested invariant is the shape where every service looks correct and the system is
not.

Worth stating for whoever picks that card up: the data needed to enforce this inside
rag_api is already on every row. A retrieval-side tenant predicate would make isolation
independent of the entity-uniqueness invariant instead of contingent on it. Whether any
legitimate flow shares an entity across tenants is a Core question, which is why this suite
names the option and does not take it.

NON-VACUITY
-----------
`test_the_foreign_tenant_can_still_see_its_own_row` is the positive control. Every
assertion in this file is a negative, and negatives are all satisfied at once by a fixture
that returns nothing to anybody, a broken token, or a store that failed to load. If that
control reddens, no other green in this file means anything.

It has already earned that: it caught the fixture storing rows under synthetic ids
(`xt0`/`xt1`) while `GET /ids` returns the `custom_id`, which in the real ingest path IS the
file_id. `test_ids_never_lists_the_other_tenants_file` was green because FILE_A was never in
the id space at all -- a negative passing for the wrong reason, invisible from its own green.

MUTATION RESULTS, INCLUDING THE TWO THAT SURVIVED
-------------------------------------------------
Measured against real pgvector, source restored from the index between each:

  postfilter removed (`_authorized_only` -> identity)   -> 10 passed   SURVIVED
  arm filter removed (entity predicate off /query)      -> 10 passed   SURVIVED
  BOTH removed together                                 ->  2 failed, 8 passed

**Neither layer is individually necessary for these routes, and that is the design rather
than a hole in this suite.** The two defences are redundant on purpose (D-KSPT-1 plus
F-ENTITLEMENT-FUSED), so removing either leaves the other holding and NO single-mutation
control applied at a ROUTE can redden. Calling that a suite defect would be wrong; treating
the greens as proof that each layer is load-bearing would also be wrong. What the double
mutation establishes is the thing that was actually in doubt: this suite detects a total
failure of the guard instead of passing by construction.

Per-LAYER necessity has to be proven where a layer is the only thing running, which is at the
producer, not the route: `test_entitlement_fused.test_each_arm_applies_the_entitlement_filter`
calls the retrieval producer directly, so a dropped filter on either arm reddens there. That
division is deliberate and it is why this file does not attempt it.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it; RAG_TEST_PG_REQUIRED=1 turns
"no DSN" from a skip into an error, so the counts prove these ran.
"""
import datetime
import os

import jwt
import psycopg2
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the cross-tenant isolation path",
)

_SECRET = "test-secret-cross-tenant"
_COLLECTION = "cross_tenant_isolation"

# Tenant A's content. The marker appears ONLY in tenant A's rows, so any leak is
# unambiguous rather than a coincidence of similar wording.
MARKER = "vorthaxil9"
A_TEXT = f"{MARKER} {MARKER} acme merger 2026 confidential board memo"
B_TEXT = "routine tenant B expense note"
NO_MATCH = "phrase that matches nothing at all"

TENANT_A, TENANT_B = "tenant-alpha", "tenant-beta"
ENT_A, ENT_B = "ent-alpha", "ent-beta"
FILE_A, FILE_B = "file-alpha-confidential", "file-beta-routine"
ABSENT_FILE = "file-that-was-never-stored"
ABSENT_ENT = "ent-that-never-existed"

# Tenant A's row is the NEAREST vector and the STRONGEST keyword match for the marker
# query, so a missing filter on either arm surfaces it rather than hiding behind ranking.
_VECTORS = {
    MARKER: [1.0, 0.0, 0.0],
    NO_MATCH: [0.0, 0.0, 1.0],
    A_TEXT: [1.0, 0.0, 0.0],
    B_TEXT: [0.0, 1.0, 0.0],
}


class _TableEmb:
    """Fixed text -> vector table. An unknown text raises: fixture drift must be loud."""

    def embed_documents(self, texts):
        return [list(_VECTORS[t]) for t in texts]

    def embed_query(self, text):
        return list(_VECTORS[text])


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


def _tok(tenant_id, entity_ids, act=("read", "write")):
    """A REAL token for a REAL tenant. The point of this suite is that the caller is
    legitimate -- valid signature, valid tenant, non-empty entitlement -- and still gets
    nothing belonging to the other tenant."""
    os.environ["JWT_SECRET"] = _SECRET
    payload = {
        "id": f"caller-of-{tenant_id}",
        "tid": tenant_id,
        "ent": list(entity_ids),
        "act": list(act),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, _SECRET, algorithm='HS256')}"}


def _real_post_init(self):
    """Run the REAL pgvector init that conftest no-ops for the session."""
    from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
    if self.create_extension:
        self.create_vector_extension()
    EmbeddingStore, CollectionStore = _get_embedding_collection_store(
        self._embedding_length, use_jsonb=self.use_jsonb
    )
    self.CollectionStore = CollectionStore
    self.EmbeddingStore = EmbeddingStore
    self.create_tables_if_not_exists()
    self.create_collection()


@pytest.fixture()
def env(monkeypatch):
    """Real pgvector holding BOTH tenants' rows in one collection.

    One collection is deliberate. Separate collections would make the test pass through
    physical separation that production does not have, and the guard under test would never
    be exercised.
    """
    from app.routes import document_routes as dr
    from app.services import database as db
    from app.services.database import PSQLDatabase
    from app.services.vector_store.factory import get_vector_store
    from main import app

    raw = _raw_dsn()
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE"
        )
        c.commit()

    os.environ["JWT_SECRET"] = _SECRET
    store = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="sync")
    _real_post_init(store)
    rows = [
        (A_TEXT, FILE_A, ENT_A, TENANT_A),
        (B_TEXT, FILE_B, ENT_B, TENANT_B),
    ]
    store.add_documents(
        [
            Document(
                page_content=t,
                metadata={"file_id": f, "user_id": u, "tenant_id": tid},
            )
            for t, f, u, tid in rows
        ],
        # custom_id IS the file_id in the real ingest path, and GET /ids returns custom_id.
        # Using synthetic row ids here made test_ids_never_lists_the_other_tenants_file pass
        # for the wrong reason: FILE_A was never in the id space at all.
        ids=[f for _, f, _, _ in rows],
    )
    with psycopg2.connect(raw) as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE langchain_pg_embedding "
            "ADD COLUMN IF NOT EXISTS document_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', document)) STORED"
        )
        c.commit()

    PSQLDatabase.pool = None
    monkeypatch.setattr(db, "DSN", raw, raising=True)

    astore = get_vector_store(_sqlalchemy_dsn(), _TableEmb(), _COLLECTION, mode="async")
    _real_post_init(astore)
    monkeypatch.setattr(dr, "vector_store", astore)
    monkeypatch.setattr("app.config.vector_store", astore, raising=False)
    monkeypatch.setattr(dr, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(dr, "RERANK_ENABLED", False)

    # NOT `with TestClient(app)`. Entering the context runs the app lifespan, and
    # LEAVING it calls app.state.thread_pool.shutdown(wait=True) on the module-level
    # app object shared by the whole test session -- so every later test that needs the
    # pool dies with "cannot schedule new futures after shutdown". Measured: with a
    # real DB this took the full suite to 158 failed / 22 errors while every per-file
    # run of this file stayed green, because the poisoning only reaches tests that run
    # AFTER it in the same process.
    #
    # Same construction as test_entitlement_fused and test_ids_entitlement_scope: build
    # the client directly and make sure the shared pool exists.
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    yield TestClient(app)


# The caller under test throughout: tenant B, legitimately entitled to ent-beta only.
def _b():
    return _tok(TENANT_B, [ENT_B])


# ---------------------------------------------------------------------------
# POSITIVE CONTROL — read this first. Every other test here is a negative.
# ---------------------------------------------------------------------------


@needs_pg
def test_the_foreign_tenant_can_still_see_its_own_row(env):
    """NON-VACUITY GUARD. If this reddens, every negative below is passing for the wrong
    reason -- an empty store, a rejected token, a fixture that never loaded."""
    r = env.post("/query", json={"query": B_TEXT, "file_id": FILE_B, "k": 10}, headers=_b())
    assert r.status_code == 200, r.text

    ids = env.get("/ids", headers=_b())
    assert ids.status_code == 200, ids.text
    assert FILE_B in ids.json(), (
        "tenant B cannot see its OWN file, so the negatives in this file prove nothing: "
        "they would be satisfied by a store that returns nothing to anybody. Got %r"
        % (ids.json(),))


# ---------------------------------------------------------------------------
# METADATA ROUTES — the ones that leak a MAP of another tenant rather than content
# ---------------------------------------------------------------------------


@needs_pg
def test_ids_never_lists_the_other_tenants_file(env):
    r = env.get("/ids", headers=_b())
    assert r.status_code == 200, r.text
    assert FILE_A not in r.json(), (
        "GET /ids disclosed tenant A's file id %r to tenant B. A file id is the argument "
        "every other route takes and is frequently the customer's own document name."
        % FILE_A)


@needs_pg
def test_documents_refuses_a_foreign_id_exactly_as_it_refuses_an_absent_one(env):
    """NO EXISTENCE ORACLE. Different answers here would let tenant B enumerate tenant A's
    files by diffing responses, which discloses the thing the filter exists to hide."""
    foreign = env.get("/documents", params={"ids": [FILE_A]}, headers=_b())
    absent = env.get("/documents", params={"ids": [ABSENT_FILE]}, headers=_b())

    assert foreign.status_code == 404, foreign.text
    assert (foreign.status_code, foreign.json()) == (absent.status_code, absent.json()), (
        "a FOREIGN file id is answered differently from an ABSENT one: %r vs %r. The "
        "difference IS the disclosure." % (foreign.json(), absent.json()))


@needs_pg
def test_context_refuses_a_foreign_id_exactly_as_it_refuses_an_absent_one(env):
    foreign = env.get(f"/documents/{FILE_A}/context", headers=_b())
    absent = env.get(f"/documents/{ABSENT_FILE}/context", headers=_b())

    assert foreign.status_code == 404, foreign.text
    assert (foreign.status_code, foreign.json()) == (absent.status_code, absent.json()), (
        "a FOREIGN file id is answered differently from an ABSENT one on /context: "
        "%r vs %r" % (foreign.json(), absent.json()))


@needs_pg
def test_a_mixed_request_discloses_nothing_about_the_foreign_half(env):
    """Asking for one own id and one foreign id must not confirm the foreign one exists."""
    mixed = env.get("/documents", params={"ids": [FILE_B, FILE_A]}, headers=_b())
    control = env.get("/documents", params={"ids": [FILE_B, ABSENT_FILE]}, headers=_b())

    assert (mixed.status_code, mixed.json()) == (control.status_code, control.json()), (
        "own+foreign is answered differently from own+absent: %r vs %r"
        % (mixed.json(), control.json()))
    assert A_TEXT not in mixed.text and MARKER not in mixed.text


# ---------------------------------------------------------------------------
# CONTENT ROUTES — with a NON-EMPTY entitlement, which is what makes this distinct
# ---------------------------------------------------------------------------


@needs_pg
def test_query_returns_nothing_of_the_other_tenant(env):
    """Tenant A's row is both the nearest vector and the only keyword match for MARKER."""
    r = env.post("/query", json={"query": MARKER, "file_id": FILE_A, "k": 10}, headers=_b())
    assert r.status_code == 200, r.text
    assert r.json() == [], (
        "/query returned tenant A content to tenant B: %r" % (r.json(),))
    assert MARKER not in r.text


@needs_pg
def test_query_by_entity_refuses_a_foreign_entity_without_confirming_it_exists(env):
    """The path entity_id is a FILTER, never an authority. Asking for tenant A's entity
    must be answered the same as asking for one that never existed."""
    foreign = env.post(f"/query/{ENT_A}", json={"query": MARKER, "k": 10}, headers=_b())
    absent = env.post(f"/query/{ABSENT_ENT}", json={"query": MARKER, "k": 10}, headers=_b())

    assert foreign.status_code == 403, foreign.text
    assert (foreign.status_code, foreign.json()) == (absent.status_code, absent.json()), (
        "a FOREIGN entity id is answered differently from an ABSENT one: %r vs %r -- "
        "tenant B can enumerate tenant A's entities by diffing" % (foreign.json(), absent.json()))
    assert MARKER not in foreign.text


@needs_pg
def test_query_multiple_returns_nothing_for_a_foreign_file(env):
    r = env.post(
        "/query_multiple",
        json={"query": MARKER, "file_ids": [FILE_A], "k": 10},
        headers=_b(),
    )
    assert r.status_code == 200, r.text
    assert r.json() == [], (
        "/query_multiple returned tenant A content for an explicitly named foreign "
        "file_id: %r" % (r.json(),))
    assert MARKER not in r.text


@needs_pg
def test_summarize_refuses_a_foreign_entity(env):
    r = env.post(
        f"/summarize/{ENT_A}",
        data={"file_id": FILE_A, "knowledge_id": ENT_B},
        headers=_b(),
    )
    assert r.status_code in (403, 503), r.text
    assert MARKER not in r.text, "summarize echoed tenant A content in its refusal"


# ---------------------------------------------------------------------------
# EQUALITY OF THE REFUSAL — a query that matches nothing must look the same
# ---------------------------------------------------------------------------


@needs_pg
def test_a_foreign_file_is_indistinguishable_from_an_absent_one(env):
    """The strongest form of the no-oracle property: naming ANOTHER TENANT'S file must be
    answered byte-identically to naming a file that does not exist. Otherwise the empty
    answer still carries one bit: "that file is real"."""
    foreign = env.post("/query", json={"query": MARKER, "file_id": FILE_A, "k": 10}, headers=_b())
    absent = env.post("/query", json={"query": MARKER, "file_id": ABSENT_FILE, "k": 10}, headers=_b())

    assert (foreign.status_code, foreign.json()) == (
        absent.status_code, absent.json()), (
        "naming another tenant's file answers differently from naming an absent one: "
        "%r vs %r" % (foreign.json(), absent.json()))
