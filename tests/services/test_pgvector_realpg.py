"""F-PGVECTOR-REALPG: the entitlement-scoping and rollback primitives of the DEFAULT
vector store (ExtendedPgVector / AsyncPgVector) proven on a REAL Postgres+pgvector.

F-COVERAGE-SWEEP measured these at zero runtime coverage: delete_rows_by_uuid (F04
rollback), _delete_multiple (DELETE /documents), get_filtered_ids (its ownership
pre-check), get_ids_for_entities (GET /ids scoping), get_documents_by_ids.

THE FIXTURE CAN EXPRESS EVERY FAILURE: two entities share a file_id ("shared"), so a
filter that keys on file_id alone crosses entities; the owner's copy has TWO chunks, so a
rollback that deletes by file_id instead of by row uuid removes one row too many.

Needs a Postgres with pgvector. RAG_TEST_PG_DSN selects it (CI provides a service);
RAG_TEST_PG_REQUIRED=1 turns "no DSN" from a skip into an error so these provably ran.
"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import psycopg2
import pytest
from langchain_core.documents import Document

PG_DSN = os.environ.get("RAG_TEST_PG_DSN")

needs_pg = pytest.mark.skipif(
    not PG_DSN and not os.environ.get("RAG_TEST_PG_REQUIRED"),
    reason="RAG_TEST_PG_DSN not set: no pgvector for the vector-store primitives",
)

_COLLECTION = "pgvector_realpg"
A, B = "uA", "uB"


class _Emb:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def _sqlalchemy_dsn():
    d = PG_DSN
    if d and d.startswith("postgresql://"):
        d = d.replace("postgresql://", "postgresql+psycopg2://", 1)
    return d


def _raw_dsn():
    return (PG_DSN or "").replace("postgresql+psycopg2://", "postgresql://")


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


# (text, file_id, owner)
ROWS = [
    ("A shared chunk one", "shared", A),
    ("A shared chunk two", "shared", A),
    ("A private file", "fa", A),
    ("B shared chunk", "shared", B),
    ("B private file", "fb", B),
]


def _rows():
    """Every row in the table as (file_id, owner, text), sorted: the ground truth."""
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT custom_id, cmetadata->>'user_id', document FROM langchain_pg_embedding")
        return sorted(cur.fetchall())


@pytest.fixture()
def store():
    from app.services.vector_store.factory import get_vector_store

    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding, langchain_pg_collection CASCADE")
        c.commit()
    s = get_vector_store(_sqlalchemy_dsn(), _Emb(), _COLLECTION, mode="sync")
    _real_post_init(s)
    s.add_documents(
        [Document(page_content=t, metadata={"file_id": f, "user_id": u, "tenant_id": "t"})
         for t, f, u in ROWS],
        ids=[f for _t, f, _u in ROWS],
    )
    assert len(_rows()) == len(ROWS)
    return s


@pytest.fixture()
def astore(store):
    from app.services.vector_store.factory import get_vector_store
    a = get_vector_store(_sqlalchemy_dsn(), _Emb(), _COLLECTION, mode="async")
    _real_post_init(a)
    return a


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---- get_ids_for_entities (GET /ids) -------------------------------------------------

@needs_pg
def test_ids_for_entities_returns_only_the_entitys_files(store):
    assert sorted(store.get_ids_for_entities([A])) == ["fa", "shared", "shared"]
    assert sorted(store.get_ids_for_entities({B})) == ["fb", "shared"]  # a set, as the route passes
    assert store.get_ids_for_entities(["nobody"]) == []


@needs_pg
def test_ids_for_entities_empty_entitlement_discloses_nothing(store):
    assert store.get_ids_for_entities([]) == []
    assert store.get_ids_for_entities(set()) == []


# ---- get_filtered_ids (DELETE /documents ownership pre-check) -------------------------

@needs_pg
def test_filtered_ids_never_reports_another_entitys_file_as_existing(store):
    got = store.get_filtered_ids(["shared", "fb"], user_id=A)
    assert sorted(got) == ["shared", "shared"], got  # fb is B's: not "existing" for A
    assert store.get_filtered_ids(["fb"], user_id=A) == []


# ---- _delete_multiple (DELETE /documents) ---------------------------------------------

@needs_pg
def test_delete_by_file_id_is_scoped_to_the_entity(store):
    store._delete_multiple(ids=["shared"], user_id=A)
    assert _rows() == sorted([
        ("fa", A, "A private file"),
        ("shared", B, "B shared chunk"),
        ("fb", B, "B private file"),
    ])


@needs_pg
def test_delete_with_empty_ids_never_crosses_entities(store):
    """Empty `ids` means NO id filter in this primitive (documented). Pinned here only for
    the property that must never break: it stays inside the entity. (What it does to the
    entity's own rows is a recorded finding for a product ruling, not asserted.)"""
    store._delete_multiple(ids=[], user_id=A)
    assert [r for r in _rows() if r[1] == B] == sorted([
        ("shared", B, "B shared chunk"), ("fb", B, "B private file")])


@needs_pg
def test_async_delete_passes_the_entity_filter_through(astore):
    """The route uses AsyncPgVector.delete / get_filtered_ids; the executor hop must not
    drop user_id."""
    pool = ThreadPoolExecutor(max_workers=1)
    existing = _run(astore.get_filtered_ids(["shared", "fb"], user_id=A, executor=pool))
    assert sorted(existing) == ["shared", "shared"]
    _run(astore.delete(ids=["shared"], user_id=A, executor=pool))
    assert ("shared", B, "B shared chunk") in _rows()
    assert not [r for r in _rows() if r[0] == "shared" and r[1] == A]


# ---- delete_rows_by_uuid (F04 / replace rollback) -------------------------------------

@needs_pg
def test_rollback_deletes_exactly_the_captured_rows(store):
    """Capture ONE of A's two 'shared' rows and roll it back: the other A row, and B's row
    with the same file_id, must survive, and the count must be exact."""
    uuids = store.get_row_uuids("shared", user_id=A)
    assert len(uuids) == 2
    with psycopg2.connect(_raw_dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT document FROM langchain_pg_embedding WHERE uuid::text = %s", (uuids[0],))
        victim = cur.fetchone()[0]

    removed = store.delete_rows_by_uuid([uuids[0]])

    assert removed == 1
    expected = sorted(r for r in [(f, u, t) for t, f, u in ROWS] if r[2] != victim)
    assert _rows() == expected


@needs_pg
def test_rollback_of_an_empty_capture_deletes_nothing(store, astore):
    assert store.delete_rows_by_uuid([]) == 0
    assert _run(astore.delete_rows_by_uuid([], executor=ThreadPoolExecutor(1))) == 0
    assert len(_rows()) == len(ROWS)


@needs_pg
def test_row_capture_is_entity_scoped(store):
    assert len(store.get_row_uuids("shared", user_id=B)) == 1
    assert len(store.get_row_uuids("shared")) == 3


# ---- get_documents_by_ids --------------------------------------------------------------

@needs_pg
def test_documents_by_ids_returns_exactly_those_files_with_metadata(store):
    docs = store.get_documents_by_ids(["fa", "fb"])
    got = sorted((d.metadata["file_id"], d.metadata["user_id"], d.page_content) for d in docs)
    assert got == [("fa", A, "A private file"), ("fb", B, "B private file")]
    assert store.get_documents_by_ids(["missing"]) == []
    assert store.get_documents_by_ids([]) == []
